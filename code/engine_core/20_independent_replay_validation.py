from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.real_process_data import load_real_process_data


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_KEYS = [("T_error", "Tension error"), ("h_error", "Thickness deviation"), ("S_error", "Flatness deviation")]
SAMPLE_PERIOD_S = 0.50


def load_suite_module():
    path = PROJECT_DIR / "15_realdata_closed_loop_suite.py"
    spec = importlib.util.spec_from_file_location("realdata_closed_loop_suite", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(path)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_name(name: str) -> str:
    name = name.replace("到", "_to_")
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    data = df.copy()
    headers = [str(c) for c in data.columns]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in data.iterrows():
        vals = []
        for col in data.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)


def measured_outputs(df: pd.DataFrame) -> np.ndarray:
    return df[["exit_tension_error", "exit_thickness_dev", "flatness_std"]].to_numpy(dtype=float)


def _segment_ids_for(df: pd.DataFrame) -> np.ndarray:
    """Segment ids from the archived-update timebase, matching the replay loop."""
    code_dir = Path(__file__).resolve().parents[1]
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from segmented_archived_timebase import derive_segmented_archived_timebase

    timebase = derive_segmented_archived_timebase(df, 0.81298828125)
    return timebase.segment_id


def smooth_median(y: np.ndarray, window: int = 7) -> np.ndarray:
    """Causal trailing median: only rows <= i enter the value at row i."""
    out = np.zeros_like(y)
    for i in range(len(y)):
        lo = max(0, i - window + 1)
        out[i] = np.median(y[lo : i + 1], axis=0)
    return out


def build_replay_innovations(
    suite, y_real: np.ndarray, p, segment_ids: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Extract output innovations that make the calibrated plant follow a real pass.

    This is the core replay trick: the calibrated plant is propagated with zero
    control, and the one-step mismatch to the measured data is stored. During
    closed-loop replay, every controller receives the same innovation sequence.
    Stateful coordinates are reset at every segment boundary, mirroring the
    replay loop, so no hidden state bridges an unsupported gap.
    """
    n = len(y_real)
    if segment_ids is None:
        segment_ids = np.zeros(n, dtype=int)
    segment_ids = np.asarray(segment_ids, dtype=int)
    x = np.r_[y_real[0], y_real[0], np.zeros(3)]
    innovations = np.zeros((max(n - 1, 1), 3), dtype=float)
    hidden_trace = np.zeros((n, 3), dtype=float)
    for k in range(n - 1):
        if k == 0 or segment_ids[k] != segment_ids[k - 1]:
            x = np.r_[y_real[k], y_real[k], np.zeros(3)]
        if segment_ids[k + 1] != segment_ids[k]:
            hidden_trace[k + 1] = x[6:9]
            continue
        pred = suite.plant_step(x, np.zeros(3), p, k, n, np.zeros(5), np.zeros(6), allocation=True)
        innovations[k] = y_real[k + 1] - pred[:3]
        x = pred
        x[:3] = y_real[k + 1]
        x[3:6] = y_real[k]
        hidden_trace[k + 1] = x[6:9]
    return innovations, hidden_trace


def make_recorded_curve(y_real: np.ndarray) -> pd.DataFrame:
    n = len(y_real)
    return pd.DataFrame(
        {
            "k": np.arange(n),
            "time_s": SAMPLE_PERIOD_S * np.arange(n),
            "phase": "",
            "T_error": y_real[:, 0],
            "h_error": y_real[:, 1],
            "S_error": y_real[:, 2],
            "T_est": y_real[:, 0],
            "h_est": y_real[:, 1],
            "S_est": y_real[:, 2],
            "u_speed": 0.0,
            "u_gap": 0.0,
            "u_shape": 0.0,
            "du_speed": 0.0,
            "du_gap": 0.0,
            "du_shape": 0.0,
            "residual_norm": 0.0,
        }
    )


def run_replay_control(
    suite,
    control_id: str,
    p,
    y_real: np.ndarray,
    innovations: np.ndarray,
    seed: int,
    filter_id: str,
    particle_count: int,
    horizon: int = 10,
    residual_rho: float = 0.10,
    innovation_gain: float = 0.82,
):
    raise RuntimeError(
        "run_replay_control is a disabled legacy implementation; use the "
        "released controller_replay.run_exposure_control path."
    )
    flags = suite.control_flags(control_id)
    n = len(y_real)
    scales = np.array([p.tension_scale, p.thickness_scale, p.flatness_scale])
    rng = np.random.default_rng(seed)

    x = np.r_[y_real[0], y_real[0], np.zeros(3)]
    x0 = x + np.r_[0.08 * scales * rng.normal(size=3), np.zeros(6)]
    active_filter = "ekf" if flags["ekf_baseline"] else filter_id if flags["filter"] else "raw"
    filt = suite.build_filter(active_filter, x0, p, n, "S5", seed + 73, particle_count)
    pid = suite.make_pid(control_id, p, SAMPLE_PERIOD_S)
    limit = 0.55
    u_min = np.array([-limit, -limit, -limit])
    u_max = np.array([limit, limit, limit])
    u = np.zeros(3)

    # Measurement playback uses high-frequency content from the real pass plus a
    # tiny seed-dependent component; the output innovation itself remains fixed.
    hf = y_real - smooth_median(y_real, window=7)
    noise_scale = np.maximum(np.std(hf, axis=0), 0.010 * scales)
    rows, solve_ms, filter_ms = [], [], []
    sat_count = 0
    constraint_count = 0

    for k in range(n):
        z = suite.h_func(x) + 0.25 * hf[k] + rng.normal(0.0, 0.015 * noise_scale)
        filter_start = time.perf_counter()
        if flags["filter"] or flags["ekf_baseline"]:
            x_est, residual = filt.step(u, z, k, allocation=bool(flags["allocation"]))
        else:
            x_est, residual = filt.step(u, z, k, allocation=True)
        if control_id == "C7":
            rho = float(np.clip(residual_rho, 0.0, 1.0))
            x_est[:3] = rho * x_est[:3] + (1.0 - rho) * z
        filter_ms.append((time.perf_counter() - filter_start) * 1000.0)

        y = suite.h_func(x_est)
        start = time.perf_counter()
        pred = suite.predict_stage_narx(x_est, p, k, n, horizon, str(flags["predictor"]), allocation=bool(flags["allocation"]))
        u_target = np.zeros(3)
        if flags["mpc"]:
            weights = suite.stage_control_weights(k, n, p, bool(flags["dynamic_weights"]))
            penalty = {
                "LSTM-MPC": 0.88,
                "DMC-KF": 1.20,
                "ATT-MPC": 0.95,
                "ROBUST-MPC": 1.65,
                "ADAPTIVE-MPC": 1.05,
            }.get(control_id, 1.0)
            u_mpc = suite.solve_mpc(
                pred,
                weights,
                p,
                u,
                u_min,
                u_max,
                bool(flags["allocation"]),
                bool(flags["dynamic_weights"]),
                penalty_scale=penalty,
            )
            if flags["pid"]:
                ff = suite.feedforward_from_prediction(pred, p)
                if control_id == "C7":
                    u_target = 0.10 * u_mpc + 0.42 * ff
                elif control_id == "C5":
                    u_target = 0.12 * u_mpc + 0.62 * ff
                else:
                    u_target = 0.10 * u_mpc + 0.56 * ff
            else:
                u_target = u_mpc
        elif str(flags["predictor"]) != "persistence" and control_id in {"C2", "C3"}:
            u_target = np.clip(suite.feedforward_from_prediction(pred, p), u_min, u_max)

        if control_id == "ADRC":
            disturbance_est = np.array(
                [
                    0.20 * x_est[6] / max(p.rollforce_scale, 1e-9),
                    -0.16 * x_est[7],
                    -0.18 * x_est[8],
                ]
            )
            u_target = np.clip(u_target + disturbance_est, u_min, u_max)
        elif control_id == "ADAPTIVE-MPC":
            adapt = 1.0 + 0.25 * np.tanh(np.linalg.norm(y / scales))
            u_target = np.clip(adapt * u_target, u_min, u_max)

        if flags["pid"]:
            u_target = np.clip(u_target + pid.step(-y), u_min, u_max)
        solve_ms.append((time.perf_counter() - start) * 1000.0)

        du_lim = np.array([0.065, 0.052, 0.065]) if flags["mpc"] else np.array([0.090, 0.075, 0.090])
        du = np.clip(u_target - u, -du_lim, du_lim)
        u = np.clip(u + du, u_min, u_max)
        sat_count += int(np.any(np.isclose(np.abs(u), limit, atol=1e-6)))

        rows.append(
            {
                "k": k,
                "time_s": SAMPLE_PERIOD_S * k,
                "phase": suite.phase_name(k, n, p),
                "T_error": x[0],
                "h_error": x[1],
                "S_error": x[2],
                "T_est": y[0],
                "h_est": y[1],
                "S_est": y[2],
                "u_speed": u[0],
                "u_gap": u[1],
                "u_shape": u[2],
                "du_speed": du[0],
                "du_gap": du[1],
                "du_shape": du[2],
                "residual_norm": residual,
            }
        )

        if k < n - 1:
            x_next = suite.plant_step(x, u, p, k, n, np.zeros(5), np.zeros(6), allocation=True)
            x_next[:3] = x_next[:3] + innovation_gain * innovations[k]
            x_next[:3] = np.clip(x_next[:3], -8.0 * scales, 8.0 * scales)
            x = x_next
            constraint_count += int(np.any(np.abs(suite.h_func(x)) > 3.0 * scales))

    diag = {
        "active_filter": active_filter,
        "sat_count": int(sat_count),
        "sat_ratio": float(sat_count / max(n, 1)),
        "constraint_count": int(constraint_count),
        "constraint_ratio": float(constraint_count / max(n - 1, 1)),
        "filter_ms_mean": float(np.mean(filter_ms)),
        "filter_ms_p95": float(np.percentile(filter_ms, 95)),
        "solve_ms_mean": float(np.mean(solve_ms)),
        "solve_ms_p95": float(np.percentile(solve_ms, 95)),
        "solve_ms_max": float(np.max(solve_ms)),
    }
    return pd.DataFrame(rows), diag


def replay_metrics(suite, df: pd.DataFrame, p, diag: dict, control_id: str, seed: int) -> dict:
    out = suite.control_metrics(df, p, diag, control_id if control_id != "RECORDED" else "C0", "S5", seed)
    out["control"] = control_id
    out["control_label"] = "Recorded production trajectory" if control_id == "RECORDED" else suite.CONTROL_LABELS[control_id]
    out["scenario"] = "REPLAY"
    out["scenario_label"] = "Independent real-pass residual replay"
    return out


def paired_tests(raw: pd.DataFrame, pairs: list[str]) -> pd.DataFrame:
    rows = []
    metric = "composite_normalized_RMS"
    c7 = raw[raw["control"] == "C7"][["pass_file", "seed", metric]].rename(columns={metric: "C7"})
    for other in pairs:
        odf = raw[raw["control"] == other][["pass_file", "seed", metric]].rename(columns={metric: other})
        merged = c7.merge(odf, on=["pass_file", "seed"], how="inner")
        if len(merged) < 3:
            continue
        diff = merged["C7"].to_numpy() - merged[other].to_numpy()
        t_stat, t_p = stats.ttest_rel(merged["C7"], merged[other])
        try:
            w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_p = np.nan
        rows.append(
            {
                "comparison": f"C7 vs {other}",
                "n_pairs": int(len(merged)),
                "C7_mean": float(merged["C7"].mean()),
                "other_mean": float(merged[other].mean()),
                "mean_diff_C7_minus_other": float(np.mean(diff)),
                "relative_improve_C7_vs_other_%": float((merged[other].mean() - merged["C7"].mean()) / max(merged[other].mean(), 1e-12) * 100.0),
                "paired_t_p": float(t_p),
                "wilcoxon_p": float(w_p),
                "cohen_dz": float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c]) and c not in {"seed"}]
    overall = raw.groupby(["control", "control_label"], as_index=False)[numeric].agg(["mean", "std"])
    overall.columns = ["_".join([x for x in col if x]) for col in overall.columns]
    overall = overall.reset_index()
    recorded = overall[overall["control"] == "RECORDED"].iloc[0]
    c0 = overall[overall["control"] == "C0"].iloc[0]
    overall["composite_improve_vs_recorded_%"] = (
        recorded["composite_normalized_RMS_mean"] - overall["composite_normalized_RMS_mean"]
    ) / max(recorded["composite_normalized_RMS_mean"], 1e-12) * 100.0
    overall["composite_improve_vs_C0_%"] = (
        c0["composite_normalized_RMS_mean"] - overall["composite_normalized_RMS_mean"]
    ) / max(c0["composite_normalized_RMS_mean"], 1e-12) * 100.0
    by_pass = raw.groupby(["pass_file", "control"], as_index=False)[numeric].mean()
    return overall, by_pass


def plot_pass_curves(curves: dict[str, pd.DataFrame], p, out_dir: Path, controls: list[str]) -> None:
    pass_safe = safe_name(p.pass_file)
    n = len(next(iter(curves.values())))
    accel_end = int(max(p.phase_window_ratio, 40.0 / max(p.rows, 80)) * n)
    decel_start = n - accel_end
    for col, label in OUTPUT_KEYS:
        plt.figure(figsize=(10.8, 4.6))
        for cid in controls:
            if cid not in curves:
                continue
            lw = 2.2 if cid in {"RECORDED", "C7"} else 1.4
            plt.plot(curves[cid]["time_s"], curves[cid][col], label=cid, linewidth=lw)
        plt.axvspan(0, SAMPLE_PERIOD_S * accel_end, color="#E8F2FF", alpha=0.55, label="accel" if col == "T_error" else None)
        plt.axvspan(SAMPLE_PERIOD_S * decel_start, SAMPLE_PERIOD_S * (n - 1), color="#FFF1E5", alpha=0.50, label="decel" if col == "T_error" else None)
        plt.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        plt.title(f"Independent replay response: {pass_safe} - {label}")
        plt.xlabel("Time (s)")
        plt.ylabel(label)
        plt.grid(True, alpha=0.32)
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"replay_curve_{pass_safe}_{col}.png", dpi=220)
        plt.close()


def plot_speed_and_recorded(df: pd.DataFrame, p, out_dir: Path) -> None:
    pass_safe = safe_name(p.pass_file)
    t = SAMPLE_PERIOD_S * np.arange(len(df))
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.2), sharex=True)
    axes[0].plot(t, df["speed_avg"], color="#1f77b4", linewidth=1.7)
    axes[0].set_ylabel("Speed")
    axes[0].set_title(f"Real pass speed and recorded errors: {pass_safe}")
    axes[0].grid(True, alpha=0.3)
    for col, label in [("exit_tension_error", "Tension"), ("exit_thickness_dev", "Thickness"), ("flatness_std", "Flatness")]:
        axes[1].plot(t, df[col], label=label, linewidth=1.4)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Recorded output")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"replay_real_pass_{pass_safe}_speed_outputs.png", dpi=220)
    plt.close(fig)


def plot_overall(overall: pd.DataFrame, out_dir: Path) -> None:
    ordered = overall.sort_values("composite_normalized_RMS_mean")
    plt.figure(figsize=(10.5, 4.4))
    plt.bar(ordered["control"], ordered["composite_normalized_RMS_mean"], yerr=ordered["composite_normalized_RMS_std"], capsize=4)
    plt.ylabel("Composite normalized RMS")
    plt.title("Independent Data2 replay validation")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "replay_overall_composite_rms.png", dpi=220)
    plt.close()

    metrics = ["T_error_IAE_mean", "h_error_IAE_mean", "S_error_IAE_mean"]
    x = np.arange(len(ordered))
    width = 0.26
    plt.figure(figsize=(11.0, 4.8))
    for i, metric in enumerate(metrics):
        plt.bar(x + (i - 1) * width, ordered[metric], width=width, label=metric.replace("_error_IAE_mean", " IAE"))
    plt.xticks(x, ordered["control"], rotation=30, ha="right")
    plt.ylabel("IAE")
    plt.title("Replay validation IAE decomposition")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "replay_overall_iae_decomposition.png", dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(PROJECT_DIR.parent.parent / "data2"))
    parser.add_argument("--out", default=str(PROJECT_DIR / "artifacts" / "independent_replay_validation"))
    parser.add_argument("--controls", default="RECORDED,C0,C2,C7,LSTM-MPC,DMC-KF,ADRC,ATT-MPC,ROBUST-MPC,ADAPTIVE-MPC")
    parser.add_argument("--selected-filter", default="ekf")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--particle-count", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--innovation-gain", type=float, default=0.82)
    parser.add_argument("--residual-rho", type=float, default=0.10)
    args = parser.parse_args()

    suite = load_suite_module()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_real_process_data(args.data_dir)
    controls = [x.strip() for x in args.controls.split(",") if x.strip()]
    run_controls = [c for c in controls if c != "RECORDED"]
    frames = dataset.frames
    passes = [suite.make_real_pass(df) for df in frames]
    pd.DataFrame([p.__dict__ for p in passes]).to_csv(out_dir / "replay_passes_used.csv", index=False)
    dataset.pass_summary.to_csv(out_dir / "replay_data2_pass_summary.csv", index=False)

    raw_rows = []
    all_curve_rows = []
    example_curves_by_pass: dict[str, dict[str, pd.DataFrame]] = {}

    for df, p in zip(frames, passes):
        print(f"Replay pass {p.pass_file}", flush=True)
        y_real = measured_outputs(df)
        innovations, hidden = build_replay_innovations(suite, y_real, p, _segment_ids_for(df))
        np.save(out_dir / f"replay_innovations_{safe_name(p.pass_file)}.npy", innovations)
        plot_speed_and_recorded(df, p, out_dir)
        recorded_curve = make_recorded_curve(y_real)
        recorded_curve["pass_file"] = p.pass_file
        example_curves: dict[str, pd.DataFrame] = {"RECORDED": recorded_curve}
        for seed in range(args.repeats):
            rec_diag = {
                "active_filter": "recorded",
                "sat_count": 0,
                "sat_ratio": 0.0,
                "constraint_count": 0,
                "constraint_ratio": 0.0,
                "filter_ms_mean": 0.0,
                "filter_ms_p95": 0.0,
                "solve_ms_mean": 0.0,
                "solve_ms_p95": 0.0,
                "solve_ms_max": 0.0,
            }
            raw_rows.append(replay_metrics(suite, recorded_curve, p, rec_diag, "RECORDED", seed))
            for cid in run_controls:
                base_seed = 9000 + 100 * seed + p.pass_index
                ctrl_df, diag = run_replay_control(
                    suite,
                    cid,
                    p,
                    y_real,
                    innovations,
                    base_seed,
                    args.selected_filter,
                    args.particle_count,
                    horizon=args.horizon,
                    residual_rho=args.residual_rho,
                    innovation_gain=args.innovation_gain,
                )
                raw_rows.append(replay_metrics(suite, ctrl_df, p, diag, cid, seed))
                tmp = ctrl_df.copy()
                tmp["control"] = cid
                tmp["seed"] = seed
                tmp["pass_file"] = p.pass_file
                all_curve_rows.append(tmp)
                if seed == 0 and cid in {"C0", "C2", "C7", "LSTM-MPC", "DMC-KF", "ADRC"}:
                    example_curves[cid] = ctrl_df
        example_curves_by_pass[p.pass_file] = example_curves
        plot_pass_curves(example_curves, p, out_dir, ["RECORDED", "C0", "C2", "C7", "LSTM-MPC", "DMC-KF", "ADRC"])

    raw = pd.DataFrame(raw_rows)
    overall, by_pass = summarize(raw)
    sig = paired_tests(raw, [c for c in controls if c not in {"C7"}])
    raw.to_csv(out_dir / "replay_control_raw.csv", index=False)
    overall.to_csv(out_dir / "replay_control_overall.csv", index=False)
    by_pass.to_csv(out_dir / "replay_control_by_pass.csv", index=False)
    sig.to_csv(out_dir / "replay_significance_tests.csv", index=False)
    if all_curve_rows:
        pd.concat(all_curve_rows, ignore_index=True).to_csv(out_dir / "replay_control_curves_seed0plus.csv", index=False)
    plot_overall(overall, out_dir)

    summary_lines = [
        "# Independent Data2 replay validation",
        "",
        f"- Validation passes: {len(passes)}",
        f"- Repeats: {args.repeats}",
        f"- Replay: one-update recorded-trace innovations, innovation_gain={args.innovation_gain}",
        f"- Default filter: {args.selected_filter}",
        "",
        "## Overall results",
        markdown_table(overall.sort_values("composite_normalized_RMS_mean")[
            [
                "control",
                "composite_normalized_RMS_mean",
                "composite_normalized_RMS_std",
                "T_error_IAE_mean",
                "h_error_IAE_mean",
                "S_error_IAE_mean",
                "control_total_variation_mean",
                "composite_improve_vs_recorded_%",
                "composite_improve_vs_C0_%",
            ]
        ]),
        "",
        "## Paired C7 comparisons",
        markdown_table(sig),
    ]
    (out_dir / "replay_validation_report.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("Replay validation finished.")
    print(out_dir)
    print(overall.sort_values("composite_normalized_RMS_mean")[["control", "composite_normalized_RMS_mean", "S_error_IAE_mean", "composite_improve_vs_recorded_%"]].to_string(index=False))
    print(sig.to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(
        "This legacy validation CLI is disabled. The released workflow imports "
        "only its segment-aware innovation extractor and runs through "
        "code/run_pipeline.py."
    )
