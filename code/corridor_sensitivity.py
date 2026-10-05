from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ENGINE = Path(__file__).resolve().parent
PROTOCOL_PATH = ENGINE / "protocol.py"
COVERAGES = [0.90, 0.95, 0.99]
COLORS = {"Recorded Data2": "#7A7A7A", "PID": "#0072B2", "OC-PID": "#009E73", "C2": "#E69F00", "C7-Core": "#D55E00"}
MARKERS = {"Recorded Data2": "x", "PID": "o", "OC-PID": "s", "C2": "^", "C7-Core": "D"}


def policy_name(label: str) -> str:
    if label.startswith("PID"):
        return "PID"
    if label.startswith("ADRC"):
        return "OC-PID"
    if label.startswith("C2"):
        return "C2"
    if label.startswith("C7"):
        return "C7-Core"
    raise ValueError(f"Unmapped control label: {label}")


def load_protocol():
    sys.path.insert(0, str(ENGINE))
    spec = importlib.util.spec_from_file_location("coverage_protocol", PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {PROTOCOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["coverage_protocol"] = module
    spec.loader.exec_module(module)
    return module


def load_batch_extension(path: Path, batch_root: Path):
    spec = importlib.util.spec_from_file_location("coverage_batch_extension", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["coverage_batch_extension"] = module
    spec.loader.exec_module(module)
    module.BATCH_ROOT = batch_root
    return module


def metrics(
    z: np.ndarray,
    phase: np.ndarray,
    envelope: pd.DataFrame,
    tail_mean_fn,
) -> dict[str, float]:
    env = envelope.set_index("phase")
    lo = np.array([env.loc[p, "S_norm_lo"] for p in phase], dtype=float)
    hi = np.array([env.loc[p, "S_norm_hi"] for p in phase], dtype=float)
    excess = np.maximum.reduce([lo - z, z - hi, np.zeros_like(z)])
    return {
        "S_out": float(np.mean(excess > 0)),
        "S_excess_mean": float(np.mean(excess)),
        "S_excess_cvar95": tail_mean_fn(excess),
    }


def build_envelopes(protocol, suite, data1_passes) -> tuple[dict[float, pd.DataFrame], pd.DataFrame]:
    rows = []
    for rp in data1_passes:
        n = len(rp.y_real)
        scale = float(protocol.scale_vec(rp.p)[2])
        for k, y in enumerate(rp.y_real):
            rows.append({"pass_id": rp.pass_id, "phase": suite.phase_name(min(k, n - 2), n, rp.p), "S_norm": float(y[2] / scale)})
    raw = pd.DataFrame(rows)
    envelopes = {}
    for coverage in COVERAGES:
        alpha = (1.0 - coverage) / 2.0
        env = raw.groupby("phase", sort=True)["S_norm"].agg(
            S_norm_lo=lambda x, a=alpha: x.quantile(a),
            S_norm_hi=lambda x, a=alpha: x.quantile(1.0 - a),
            S_norm_median="median",
        ).reset_index()
        env.insert(0, "coverage", coverage)
        envelopes[coverage] = env
    return envelopes, raw


def aggregate_file_metrics(rows: list[dict]) -> pd.DataFrame:
    files = pd.DataFrame(rows)
    metric_cols = ["S_out", "S_excess_mean", "S_excess_cvar95"]
    condition = files.groupby(["coverage", "object", "policy", "condition_id"], as_index=False)[metric_cols].mean()
    summary = condition.groupby(["coverage", "object", "policy"], as_index=False)[metric_cols].mean()
    return files, condition, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--extension-script", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    nominal_dir = run / "03_nominal"
    out = args.out_dir.resolve() if args.out_dir else run / "10_appendix_audit" / "06_envelope_coverage_sensitivity"
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    suite, replay = protocol.load_suite_modules()
    extension = load_batch_extension(args.extension_script.resolve(), args.batch_root.resolve())
    all_passes, _ = extension.load_batch_passes(protocol, suite, replay)
    data1 = [item for item in all_passes if item.pass_id in extension.DATA1_IDS]
    data2 = [item for item in all_passes if item.pass_id in extension.CANONICAL_DATA2_IDS]
    envelopes, data1_raw = build_envelopes(protocol, suite, data1)
    # The 95% row must be the exact frozen envelope used by the canonical
    # replay, not a separately regenerated approximation.
    canonical_envelope = pd.read_csv(
        nominal_dir / "audit" / "data1_stage_quantile_envelope_p025_p975.csv"
    )
    envelopes[0.95] = canonical_envelope[
        ["phase", "S_norm_lo", "S_norm_hi", "S_norm_median"]
    ].copy()
    pd.concat(envelopes.values(), ignore_index=True).to_csv(out / "data1_phase_envelopes_90_95_99.csv", index=False, encoding="utf-8-sig")

    calibration_rows = []
    for coverage, env in envelopes.items():
        m = metrics(
            data1_raw["S_norm"].to_numpy(), data1_raw["phase"].to_numpy(),
            env, protocol.tail_mean,
        )
        calibration_rows.append({"coverage": coverage, "expected_outside": 1.0 - coverage, **m})
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(out / "data1_in_sample_coverage_check.csv", index=False, encoding="utf-8-sig")

    rows = []
    for rp in data2:
        n = len(rp.y_real)
        phase = np.array([suite.phase_name(min(k, n - 2), n, rp.p) for k in range(n)])
        z = rp.y_real[:, 2] / float(protocol.scale_vec(rp.p)[2])
        for coverage, env in envelopes.items():
            rows.append({"coverage": coverage, "object": "recorded", "policy": "Recorded Data2", "pass_id": rp.pass_id, "condition_id": rp.condition_id, **metrics(z, phase, env, protocol.tail_mean)})

    step_path = nominal_dir / "stage_allocation_ablation_step_logs.csv.gz"
    steps = pd.read_csv(step_path)
    steps["policy"] = steps["control"].map(policy_name)
    pass_map = {rp.pass_id: rp for rp in data2}
    for (control, pass_id), g in steps.groupby(["control", "pass_id"], sort=False):
        rp = pass_map[pass_id]
        z = g["S_control_error"].to_numpy(dtype=float) / float(protocol.scale_vec(rp.p)[2])
        phase = g["phase"].to_numpy()
        for coverage, env in envelopes.items():
            rows.append({"coverage": coverage, "object": "policy_replay", "policy": policy_name(control), "pass_id": pass_id, "condition_id": rp.condition_id, **metrics(z, phase, env, protocol.tail_mean)})

    files, conditions, summary = aggregate_file_metrics(rows)
    files.to_csv(out / "coverage_sensitivity_file_metrics.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(out / "coverage_sensitivity_condition_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "coverage_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "reference_corridor_coverage_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.concat(envelopes.values(), ignore_index=True).to_csv(
        out / "reference_corridor_phase_envelopes.csv", index=False, encoding="utf-8-sig"
    )

    policy = summary[summary["object"].eq("policy_replay")].copy()
    effects = []
    for coverage in COVERAGES:
        indexed = policy[policy.coverage.eq(coverage)].set_index("policy")
        for baseline in ["PID", "OC-PID", "C2"]:
            for metric_name in ["S_out", "S_excess_mean", "S_excess_cvar95"]:
                base = float(indexed.loc[baseline, metric_name])
                c7 = float(indexed.loc["C7-Core", metric_name])
                effects.append({"coverage": coverage, "metric": metric_name, "comparison": f"C7-Core vs {baseline}", "effect_pct_ratio_of_aggregates": 100.0 * (c7 - base) / base, "favours_c7": c7 < base})
    effects = pd.DataFrame(effects)
    effects.to_csv(out / "coverage_sensitivity_policy_effects.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(out / "reference_corridor_policy_effects.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
    labels = [("S_out", "Reference-corridor exceedance rate"), ("S_excess_mean", "Mean excess"), ("S_excess_cvar95", "CVaR95 excess")]
    order = ["Recorded Data2", "PID", "OC-PID", "C2", "C7-Core"]
    for ax, (metric_name, title) in zip(axes, labels):
        for name in order:
            g = summary[summary.policy.eq(name)].sort_values("coverage")
            ax.plot(100 * g.coverage, g[metric_name], color=COLORS[name], marker=MARKERS[name], linewidth=1.8, markersize=6, label=name)
        ax.set_title(title)
        ax.set_xlabel("Data1 reference coverage (%)")
        ax.set_xticks([90, 95, 99])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Condition-weighted metric")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Sensitivity to the width of the Data1 flatness reference corridor", fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(fig_dir / "figS_reference_corridor_coverage_sensitivity.png", dpi=240, bbox_inches="tight")
    fig.savefig(fig_dir / "figS_reference_corridor_coverage_sensitivity.svg", bbox_inches="tight")
    plt.close(fig)

    nominal = pd.read_csv(nominal_dir / "stage_allocation_ablation_metrics.csv")
    nominal["policy"] = nominal["control"].map(policy_name)
    metric_names = ["S_out", "S_excess_mean", "S_excess_cvar95"]
    expected = (
        nominal.groupby(["policy", "condition_id", "file"], as_index=False)[metric_names].mean()
        .groupby(["policy", "condition_id"], as_index=False)[metric_names].mean()
        .groupby("policy")[metric_names].mean()
    )
    observed = policy[policy.coverage.eq(0.95)].set_index("policy")[metric_names]
    metric_deltas = {
        metric: float((observed[metric] - expected[metric]).abs().max())
        for metric in metric_names
    }
    max_delta = max(metric_deltas.values())
    hard = {
        "data1_coverage_max_abs_error": float((calibration.S_out - calibration.expected_outside).abs().max()),
        "coverage_095_matches_canonical_S_out_max_abs_delta": metric_deltas["S_out"],
        "coverage_095_matches_canonical_all_metrics_max_abs_delta": max_delta,
        "coverage_095_matches_canonical_by_metric_max_abs_delta": metric_deltas,
        "c7_mean_excess_favours_pid_all_widths": bool(effects[(effects.metric == "S_excess_mean") & (effects.comparison == "C7-Core vs PID")].favours_c7.all()),
        "c7_mean_excess_favours_oc_pid_all_widths": bool(effects[(effects.metric == "S_excess_mean") & (effects.comparison == "C7-Core vs OC-PID")].favours_c7.all()),
        "coverages": COVERAGES,
        "data1_passes": len(data1),
        "data2_files": len(data2),
    }
    (out / "coverage_sensitivity_hard_checks.json").write_text(json.dumps(hard, indent=2), encoding="utf-8")
    print(json.dumps(hard, indent=2))
    if not np.isfinite(max_delta) or max_delta > 1e-12:
        raise AssertionError(
            f"95% corridor metrics differ from canonical aggregation by {max_delta:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
