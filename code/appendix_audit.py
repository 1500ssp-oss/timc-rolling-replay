"""Appendix A/B/C/H for the mature-production Pareto narrative.

The script deliberately separates historical-data audit from replay outcomes.
Data1 defines every threshold; Data2 is evaluation only.  It never assumes a
fixed sampling period: time and distance come from the archived-update logs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocol.py"
RNG = np.random.default_rng(20260711)


def load_protocol():
    spec = importlib.util.spec_from_file_location("appendix_protocol", PROTOCOL_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["appendix_protocol"] = module
    spec.loader.exec_module(module)
    return module


def load_batch_extension(path: Path, batch_root: Path):
    spec = importlib.util.spec_from_file_location("appendix_batch_extension", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["appendix_batch_extension"] = module
    spec.loader.exec_module(module)
    module.BATCH_ROOT = batch_root
    return module


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip" if path.suffix == ".gz" else None)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_key(row: pd.Series) -> str:
    base = str(row.get("base_control", ""))
    label = str(row.get("control", ""))
    if base == "C0" or label.startswith("PID"):
        return "PID"
    if base == "ADRC" or label.startswith("ADRC"):
        return "ADRC"
    if base == "C2" or label.startswith("C2"):
        return "C2"
    if base == "C7" or label.startswith("C7"):
        return "C7-Core"
    return label


def collect_real_rows(protocol, suite, passes, dataset: str) -> pd.DataFrame:
    rows: list[dict] = []
    for rp in passes:
        scale = protocol.scale_vec(rp.p)
        values = np.asarray(rp.y_real, dtype=float) / scale
        for target, column in enumerate(("T", "h", "S")):
            z = values[:, target]
            for value in z[np.isfinite(z)]:
                rows.append({"dataset": dataset, "pass_id": rp.pass_id, "target": column, "value_norm": float(value)})
    return pd.DataFrame(rows)


def run_maturity(protocol, suite, data1, data2, out: Path) -> None:
    real1 = collect_real_rows(protocol, suite, data1, "Data1")
    real2 = collect_real_rows(protocol, suite, data2, "Data2")
    thresholds: dict[str, dict[str, float]] = {}
    summary_rows: list[dict] = []
    for target, g1 in real1.groupby("target"):
        center = float(g1.value_norm.median())
        absdev = (g1.value_norm - center).abs()
        q80, q95 = (float(absdev.quantile(q)) for q in (0.80, 0.95))
        thresholds[target] = {"data1_median": center, "central_absdev_p80": q80, "near_absdev_p95": q95}
        for dataset, data in (("Data1", real1), ("Data2", real2)):
            target_data = data.loc[data.target.eq(target), ["pass_id", "value_norm"]].dropna()
            x = target_data["value_norm"]
            d = (x - center).abs()
            pass_stats = []
            for pass_id, pass_group in target_data.groupby("pass_id", sort=False):
                values = pass_group["value_norm"].to_numpy(float)
                rho_i = float(pd.Series(values).autocorr(lag=1)) if len(values) > 2 else np.nan
                rho_i = float(np.clip(rho_i, -0.99, 0.99)) if np.isfinite(rho_i) else np.nan
                ess_i = float(len(values) * (1.0 - rho_i) / (1.0 + rho_i)) if np.isfinite(rho_i) else np.nan
                pass_stats.append((len(values), rho_i, ess_i))
            finite_stats = [item for item in pass_stats if np.isfinite(item[1])]
            rho = float(np.average([item[1] for item in finite_stats], weights=[item[0] for item in finite_stats]))
            rho_median = float(np.median([item[1] for item in finite_stats]))
            ess = float(np.sum([item[2] for item in finite_stats]))
            summary_rows.append({
                "dataset": dataset, "target": target, "n_rows": len(x), "n_passes": len(pass_stats), "median_norm": float(x.median()),
                "IQR": float(x.quantile(.75) - x.quantile(.25)), "MAD": float((x - x.median()).abs().median()),
                "P5": float(x.quantile(.05)), "P95": float(x.quantile(.95)),
                "central_share_locked": float((d <= q80).mean()),
                "near_limit_share_locked": float(((d > q80) & (d <= q95)).mean()),
                "extreme_share_locked": float((d > q95).mean()),
                "lag1_autocorrelation": rho, "lag1_autocorrelation_pass_median": rho_median,
                "effective_n_AR1_approx": ess,
            })
    pd.DataFrame(summary_rows).to_csv(out / "10_maturity_headroom_summary.csv", index=False, encoding="utf-8-sig")
    (out / "10_maturity_thresholds_locked_data1.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    real2.assign(**{"absdev_from_data1_median": real2.apply(lambda r: abs(r.value_norm - thresholds[r.target]["data1_median"]), axis=1)}).to_csv(
        out / "10_maturity_data2_real_distribution.csv", index=False, encoding="utf-8-sig"
    )

    # Two first-row channels and one full-width second-row channel keep
    # the original independent distributions legible at manuscript width.
    fig = plt.figure(figsize=(6.8, 5.4))
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    for ax, target in zip(axes, ("T", "h", "S")):
        for dataset, color in (("Data1", "#1976D2"), ("Data2", "#E15759")):
            x = (real1 if dataset == "Data1" else real2).query("target == @target").value_norm
            ax.hist(x, bins=50, density=True, alpha=.38, label=dataset, color=color)
        ax.axvline(thresholds[target]["data1_median"], color="#222222", lw=.9)
        ax.set_title(f"{target}: archived value distribution", fontsize=10.5, pad=8)
        ax.set_xlabel("normalized value", fontsize=10.0)
        ax.set_ylabel("density", fontsize=10.0)
        ax.tick_params(axis="both", labelsize=9.5)
    axes[-1].legend(frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.095, hspace=0.73, wspace=0.34)
    fig.savefig(out / "10_maturity_distributions.png", dpi=300, facecolor="white")
    fig.savefig(out / "10_maturity_distributions.svg", facecolor="white")
    plt.close(fig)


def hierarchical_bootstrap(effect_by_file: pd.DataFrame, n_boot: int) -> np.ndarray:
    groups = [g.effect_pct.to_numpy(float) for _, g in effect_by_file.groupby("condition_id", sort=False)]
    if not groups:
        return np.array([])
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        selected = RNG.integers(0, len(groups), size=len(groups))
        values = [groups[i][RNG.integers(0, len(groups[i]))] for i in selected if len(groups[i])]
        draws[b] = float(np.mean(values)) if values else np.nan
    return draws


def run_mde(run: Path, out: Path, n_boot: int) -> None:
    metrics = read_csv(run / "03_nominal" / "stage_allocation_ablation_metrics.csv")
    metrics = metrics.loc[metrics.block.eq("data1_controller_search")].copy()
    metrics["policy"] = metrics.apply(policy_key, axis=1)
    metric_map = {
        "composite_RMS": "composite_normalized_RMS", "TV_L_per_100m": "TV_L_per_100m",
        "S_excess_mean": "S_excess_mean", "S_excess_CVaR95": "S_excess_cvar95",
    }
    ropes = {"composite_RMS": 0.5, "TV_L_per_100m": 5.0, "S_excess_mean": 5.0, "S_excess_CVaR95": 5.0}
    effect_rows: list[dict] = []
    mde_rows: list[dict] = []
    forest: list[dict] = []
    for metric_label, metric in metric_map.items():
        for baseline in ("PID", "ADRC", "C2"):
            a = metrics.loc[metrics.policy.eq("C7-Core"), ["pass_id", "condition_id", metric]].rename(columns={metric: "c7"})
            b = metrics.loc[metrics.policy.eq(baseline), ["pass_id", "condition_id", metric]].rename(columns={metric: "base"})
            paired = a.merge(b, on=["pass_id", "condition_id"], how="inner")
            paired = paired.loc[paired.base.abs() > 1e-12].copy()
            paired["effect_pct"] = 100.0 * (paired.c7 - paired.base) / paired.base
            draws = hierarchical_bootstrap(paired, n_boot)
            lo, hi = np.nanpercentile(draws, [2.5, 97.5])
            # Match the estimand used by the hierarchical bootstrap: first
            # average repeated files inside condition, then weight conditions
            # equally. A direct file mean overweights conditions with repeats.
            point = float(paired.groupby("condition_id", sort=False).effect_pct.mean().mean())
            boot_se = float(np.nanstd(draws, ddof=1))
            rope = ropes[metric_label]
            effect_rows.append({"metric": metric_label, "comparison": f"C7-Core vs {baseline}", "n_files": len(paired), "n_conditions": paired.condition_id.nunique(), "effect_pct": point, "CI_low_pct": lo, "CI_high_pct": hi, "bootstrap_SE_pct": boot_se, "ROPE_pct": rope, "practically_equivalent": bool(lo >= -rope and hi <= rope), "point_aggregation": "mean files within condition, then equal condition mean", "bootstrap": "hierarchical condition then file, 10000 resamples"})
            for _, row in paired.iterrows():
                forest.append({"metric": metric_label, "comparison": f"C7-Core vs {baseline}", "pass_id": row.pass_id, "condition_id": row.condition_id, "effect_pct": row.effect_pct})
            for delta in (.25, .5, .75, 1., 1.5, 2., 3., 5., 7.5, 10.):
                # The injection power is computed from the paired hierarchical
                # bootstrap SE.  It is intentionally a sensitivity calculation,
                # not a claim that the ten files are independent observations.
                if boot_se <= 0 or not np.isfinite(boot_se):
                    power = np.nan
                else:
                    z = delta / boot_se
                    phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
                    power = 1.0 - phi(1.959964 - z) + phi(-1.959964 - z)
                mde_rows.append({"metric": metric_label, "comparison": f"C7-Core vs {baseline}", "injected_abs_effect_pct": delta, "detection_power": power, "method": "paired hierarchical-bootstrap SE injection"})
    effects = pd.DataFrame(effect_rows); effects.to_csv(out / "11_paired_effect_ci.csv", index=False, encoding="utf-8-sig")
    curve = pd.DataFrame(mde_rows); curve.to_csv(out / "11_mde_curve.csv", index=False, encoding="utf-8-sig")
    all_pairs = effects[["metric", "comparison"]].drop_duplicates()
    detected = curve.loc[curve.detection_power.ge(.8)].groupby(["metric", "comparison"], as_index=False).injected_abs_effect_pct.min().rename(columns={"injected_abs_effect_pct": "MDE_80pct_power_pct"})
    mde = all_pairs.merge(detected, on=["metric", "comparison"], how="left")
    mde["MDE_80pct_power_reached_within_10pct"] = mde.MDE_80pct_power_pct.notna()
    mde.to_csv(out / "11_mde_by_metric.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(forest).to_csv(out / "11_condition_level_effects.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for (metric, comp), g in curve.groupby(["metric", "comparison"]):
        ax.plot(g.injected_abs_effect_pct, g.detection_power, marker="o", ms=3, label=f"{metric}: {comp.split()[-1]}")
    ax.axhline(.8, color="#333333", ls="--", lw=.9)
    ax.set(xlabel="Injected absolute paired effect (%)", ylabel="Detection power", ylim=(-.03, 1.03), title="Paired-bootstrap-calibrated detection sensitivity")
    ax.legend(fontsize=7, ncol=2, frameon=False); fig.tight_layout(); fig.savefig(out / "11_mde_curve.png", dpi=220); plt.close(fig)
    forest_df = pd.DataFrame(forest)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = list(forest_df.comparison.drop_duplicates())
    for i, comparison in enumerate(order):
        g = forest_df.loc[(forest_df.comparison == comparison) & (forest_df.metric == "S_excess_mean")]
        ax.scatter(g.effect_pct, np.full(len(g), i), alpha=.75, label=comparison)
    ax.axvline(0, color="#333333", lw=.8)
    ax.set(yticks=range(len(order)), yticklabels=order, xlabel="C7-Core relative effect on mean flatness excess (%)", title="Condition/file paired effects")
    fig.tight_layout(); fig.savefig(out / "11_condition_forest.png", dpi=220); plt.close(fig)


def run_episodes(protocol, run: Path, out: Path, passes) -> None:
    scales = {rp.pass_id: float(protocol.scale_vec(rp.p)[2]) for rp in passes}
    env = pd.read_csv(run / "03_nominal" / "audit" / "data1_stage_quantile_envelope_p025_p975.csv").set_index("phase")
    logs = read_csv(run / "03_nominal" / "stage_allocation_ablation_step_logs.csv.gz")
    logs["policy"] = logs.apply(policy_key, axis=1)
    episodes: list[dict] = []
    for (policy, pass_id, segment), g in logs.groupby(["policy", "pass_id", "segment_id"], sort=False):
        g = g.sort_values("k").reset_index(drop=True)
        scale = scales[str(pass_id)]
        upper = g.phase.map(env.S_norm_hi).fillna(float(env.S_norm_hi.median())).to_numpy(float)
        lower = g.phase.map(env.S_norm_lo).fillna(float(env.S_norm_lo.median())).to_numpy(float)
        z = g.S_control_error.to_numpy(float) / scale
        excess = np.maximum.reduce([lower - z, z - upper, np.zeros(len(g))])
        mask = excess > 0
        start = None
        for i, flag in enumerate(np.r_[mask, False]):
            if flag and start is None: start = i
            if not flag and start is not None:
                j = i - 1; part = g.iloc[start:i]; rec = np.nan
                inside = 0
                for q in range(i, len(g)):
                    inside = inside + 1 if not mask[q] else 0
                    if inside >= 2:
                        rec = float(g.distance_step_m.iloc[i:q + 1].sum()); break
                # Legacy column names are retained for deterministic compatibility;
                # the values are two-sided recorded-band distance descriptors.
                episodes.append({"policy": policy, "pass_id": pass_id, "segment_id": segment, "start_k": int(g.k.iloc[start]), "end_k": int(g.k.iloc[j]), "duration_s": float(part.timestamp_dt_s.sum()), "distance_m": float(part.distance_step_m.sum()), "peak_excess_norm": float(excess[start:i].max()), "integrated_severity_excess_m": float(np.sum(excess[start:i] * part.distance_step_m.to_numpy(float))), "recovery_distance_m": rec})
                start = None
    ep = pd.DataFrame(episodes)
    per_file = ep.groupby(["policy", "pass_id"], as_index=False).agg(episode_count=("start_k", "size"), episode_distance_m=("distance_m", "sum"), peak_excess_norm=("peak_excess_norm", "max"), integrated_severity_excess_m=("integrated_severity_excess_m", "sum"), P95_episode_peak=("peak_excess_norm", lambda x: x.quantile(.95)), CVaR95_episode_peak=("peak_excess_norm", lambda x: x.loc[x >= x.quantile(.95)].mean()), recovery_distance_m=("recovery_distance_m", "mean"))
    total_dist = logs.groupby(["policy", "pass_id"], as_index=False).distance_step_m.sum().rename(columns={"distance_step_m": "total_distance_m"})
    per_file = total_dist.merge(per_file, on=["policy", "pass_id"], how="left").fillna(0)
    per_file["episodes_per_100m"] = 100 * per_file.episode_count / per_file.total_distance_m
    ep.to_csv(out / "20_episode_events.csv", index=False, encoding="utf-8-sig")
    per_file.to_csv(out / "20_episode_metrics_by_file.csv", index=False, encoding="utf-8-sig")
    pivot = per_file.pivot(index="pass_id", columns="policy", values="integrated_severity_excess_m")
    paired = []
    for base in ("PID", "ADRC", "C2"):
        if base in pivot and "C7-Core" in pivot:
            d = 100 * (pivot["C7-Core"] - pivot[base]) / pivot[base].replace(0, np.nan)
            paired.append({"comparison": f"C7-Core vs {base}", "metric": "integrated_severity_excess_m", "mean_relative_effect_pct": float(d.mean()), "median_relative_effect_pct": float(d.median()), "n_files": int(d.notna().sum())})
    pd.DataFrame(paired).to_csv(out / "21_episode_paired_effects.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for policy, g in per_file.groupby("policy"):
        ax.scatter(np.full(len(g), policy), g.integrated_severity_excess_m, alpha=.75, label=policy)
    ax.set(ylabel="Integrated two-sided band distance (normalized*m)", title="Recorded-band departure episodes")
    fig.tight_layout(); fig.savefig(out / "20_episode_integrated_severity.png", dpi=220); plt.close(fig)


def run_hard_fail(run: Path, out: Path) -> None:
    required = [
        out / "10_maturity_headroom_summary.csv", out / "11_mde_by_metric.csv", out / "20_episode_metrics_by_file.csv",
        run / "03_nominal" / "stage_allocation_ablation_metrics.csv",
    ]
    protocol_manifest = json.loads((run / "03_nominal" / "stage_allocation_ablation_manifest.json").read_text(encoding="utf-8"))
    batch_manifest = json.loads((run / "03_nominal" / "current_batch_sweep_manifest.json").read_text(encoding="utf-8"))
    nominal_metrics = pd.read_csv(run / "03_nominal" / "stage_allocation_ablation_metrics.csv")
    wording_files = [run / "00_protocol" / "no_fixed_sampling_period_audit.md", run / "02_tuning" / "controller_lock.json", out / "APPENDIX_EXPERIMENT_REPORT.md"]
    wording = "\n".join(path.read_text(encoding="utf-8") for path in wording_files if path.exists()).lower()
    forbidden_timebase_wording = ("0.50 s sampling period", "10 steps / 5.0 s", "fixed 0.5 s sampling")
    checks = [
        ("required_appendix_outputs_nonempty", all(p.exists() and p.stat().st_size > 0 for p in required)),
        ("archived_update_timebase", protocol_manifest.get("timebase") == "archived_update_no_fixed_sampling_period"),
        ("predictor_horizon_is_update_indexed", int(protocol_manifest.get("archived_update_horizon", 0)) == 10),
        ("segment_gap_v2_batch_loader", batch_manifest.get("timebase") == "segment_gap_v2"),
        ("canonical_Data2_has_10_passes", nominal_metrics.pass_id.nunique() == 10),
        ("canonical_summary_has_8_conditions", nominal_metrics.condition_id.nunique() == 8),
        ("nominal_has_four_locked_policies", nominal_metrics.base_control.nunique() >= 4),
        ("nominal_has_40_policy_pass_rows", len(nominal_metrics) == 40),
        ("nominal_commands_are_finite", int(nominal_metrics.invalid_command_count.sum()) == 0),
        ("no_forbidden_fixed_sampling_wording_in_final_audit_outputs", not any(phrase in wording for phrase in forbidden_timebase_wording)),
    ]
    report = pd.DataFrame(checks, columns=["check", "passed"])
    report["severity"] = np.where(report.passed, "pass", "HARD_FAIL")
    report.to_csv(out / "60_data_provenance_and_unit_audit.csv", index=False, encoding="utf-8-sig")
    manifest_rows = []
    for path in sorted(run.rglob("*.csv")):
        manifest_rows.append({"relative_path": str(path.relative_to(run)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(out / "00_canonical_manifest.csv", index=False, encoding="utf-8-sig")
    lock = hashlib.sha256((out / "00_canonical_manifest.csv").read_bytes()).hexdigest()
    (out / "run_lock.sha256").write_text(f"{lock}  00_canonical_manifest.csv\n", encoding="ascii")
    (out / "60_hard_fail_audit.json").write_text(json.dumps({"all_passed": bool(report.passed.all()), "checks": report.to_dict(orient="records")}, indent=2), encoding="utf-8")
    if not bool(report.passed.all()):
        raise SystemExit("Appendix H hard-fail audit did not pass")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--extension-script", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve(); out = run / "10_appendix_audit"; out.mkdir(exist_ok=True)
    protocol = load_protocol(); suite, replay = protocol.load_suite_modules()
    extension = load_batch_extension(args.extension_script.resolve(), args.batch_root.resolve())
    all_passes, _ = extension.load_batch_passes(protocol, suite, replay)
    data1 = [item for item in all_passes if item.pass_id in extension.DATA1_IDS]
    data2 = [item for item in all_passes if item.pass_id in extension.CANONICAL_DATA2_IDS]
    run_maturity(protocol, suite, data1, data2, out)
    run_mde(run, out, args.bootstrap)
    run_episodes(protocol, run, out, data2)
    run_hard_fail(run, out)
    print(f"[done] appendix A/B/C/H: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
