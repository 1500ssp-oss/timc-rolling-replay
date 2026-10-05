from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "outputs" / "sensitivity"
CANON = ROOT / "outputs" / "canonical"
OUT = BASE / "04_analysis"
FIG = OUT / "figures"
SEED = 20260712
N_BOOT = 10_000

POLICY_MAP = {
    "PID-g1.20-du0.60": "PID",
    "ADRC-g1.20-du0.95": "OC-PID",
    "C2-g1.20-du0.80": "C2",
    "C7-Core-g0.75-mpc0.14": "C7-Core",
}
METRICS = {
    "composite_normalized_RMS": "RMS_c",
    "TV_L_per_100m": "TV/100 m",
    "S_excess_mean": "Mean excess",
    "S_excess_cvar95": "CVaR95 excess",
}
COLORS = {"PID": "#0072B2", "OC-PID": "#009E73", "C2": "#E69F00", "C7-Core": "#D55E00"}
MARKERS = {"PID": "o", "OC-PID": "s", "C2": "^", "C7-Core": "D"}


def policy_name(label: str) -> str:
    if label.startswith("PID"):
        return "PID"
    if label.startswith("ADRC") or label.startswith("OC-PID"):
        return "OC-PID"
    if label.startswith("C2"):
        return "C2"
    if label.startswith("C7"):
        return "C7-Core"
    raise ValueError(label)


def publication_conditions(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    if "pass_id" in result and "condition_id" in result:
        result.loc[result["pass_id"].eq("P06"), "condition_id"] = (
            "0.628->0.458 acceleration"
        )
    return result


def condition_weighted(data: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    file_level = data.groupby(group + ["condition_id", "file"], as_index=False)[metrics].mean()
    condition_level = file_level.groupby(group + ["condition_id"], as_index=False)[metrics].mean()
    return condition_level.groupby(group, as_index=False)[metrics].mean()


def paired_effects(data: pd.DataFrame, metric: str, baseline: str) -> tuple[float, pd.Series]:
    file_level = data.groupby(["condition_id", "file", "policy"], as_index=False)[metric].mean()
    wide = file_level.pivot(index=["condition_id", "file"], columns="policy", values=metric).dropna(subset=[baseline, "C7-Core"])
    effect = 100.0 * (wide["C7-Core"] - wide[baseline]) / wide[baseline]
    condition = effect.groupby(level="condition_id").mean()
    return float(condition.mean()), condition


def hierarchical_bootstrap(data: pd.DataFrame, metric: str, baseline: str, rng: np.random.Generator) -> np.ndarray:
    file_level = data.groupby(["condition_id", "file", "policy"], as_index=False)[metric].mean()
    wide = file_level.pivot(index=["condition_id", "file"], columns="policy", values=metric).dropna(subset=[baseline, "C7-Core"])
    wide["effect"] = 100.0 * (wide["C7-Core"] - wide[baseline]) / wide[baseline]
    by_condition = {str(c): g["effect"].to_numpy(float) for c, g in wide.reset_index().groupby("condition_id")}
    conditions = np.array(sorted(by_condition), dtype=object)
    draws = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        sampled_conditions = rng.choice(conditions, size=len(conditions), replace=True)
        values = []
        for condition in sampled_conditions:
            files = by_condition[str(condition)]
            values.append(float(rng.choice(files, size=len(files), replace=True).mean()))
        draws[b] = float(np.mean(values))
    return draws


def non_dominated_probability(data: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    objective_cols = list(METRICS)
    file_level = data.groupby(["condition_id", "file", "policy"], as_index=False)[objective_cols].mean()
    conditions = sorted(file_level["condition_id"].unique())
    policies = ["PID", "OC-PID", "C2", "C7-Core"]
    by_condition = {c: g for c, g in file_level.groupby("condition_id")}

    def aggregate(sampled_conditions: list[str], local_rng: np.random.Generator | None) -> pd.DataFrame:
        pieces = []
        for slot, condition in enumerate(sampled_conditions):
            g = by_condition[condition]
            files = g["file"].unique()
            selected = files if local_rng is None else local_rng.choice(files, size=len(files), replace=True)
            sampled = pd.concat([g[g.file.eq(f)] for f in selected], ignore_index=True)
            part = sampled.groupby("policy", as_index=False)[objective_cols].mean()
            part["slot"] = slot
            pieces.append(part)
        return pd.concat(pieces).groupby("policy", as_index=False)[objective_cols].mean().set_index("policy").loc[policies]

    point = aggregate(conditions, None)

    def mask(values: np.ndarray) -> np.ndarray:
        keep = np.ones(len(values), dtype=bool)
        for i in range(len(values)):
            for j in range(len(values)):
                if i != j and np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                    keep[i] = False
                    break
        return keep

    point_mask = mask(point.to_numpy(float))
    counts = np.zeros(len(policies), dtype=int)
    for _ in range(N_BOOT):
        sampled_conditions = list(rng.choice(conditions, size=len(conditions), replace=True))
        sample = aggregate(sampled_conditions, rng)
        counts += mask(sample.to_numpy(float)).astype(int)
    probabilities = pd.DataFrame({"policy": policies, "point_non_dominated": point_mask, "bootstrap_non_dominated_probability": counts / N_BOOT})
    return point.reset_index(), probabilities


def style_axes(ax) -> None:
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    global CANON, BASE, OUT, FIG, N_BOOT
    CANON = args.run.resolve()
    BASE = args.base.resolve() if args.base else CANON / "implementation_sensitivity"
    OUT = BASE / "04_analysis"
    FIG = OUT / "figures"
    N_BOOT = args.bootstrap
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5})

    nominal = publication_conditions(
        pd.read_csv(CANON / "03_nominal" / "stage_allocation_ablation_metrics.csv")
    )
    nominal["policy"] = nominal["control"].map(policy_name)
    canonical_summary = pd.read_csv(CANON / "03_nominal" / "stage_allocation_ablation_condition_weighted_summary.csv")
    canonical_summary.insert(0, "policy", canonical_summary["control"].map(policy_name))
    canonical_summary.to_csv(OUT / "nominal_condition_weighted_policy_summary.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(SEED)
    effect_rows = []
    condition_rows = []
    loco_rows = []
    for metric, label in METRICS.items():
        for baseline in ["PID", "OC-PID", "C2"]:
            point, by_condition = paired_effects(nominal, metric, baseline)
            draws = hierarchical_bootstrap(nominal, metric, baseline, rng)
            bootstrap_se = float(draws.std(ddof=1))
            power_grid = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]
            normal = NormalDist()
            mde = np.nan
            for delta in power_grid:
                power = 1.0 - normal.cdf(1.96 - delta / bootstrap_se) + normal.cdf(-1.96 - delta / bootstrap_se)
                if power >= 0.80:
                    mde = delta
                    break
            if not np.isclose(point, by_condition.mean(), atol=1e-12):
                raise RuntimeError("Point functional mismatch")
            effect_rows.append({
                "metric": metric,
                "metric_label": label,
                "comparison": f"C7-Core vs {baseline}",
                "estimand": "mean file-relative effect within condition, then equal-condition mean",
                "point_pct": point,
                "bootstrap_mean_pct": float(draws.mean()),
                "bootstrap_median_pct": float(np.median(draws)),
                "bootstrap_se_pct": bootstrap_se,
                "ci_low_pct": float(np.percentile(draws, 2.5)),
                "ci_high_pct": float(np.percentile(draws, 97.5)),
                "bootstrap_seed": SEED,
                "bootstrap_draws": N_BOOT,
                "mde_80pct_grid_pct": mde,
                "point_inside_ci": bool(np.percentile(draws, 2.5) <= point <= np.percentile(draws, 97.5)),
            })
            for condition, value in by_condition.items():
                condition_rows.append({"metric": metric, "comparison": f"C7-Core vs {baseline}", "condition_id": condition, "effect_pct": value, "direction_favours_c7": value < 0})
            for omitted in by_condition.index:
                remaining = by_condition.drop(index=omitted)
                loco_rows.append({"metric": metric, "comparison": f"C7-Core vs {baseline}", "omitted_condition": omitted, "loco_effect_pct": float(remaining.mean())})

    effects = pd.DataFrame(effect_rows)
    conditions = pd.DataFrame(condition_rows)
    loco = pd.DataFrame(loco_rows)
    effects.to_csv(OUT / "paired_effect_bootstrap_recomputed.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(OUT / "condition_level_effects.csv", index=False, encoding="utf-8-sig")
    loco.to_csv(OUT / "leave_one_condition_out_jackknife.csv", index=False, encoding="utf-8-sig")
    direction = conditions.groupby(["metric", "comparison"], as_index=False).agg(conditions_favouring_c7=("direction_favours_c7", "sum"), conditions_total=("condition_id", "nunique"), min_effect_pct=("effect_pct", "min"), max_effect_pct=("effect_pct", "max"))
    direction.to_csv(OUT / "condition_direction_consistency.csv", index=False, encoding="utf-8-sig")

    point_objectives, nd_prob = non_dominated_probability(nominal, rng)
    point_objectives.to_csv(OUT / "point_objectives_for_dominance.csv", index=False, encoding="utf-8-sig")
    nd_prob.to_csv(OUT / "bootstrap_non_dominated_probability.csv", index=False, encoding="utf-8-sig")

    emulator = publication_conditions(
        pd.read_csv(BASE / "01_emulator_compact" / "stage_allocation_ablation_metrics.csv")
    )
    emulator_cw = condition_weighted(emulator.rename(columns={"control": "policy"}), ["scenario", "policy"], list(METRICS) + ["output_clip_ratio"])
    emulator_cw.to_csv(OUT / "emulator_envelope_condition_weighted.csv", index=False, encoding="utf-8-sig")
    emulator_effects = []
    for scenario, group in emulator.groupby("scenario"):
        work = group.rename(columns={"control": "policy"})
        for metric in ["S_excess_mean", "S_excess_cvar95", "composite_normalized_RMS", "TV_L_per_100m"]:
            for baseline in ["PID", "OC-PID"]:
                point, _ = paired_effects(work, metric, baseline)
                emulator_effects.append({"scenario": scenario, "metric": metric, "comparison": f"C7-Core vs {baseline}", "effect_pct": point, "direction_favours_c7": point < 0})
    emulator_effects = pd.DataFrame(emulator_effects)
    emulator_effects.to_csv(OUT / "emulator_envelope_effects.csv", index=False, encoding="utf-8-sig")
    emulator_stability = emulator_effects.groupby(["metric", "comparison"], as_index=False).agg(models_favouring_c7=("direction_favours_c7", "sum"), models_total=("scenario", "nunique"), min_effect_pct=("effect_pct", "min"), max_effect_pct=("effect_pct", "max"))
    emulator_stability.to_csv(OUT / "emulator_direction_stability.csv", index=False, encoding="utf-8-sig")

    energy_cols = [c for c in emulator.columns if c.startswith("command_effect_") or c.startswith("innovation_")]
    energy = emulator[emulator.scenario.eq("nominal")].groupby("control", as_index=False)[energy_cols].mean(numeric_only=True)
    energy.to_csv(OUT / "command_innovation_energy_audit.csv", index=False, encoding="utf-8-sig")

    amplitude = publication_conditions(
        pd.read_csv(BASE / "02_amplitude_envelope" / "stage_allocation_ablation_metrics.csv")
    ).rename(columns={"control": "policy"})
    amp_metrics = list(METRICS) + ["amplitude_projection_ratio", "amplitude_active_speed_ratio", "amplitude_active_gap_ratio", "amplitude_active_shape_ratio", "raw_to_applied_L2_mean", "output_clip_ratio"]
    amplitude_cw = condition_weighted(amplitude, ["scenario", "amplitude_bound", "policy"], amp_metrics)
    amplitude_cw.to_csv(OUT / "amplitude_envelope_condition_weighted.csv", index=False, encoding="utf-8-sig")

    anti = publication_conditions(
        pd.read_csv(BASE / "03_anti_windup" / "stage_allocation_ablation_metrics.csv")
    ).rename(columns={"control": "policy"})
    anti_metrics = list(METRICS) + ["pid_aw_vector_freeze_count", "pid_aw_speed_count", "pid_aw_gap_count", "pid_aw_shape_count"]
    anti_cw = condition_weighted(anti, ["scenario", "anti_windup_mode", "policy"], anti_metrics)
    anti_cw.to_csv(OUT / "anti_windup_condition_weighted.csv", index=False, encoding="utf-8-sig")

    output_clip = publication_conditions(
        pd.read_csv(BASE / "04_output_clip" / "stage_allocation_ablation_metrics.csv")
    ).rename(columns={"control": "policy"})
    no_output_clip = publication_conditions(
        pd.read_csv(BASE / "05_no_output_clip" / "stage_allocation_ablation_metrics.csv")
    ).rename(columns={"control": "policy"})
    output_clip = pd.concat([output_clip[~output_clip.scenario.eq("clip_effectively_off")], no_output_clip], ignore_index=True, sort=False)
    clip_metrics = list(METRICS) + ["P95_abs_S", "output_clip_ratio"]
    output_clip_cw = condition_weighted(output_clip, ["scenario", "output_clip_scale", "policy"], clip_metrics)
    output_clip_cw.to_csv(OUT / "output_clip_condition_weighted.csv", index=False, encoding="utf-8-sig")
    clip_effects = []
    for scenario, group in output_clip.groupby("scenario"):
        for metric in ["composite_normalized_RMS", "S_excess_mean", "S_excess_cvar95"]:
            for baseline in ["PID", "OC-PID"]:
                point, _ = paired_effects(group, metric, baseline)
                clip_effects.append({"scenario": scenario, "metric": metric, "comparison": f"C7-Core vs {baseline}", "effect_pct": point})
    pd.DataFrame(clip_effects).to_csv(OUT / "output_clip_effects.csv", index=False, encoding="utf-8-sig")

    # Emulator envelope figure.
    plot = emulator_effects[emulator_effects.metric.eq("S_excess_mean")].copy()
    scenarios = ["nominal", "low_response_corner", "high_response_corner", "diagonal_response", "alternative_linear", "alternative_weakened"]
    labels = ["Nominal", "Low response", "High response", "Diagonal", "Linear alt.", "Weakened alt."]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(scenarios))
    for offset, baseline in [(-0.17, "PID"), (0.17, "OC-PID")]:
        vals = [float(plot[(plot.scenario.eq(s)) & plot.comparison.eq(f"C7-Core vs {baseline}" )].effect_pct.iloc[0]) for s in scenarios]
        ax.bar(x + offset, vals, width=0.32, label=f"vs {baseline}", color=COLORS[baseline], edgecolor="black", linewidth=0.6, hatch="//" if baseline == "OC-PID" else "")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("C7-Core relative mean-excess effect (%)")
    ax.set_title("Effect direction across locked response-emulator envelope")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_emulator_envelope.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "fig07_emulator_envelope.svg", bbox_inches="tight")
    plt.close(fig)

    # Amplitude co-design figure.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25))
    for policy in ["PID", "OC-PID", "C2", "C7-Core"]:
        g = amplitude_cw[amplitude_cw.policy.eq(policy)].sort_values("amplitude_bound")
        axes[0].plot(g.amplitude_bound, g.S_excess_mean, marker=MARKERS[policy], color=COLORS[policy], label=policy, linewidth=1.6)
        axes[1].plot(g.amplitude_bound, g.TV_L_per_100m, marker=MARKERS[policy], color=COLORS[policy], label=policy, linewidth=1.6)
    axes[0].set_ylabel("Mean flatness-envelope excess")
    axes[1].set_ylabel("Applied TV per 100 m")
    for ax in axes:
        ax.set_xlabel("Normalized amplitude bound")
        ax.set_xticks([0.45, 0.55, 0.65, 0.75])
        style_axes(ax)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Amplitude-envelope audit at locked policy-specific rate scales", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "fig08_amplitude_envelope.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "fig08_amplitude_envelope.svg", bbox_inches="tight")
    plt.close(fig)

    # Condition forest plot for the two principal severity comparisons.
    forest = conditions[(conditions.metric.eq("S_excess_mean")) & conditions.comparison.isin(["C7-Core vs PID", "C7-Core vs OC-PID"])].copy()
    condition_order = sorted(forest.condition_id.unique())
    code = {c: f"C{i+1}" for i, c in enumerate(condition_order)}
    map_df = pd.DataFrame({"condition_code": [code[c] for c in condition_order], "condition_id": condition_order})
    map_df.to_csv(OUT / "condition_code_register.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(condition_order))
    for offset, baseline in [(-0.12, "PID"), (0.12, "OC-PID")]:
        vals = [float(forest[(forest.condition_id.eq(c)) & forest.comparison.eq(f"C7-Core vs {baseline}")].effect_pct.iloc[0]) for c in condition_order]
        ax.scatter(vals, y + offset, marker=MARKERS[baseline], color=COLORS[baseline], edgecolor="black", linewidth=0.4, label=f"vs {baseline}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [code[c] for c in condition_order])
    ax.set_xlabel("Condition-level mean-excess effect (%)")
    ax.set_ylabel("Thickness condition")
    ax.set_title("Condition-level direction check")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig09_condition_effect_forest.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "fig09_condition_effect_forest.svg", bbox_inches="tight")
    plt.close(fig)

    # Output-protection audit: quality tail and composite RMS respond differently.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25))
    clip_order = [6.0, 8.0, 12.0, np.inf]
    clip_labels = ["6", "8", "12", "No guardrail"]
    for policy in ["PID", "OC-PID", "C2", "C7-Core"]:
        g = output_clip_cw[output_clip_cw.policy.eq(policy)].set_index("output_clip_scale").loc[clip_order]
        axes[0].plot(range(len(clip_order)), g.composite_normalized_RMS, marker=MARKERS[policy], color=COLORS[policy], label=policy, linewidth=1.6)
        axes[1].plot(range(len(clip_order)), g.S_excess_cvar95, marker=MARKERS[policy], color=COLORS[policy], label=policy, linewidth=1.6)
    axes[0].set_ylabel("Composite normalized RMS")
    axes[1].set_ylabel("Flatness-excess CVaR95")
    for ax in axes:
        ax.set_xticks(range(len(clip_order)), clip_labels)
        ax.set_xlabel("Numerical state-guardrail scale")
        style_axes(ax)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Evaluation sensitivity to the numerical state guardrail", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_output_clip_sensitivity.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "fig10_output_clip_sensitivity.svg", bbox_inches="tight")
    plt.close(fig)

    checks = {
        "bootstrap_rows": int(len(effects)),
        "all_points_inside_ci": bool(effects.point_inside_ci.all()),
        "conditions": int(nominal.condition_id.nunique()),
        "files": int(nominal.file.nunique()),
        "emulator_models": int(emulator.scenario.nunique()),
        "emulator_replays": int(len(emulator)),
        "amplitude_replays": int(len(amplitude)),
        "anti_windup_replays": int(len(anti)),
        "output_clip_replays": int(len(output_clip)),
        "nominal_output_clip_events": int(emulator.loc[emulator.scenario.eq("nominal"), "output_clip_count"].sum()),
        "bootstrap_seed": SEED,
        "bootstrap_draws": N_BOOT,
    }
    (OUT / "analysis_hard_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
