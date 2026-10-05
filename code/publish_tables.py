"""Assemble publication-level result tables from the replay pipeline outputs.

Every results/*.csv that depends on replay data is (re)written here from the
outputs/ directory using the exact published column schemas.  Static files
that do not depend on the replay pipeline (frozen predictor LOPO summary,
prediction-horizon audit) are kept unchanged.  Run with --manifest-only to
regenerate only MANIFEST_SHA256.csv after the rate-table step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

import batch_archive as archive  # noqa: E402
import controller_replay as runner  # noqa: E402
import implementation_sensitivity as impl  # noqa: E402

CANON = ROOT / "outputs" / "canonical" / "03_nominal"
SENS = ROOT / "outputs" / "sensitivity"
EXPANDED = ROOT / "outputs" / "expanded"
RESULTS = ROOT / "results"
ENVELOPE_PATH = ROOT / "config" / "reference_run" / "03_nominal" / "audit" / "data1_stage_quantile_envelope_p025_p975.csv"

METRICS4 = ["composite_normalized_RMS", "TV_L_per_100m", "S_excess_mean", "S_excess_cvar95"]
BOOT_SEED = 20260712
N_BOOT = 10_000
AGG_STR = "equal thickness transition; files first averaged within transition"
BAND_MEAN_LABEL = "Mean two-sided band distance"
BAND_TAIL_LABEL = "Upper-tail band distance"
BAND_FREQUENCY_LABEL = "Recorded-band departure frequency"


def publication_conditions(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "pass_id" in data and "condition_id" in data:
        data.loc[data["pass_id"].eq("P06"), "condition_id"] = "0.628->0.458 acceleration"
    return data


def condition_weighted(data: pd.DataFrame, metrics: list[str], group: list[str]) -> pd.DataFrame:
    data = publication_conditions(data)
    file_level = data.groupby(group + ["condition_id", "file"], as_index=False)[metrics].mean()
    condition_level = file_level.groupby(group + ["condition_id"], as_index=False)[metrics].mean()
    return condition_level.groupby(group, as_index=False)[metrics].mean()


def load_canonical() -> pd.DataFrame:
    metrics = pd.read_csv(CANON / "stage_allocation_ablation_metrics.csv")
    # Full policy names are required by the paired-effect machinery.
    metrics["policy"] = metrics["control"].map(impl.policy_name)
    return metrics


def load_episodes() -> pd.DataFrame:
    per_file = pd.read_csv(ROOT / "outputs" / "canonical" / "10_appendix_audit" / "20_episode_metrics_by_file.csv")
    return per_file


def write_pairwise_bootstrap(nominal: pd.DataFrame) -> None:
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for metric in METRICS4:
        for baseline in ["PID", "OC-PID", "C2"]:
            point, by_condition = impl.paired_effects(nominal, metric, baseline)
            file_level = nominal.groupby(["condition_id", "file", "policy"], as_index=False)[metric].mean()
            wide = file_level.pivot(index=["condition_id", "file"], columns="policy", values=metric).dropna(subset=[baseline, "C7-Core"])
            zero_denom = int((wide[baseline].abs() <= 1e-12).sum())
            draws = impl.hierarchical_bootstrap(nominal, metric, baseline, rng)
            rows.append({
                "metric": metric,
                "comparison": f"C7 vs {baseline}",
                "estimand": "mean file-relative effect within transition, then equal-transition mean",
                "point_pct": point,
                "bootstrap_mean_pct": float(draws.mean()),
                "bootstrap_se_pct": float(draws.std(ddof=1)),
                "ci_low_pct": float(np.percentile(draws, 2.5)),
                "ci_high_pct": float(np.percentile(draws, 97.5)),
                "n_conditions": int(by_condition.shape[0]),
                "n_passes": int(len(wide)),
                "zero_denominator_rows": zero_denom,
                "bootstrap_draws": N_BOOT,
            })
    pd.DataFrame(rows).to_csv(RESULTS / "paired_effect_bootstrap.csv", index=False, encoding="utf-8-sig")


def write_nondominated(nominal: pd.DataFrame) -> None:
    rng = np.random.default_rng(BOOT_SEED)
    point, probabilities = impl.non_dominated_probability(nominal, rng)
    probabilities["bootstrap_draws"] = N_BOOT
    probabilities.to_csv(RESULTS / "bootstrap_non_dominated_probability.csv", index=False, encoding="utf-8-sig")


def write_condition_and_loco(nominal: pd.DataFrame) -> None:
    rows, loco_rows = [], []
    for metric in METRICS4:
        for baseline in ["PID", "OC-PID", "C2"]:
            _, by_condition = impl.paired_effects(nominal, metric, baseline)
            for condition, value in by_condition.items():
                rows.append({"condition": condition, "comparison": f"C7 vs {baseline}", "metric": metric, "relative_effect_pct": value})
            for omitted in by_condition.index:
                remaining = by_condition.drop(index=omitted)
                loco_rows.append({"comparison": f"C7 vs {baseline}", "metric": metric, "omitted_condition": omitted, "loco_effect_pct": float(remaining.mean())})
    conditions = pd.DataFrame(rows)
    conditions.to_csv(RESULTS / "condition_level_effects.csv", index=False, encoding="utf-8-sig")
    loco = pd.DataFrame(loco_rows)
    out = []
    for (comparison, metric), group in conditions.groupby(["comparison", "metric"]):
        per_condition = group.set_index("condition")["relative_effect_pct"]
        loco_sub = loco[(loco.comparison.eq(comparison)) & (loco.metric.eq(metric))]
        out.append({
            "comparison": comparison,
            "metric": metric,
            "conditions_favouring_C7": int((per_condition < 0).sum()),
            "n_conditions": int(len(per_condition)),
            "condition_min_pct": float(per_condition.min()),
            "condition_max_pct": float(per_condition.max()),
            "loco_min_pct": float(loco_sub.loco_effect_pct.min()),
            "loco_max_pct": float(loco_sub.loco_effect_pct.max()),
        })
    pd.DataFrame(out).to_csv(RESULTS / "leave_one_condition_out.csv", index=False, encoding="utf-8-sig")


def write_canonical_tables() -> None:
    nominal = load_canonical()  # full policy names for the paired-effect machinery
    write_pairwise_bootstrap(nominal)
    write_nondominated(nominal)
    write_condition_and_loco(nominal)

    episodes = load_episodes()
    episodes["policy"] = episodes["policy"].astype(str).replace({"C7-Core": "C7", "ADRC": "OC-PID"})
    # Attach condition identity to the per-file episode rows (one file per pass
    # in the canonical archive).
    condition_map = nominal.groupby("pass_id", as_index=False)["condition_id"].first()
    episodes = episodes.merge(condition_map, on="pass_id", how="left")
    episodes["file"] = episodes["pass_id"]
    metrics = METRICS4 + ["S_out", "amplitude_projection_ratio", "sat_ratio_logged"]
    # Publication tables use the short C7 name.
    nominal["policy"] = nominal["policy"].replace({"C7-Core": "C7"})

    # Canonical per-pass metrics (full schema + identity + episode merge).
    inventory = pd.read_csv(EXPANDED / "production_batch_map.csv")
    pass_metrics = publication_conditions(nominal).merge(
        inventory[["pass_id", "batch", "role", "transition"]], on="pass_id", how="left"
    )
    episode_cols = ["integrated_severity_excess_m", "episodes_per_100m", "peak_excess_norm"]
    pass_metrics = pass_metrics.merge(
        episodes[["policy", "pass_id"] + episode_cols], on=["policy", "pass_id"], how="left"
    )
    # Remove engine-internal promotional labels and machine-specific paths from
    # the distributed publication table without altering any numerical field.
    if "control_label" in pass_metrics:
        pass_metrics["control_label"] = pass_metrics["policy"].map({
            "PID": "PID",
            "OC-PID": "Observer-compensated PID (OC-PID)",
            "C2": "C2 predictive PID",
            "C7": "C7-Core",
        })
    if "predictor_bank_path" in pass_metrics:
        has_predictor = (
            pass_metrics["predictor_bank_path"].notna()
            & pass_metrics["predictor_bank_path"].astype(str).str.strip().ne("")
        )
        pass_metrics.loc[has_predictor, "predictor_bank_path"] = "models/final_thickness_model.pt"
    pass_metrics.to_csv(RESULTS / "canonical_nominal_pass_metrics.csv", index=False, encoding="utf-8-sig")

    # Condition-weighted canonical summary + condition-weighted episode rows.
    cw = condition_weighted(nominal, metrics, ["policy"]).set_index("policy")
    ep_cw = condition_weighted(episodes, ["episodes_per_100m", "integrated_severity_excess_m", "peak_excess_norm"], ["policy"]).set_index("policy")
    summary = pd.DataFrame({
        "policy": cw.index,
        "composite_normalized_RMS": cw["composite_normalized_RMS"],
        "S_out": cw["S_out"],
        "S_excess_mean": cw["S_excess_mean"],
        "S_excess_cvar95": cw["S_excess_cvar95"],
        "TV_L_per_100m": cw["TV_L_per_100m"],
        "amplitude_projection_ratio": cw["amplitude_projection_ratio"],
        "sat_ratio_logged": cw["sat_ratio_logged"],
        "integrated_severity_excess_m": ep_cw["integrated_severity_excess_m"],
        "episodes_per_100m": ep_cw["episodes_per_100m"],
        "n_conditions": 8,
        "aggregation": AGG_STR,
    }).reset_index(drop=True)
    summary.to_csv(RESULTS / "canonical_nominal_condition_weighted.csv", index=False, encoding="utf-8-sig")

    ep_table = pd.DataFrame({
        "policy": ep_cw.index,
        "episodes_per_100m": ep_cw["episodes_per_100m"],
        "integrated_severity_excess_m": ep_cw["integrated_severity_excess_m"],
        "peak_excess_norm": ep_cw["peak_excess_norm"],
    }).reset_index(drop=True)
    ep_table.to_csv(RESULTS / "canonical_episode_condition_weighted.csv", index=False, encoding="utf-8-sig")

    # Aggregate-ratio relative effects.
    rows = []
    for baseline in ["PID", "OC-PID", "C2"]:
        for metric in METRICS4 + ["S_out"]:
            c7 = float(cw.loc["C7", metric])
            base = float(cw.loc[baseline, metric])
            rows.append({
                "comparison": f"C7 vs {baseline}",
                "metric": metric,
                "c7_aggregate": c7,
                "baseline_aggregate": base,
                "relative_effect_pct": 100.0 * (c7 - base) / base if abs(base) > 1e-12 else np.nan,
                "zero_denominator": bool(abs(base) <= 1e-12),
                "estimand": "ratio of equal-transition aggregate absolute metrics",
            })
    pd.DataFrame(rows).to_csv(RESULTS / "canonical_relative_effects.csv", index=False, encoding="utf-8-sig")


def policy_order(frame: pd.DataFrame, col: str = "policy") -> pd.DataFrame:
    order = {"C2": 0, "C7-Core": 1, "OC-PID": 2, "PID": 3}
    frame = frame.copy()
    frame["_o"] = frame[col].map(order)
    return frame.sort_values("_o").drop(columns="_o")


def write_sensitivity_tables() -> None:
    # Response family.
    emulator = publication_conditions(pd.read_csv(SENS / "01_emulator_compact" / "stage_allocation_ablation_metrics.csv"))
    emulator["policy"] = emulator["control"].map(impl.policy_name)
    em_cw = condition_weighted(emulator, METRICS4 + ["output_clip_ratio"], ["scenario", "policy"])
    em_cw[["scenario", "policy"] + METRICS4 + ["output_clip_ratio"]].to_csv(
        RESULTS / "response_family_condition_weighted.csv", index=False, encoding="utf-8-sig"
    )
    rows = []
    for scenario, group in emulator.groupby("scenario"):
        work = group.copy()
        for metric in METRICS4:
            for baseline in ["PID", "OC-PID"]:
                point, _ = impl.paired_effects(work, metric, baseline)
                rows.append({"scenario": scenario, "metric": metric, "comparison": f"C7-Core vs {baseline}", "effect_pct": point, "direction_favours_c7": point < 0})
    pd.DataFrame(rows).to_csv(RESULTS / "response_family_effects.csv", index=False, encoding="utf-8-sig")

    # Amplitude envelope.
    amplitude = publication_conditions(pd.read_csv(SENS / "02_amplitude_envelope" / "stage_allocation_ablation_metrics.csv"))
    amplitude["policy"] = amplitude["control"].map(impl.policy_name)
    amp_cw = condition_weighted(
        amplitude,
        METRICS4 + ["amplitude_active_speed_ratio", "amplitude_active_gap_ratio", "amplitude_active_shape_ratio", "raw_to_applied_L2_mean"],
        ["amplitude_bound", "policy"],
    )
    quality = policy_order(amp_cw).rename(columns={
        "composite_normalized_RMS": "RMS_c",
        "TV_L_per_100m": "TV/100 m",
        "S_excess_mean": BAND_MEAN_LABEL,
        "S_excess_cvar95": BAND_TAIL_LABEL,
        "amplitude_bound": "Bound",
        "policy": "Policy",
    })
    quality[["Policy", "Bound", "RMS_c", "TV/100 m", BAND_MEAN_LABEL, BAND_TAIL_LABEL]].to_csv(
        RESULTS / "Table_S12a_amplitude_quality.csv", index=False, encoding="utf-8-sig"
    )
    channels = policy_order(amp_cw).rename(columns={
        "amplitude_active_speed_ratio": "Speed active",
        "amplitude_active_gap_ratio": "Gap active",
        "amplitude_active_shape_ratio": "Shape active",
        "raw_to_applied_L2_mean": "Mean raw-applied L2",
        "amplitude_bound": "Bound",
        "policy": "Policy",
    })
    channels[["Policy", "Bound", "Speed active", "Gap active", "Shape active", "Mean raw-applied L2"]].to_csv(
        RESULTS / "Table_S12b_amplitude_channels.csv", index=False, encoding="utf-8-sig"
    )

    # Anti-windup.
    anti = publication_conditions(pd.read_csv(SENS / "03_anti_windup" / "stage_allocation_ablation_metrics.csv"))
    anti["policy"] = anti["control"].map(impl.policy_name)
    anti_cw = condition_weighted(anti, METRICS4, ["anti_windup_mode", "policy"])
    anti_cw = policy_order(anti_cw).rename(columns={
        "composite_normalized_RMS": "RMS_c",
        "TV_L_per_100m": "TV/100 m",
        "S_excess_mean": BAND_MEAN_LABEL,
        "S_excess_cvar95": BAND_TAIL_LABEL,
        "anti_windup_mode": "Mode",
        "policy": "Policy",
    })
    anti_cw[["Policy", "Mode", "RMS_c", "TV/100 m", BAND_MEAN_LABEL, BAND_TAIL_LABEL]].to_csv(
        RESULTS / "Table_S12c_anti_windup.csv", index=False, encoding="utf-8-sig"
    )

    # Numerical emulator state guardrail.
    clip_sweep = publication_conditions(pd.read_csv(SENS / "04_output_clip" / "stage_allocation_ablation_metrics.csv"))
    noclip = publication_conditions(pd.read_csv(SENS / "05_no_output_clip" / "stage_allocation_ablation_metrics.csv"))
    clip = pd.concat([clip_sweep, noclip], ignore_index=True, sort=False)
    clip["policy"] = clip["control"].map(impl.policy_name)
    clip_cw = condition_weighted(clip, METRICS4 + ["output_clip_ratio"], ["output_clip_scale", "policy"])
    clip_activation_cw = clip_cw[["output_clip_scale", "policy", "output_clip_ratio"]].rename(
        columns={"output_clip_ratio": "condition_equal_mean_file_ratio"}
    )
    clip_cw = policy_order(clip_cw).rename(columns={
        "composite_normalized_RMS": "RMS_c",
        "S_excess_mean": BAND_MEAN_LABEL,
        "S_excess_cvar95": BAND_TAIL_LABEL,
        "output_clip_scale": "Guardrail scale",
        "output_clip_ratio": "Activation",
        "policy": "Policy",
    })
    clip_cw[["Policy", "Guardrail scale", "RMS_c", BAND_MEAN_LABEL, BAND_TAIL_LABEL, "Activation"]].to_csv(
        RESULTS / "Table_S12d_state_guardrail.csv", index=False, encoding="utf-8-sig"
    )

    clip_counts = clip.groupby(
        ["output_clip_scale", "policy", "condition_id"], as_index=False
    ).agg(
        event_count=("output_clip_count", "sum"),
        eligible_transitions=("output_clip_eligible_transition_count", "sum"),
        T_count=("output_clip_T_count", "sum"),
        h_count=("output_clip_h_count", "sum"),
        S_count=("output_clip_S_count", "sum"),
        files=("file", "nunique"),
    )
    clip_counts["pooled_event_rate"] = (
        clip_counts.event_count / clip_counts.eligible_transitions.clip(lower=1)
    )
    clip_counts.to_csv(
        RESULTS / "output_guardrail_activation_by_condition.csv",
        index=False, encoding="utf-8-sig",
    )
    clip_summary = clip_counts.groupby(
        ["output_clip_scale", "policy"], as_index=False
    ).agg(
        event_count=("event_count", "sum"),
        eligible_transitions=("eligible_transitions", "sum"),
        T_count=("T_count", "sum"),
        h_count=("h_count", "sum"),
        S_count=("S_count", "sum"),
        equal_condition_pooled_transition_rate=("pooled_event_rate", "mean"),
        conditions=("condition_id", "nunique"),
    )
    clip_summary["pooled_event_rate"] = (
        clip_summary.event_count / clip_summary.eligible_transitions.clip(lower=1)
    )
    clip_summary = clip_summary.merge(
        clip_activation_cw, on=["output_clip_scale", "policy"], how="left"
    )
    clip_summary.to_csv(
        RESULTS / "output_guardrail_activation_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    # Canonical (scale 8) activation ledger at the segment level.  Counts use
    # only transitions for which a next row exists inside the same segment.
    canonical_steps = pd.read_csv(
        CANON / "stage_allocation_ablation_step_logs.csv.gz",
        compression="gzip",
        usecols=[
            "block", "control", "condition_id", "pass_id", "segment_id",
            "output_transition_evaluated", "output_clip_any_active",
            "output_clip_T_active", "output_clip_h_active", "output_clip_S_active",
        ],
    )
    canonical_steps = canonical_steps.loc[
        canonical_steps.block.eq("data1_controller_search")
    ].copy()
    canonical_steps["policy"] = canonical_steps["control"].map(impl.policy_name)
    segment_guard = canonical_steps.groupby(
        ["policy", "condition_id", "pass_id", "segment_id"], as_index=False
    ).agg(
        eligible_transitions=("output_transition_evaluated", "sum"),
        event_count=("output_clip_any_active", "sum"),
        T_count=("output_clip_T_active", "sum"),
        h_count=("output_clip_h_active", "sum"),
        S_count=("output_clip_S_active", "sum"),
    )
    segment_guard["event_rate"] = (
        segment_guard.event_count
        / segment_guard.eligible_transitions.replace(0, np.nan)
    )
    segment_guard.to_csv(
        RESULTS / "output_guardrail_activation_by_segment.csv",
        index=False, encoding="utf-8-sig",
    )

    architecture = publication_conditions(pd.read_csv(
        SENS / "07_architecture_ablation" / "stage_allocation_ablation_metrics.csv"
    ))
    arch_cw = condition_weighted(architecture, METRICS4 + ["S_out"], ["control"])
    arch_cw.rename(columns={
        "control": "Variant",
        "composite_normalized_RMS": "RMS_c",
        "TV_L_per_100m": "TV/100 m",
        "S_out": BAND_FREQUENCY_LABEL,
        "S_excess_mean": BAND_MEAN_LABEL,
        "S_excess_cvar95": BAND_TAIL_LABEL,
    }).to_csv(RESULTS / "strict_c7_component_ablation.csv", index=False, encoding="utf-8-sig")


def write_expanded_tables() -> None:
    expanded = pd.read_csv(EXPANDED / "expanded_26pass_batch_weighted_policy_summary.csv")
    expanded.to_csv(RESULTS / "expanded_batch_weighted_absolute.csv", index=False, encoding="utf-8-sig")
    rows = []
    for baseline in ["PID", "OC-PID"]:
        for metric in METRICS4:
            c7 = float(expanded.loc[expanded.policy.eq("C7"), metric].iloc[0])
            base = float(expanded.loc[expanded.policy.eq(baseline), metric].iloc[0])
            rows.append({
                "comparison": f"C7 vs {baseline}",
                "metric": metric,
                "c7_aggregate": c7,
                "baseline_aggregate": base,
                "relative_effect_pct": 100.0 * (c7 - base) / base if abs(base) > 1e-12 else np.nan,
                "zero_denominator": bool(abs(base) <= 1e-12),
                "estimand": "ratio of final absolute metrics after transition-within-batch and equal-batch aggregation",
            })
    pd.DataFrame(rows).to_csv(RESULTS / "expanded_batch_relative_effects.csv", index=False, encoding="utf-8-sig")

    shutil.copy2(EXPANDED / "repeat_sequence_extension_metrics.csv", RESULTS / "repeat_sequence_extension_metrics.csv")
    shutil.copy2(EXPANDED / "data2_leave_one_batch_out_policy_summary.csv", RESULTS / "data2_leave_one_batch_out.csv")

    # Publication-level fixed-dt versus segment-local comparison. The
    # intermediate per-pass file is used only to form this registered table
    # and is not part of the publication-level result set.
    fixed = pd.read_csv(EXPANDED / "counterfactual_fixed_0p50s_10pass_metrics.csv")
    canonical = load_canonical()
    canonical["policy"] = canonical["policy"].replace({"C7-Core": "C7"})
    episodes = load_episodes()
    episodes["policy"] = episodes["policy"].astype(str).replace({"C7-Core": "C7", "ADRC": "OC-PID"})
    episode_cols = ["integrated_severity_excess_m", "episodes_per_100m"]
    canonical = canonical.merge(episodes[["policy", "pass_id"] + episode_cols], on=["policy", "pass_id"], how="left")
    metrics = METRICS4 + ["S_out", "amplitude_projection_ratio", "sat_ratio_logged", "integrated_severity_excess_m", "episodes_per_100m"]
    cw_gap = condition_weighted(canonical, metrics, ["policy"]).set_index("policy")
    cw_fix = condition_weighted(fixed, metrics, ["policy"]).set_index("policy")
    rows = []
    for policy in cw_gap.index:
        row = {"policy": policy}
        for metric in metrics:
            row[f"{metric}_gap_v2"] = float(cw_gap.loc[policy, metric])
            row[f"{metric}_fixed0p50"] = float(cw_fix.loc[policy, metric])
            row[f"{metric}_relative_delta_pct"] = float(
                100.0 * (cw_fix.loc[policy, metric] - cw_gap.loc[policy, metric]) / cw_gap.loc[policy, metric]
            ) if abs(cw_gap.loc[policy, metric]) > 1e-12 else np.nan
        row["n_conditions_gap_v2"] = 8
        row["aggregation_gap_v2"] = AGG_STR
        row["n_conditions_fixed0p50"] = 8
        row["aggregation_fixed0p50"] = AGG_STR
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "fixed_dt_vs_segment_local_dt.csv", index=False, encoding="utf-8-sig")


def write_time_protocol_qa(batch_root: Path) -> None:
    from segmented_archived_timebase import (
        DISTANCE_STEP_RATIO_HIGH,
        DISTANCE_STEP_RATIO_LOW,
        TIMESTAMP_DISTANCE_RATIO_HIGH,
        TIMESTAMP_DISTANCE_RATIO_LOW,
        derive_segmented_archived_timebase,
    )

    runner.protocol.PROJECT_DIR = archive.FULL_PROJECT
    suite, replay = runner.protocol.load_suite_modules()
    archive.BATCH_ROOT = Path(batch_root)
    all_passes, _ = archive.load_batch_passes(runner.protocol, suite, replay)
    inventory = pd.read_csv(EXPANDED / "production_batch_map.csv")
    rows = []
    scale_rows = []
    for rp in all_passes:
        frame = rp.df
        info = inventory[inventory.pass_id.eq(rp.pass_id)].iloc[0]
        timebase = derive_segmented_archived_timebase(
            frame, 0.81298828125,
            DISTANCE_STEP_RATIO_LOW, DISTANCE_STEP_RATIO_HIGH,
        )
        timestamp = pd.to_numeric(frame["raw_timestamp"], errors="coerce").to_numpy(float)
        raw_dt = np.diff(timestamp, prepend=np.nan)
        raw_chronological = np.isfinite(raw_dt) & (raw_dt > 0.0)
        span = float(np.nanmax(frame["strip_length_actual"]) - np.nanmin(frame["strip_length_actual"]))
        observed = float(timebase.observed_dlength_m.sum())
        rows.append({
            "pass_id": info.pass_id,
            "batch": info.batch,
            "role": info.role,
            "segments": int(timebase.segment_id.max() + 1),
            "rows": int(len(frame)),
            "raw_positive_timestamp_count": int(raw_chronological.sum()),
            "raw_timestamp_consistent_count": int(timebase.raw_timestamp_consistent.sum()),
            "effective_raw_timestamp_count": int((timebase.source == "raw_timestamp_consistent").sum()),
            "effective_distance_over_speed_count": int((timebase.source == "distance_over_speed").sum()),
            "segment_start_local_median_count": int((timebase.source == "segment_start_local_median").sum()),
            "segment_start_global_median_count": int((timebase.source == "segment_start_global_median").sum()),
            "segment_local_median_fallback_count": int((timebase.source == "segment_local_median_fallback").sum()),
            "effective_source_count_sum_equals_rows": bool(
                sum(int((timebase.source == label).sum()) for label in [
                    "raw_timestamp_consistent", "distance_over_speed",
                    "segment_start_local_median", "segment_start_global_median",
                    "segment_local_median_fallback",
                ]) == len(frame)
            ),
            "segment_start_count": int(timebase.segment_start.sum()),
            "distance_step_ratio_low": DISTANCE_STEP_RATIO_LOW,
            "distance_step_ratio_high": DISTANCE_STEP_RATIO_HIGH,
            "timestamp_distance_ratio_low": TIMESTAMP_DISTANCE_RATIO_LOW,
            "timestamp_distance_ratio_high": TIMESTAMP_DISTANCE_RATIO_HIGH,
            "max_positive_raw_timestamp_gap_s": float(np.nanmax(raw_dt[raw_chronological])),
            "max_accepted_raw_timestamp_dt_s": float(
                np.nanmax(raw_dt[timebase.raw_timestamp_consistent])
            ),
            "median_dt_control_s": float(np.median(timebase.dt_control_s)),
            "max_dt_control_s": float(np.max(timebase.dt_control_s)),
            "observed_distance_sum_m": observed,
            "gap_distance_sum_m": float(timebase.gap_distance_unknown_m.sum()),
            "file_length_span_m": span,
            "distance_ratio": float(observed / span) if span > 1e-9 else np.nan,
            "control_dt_locality_check": bool(np.isfinite(timebase.dt_control_s).all() & (timebase.dt_control_s > 0).all()),
            "segment_start_distance_zero": bool((timebase.observed_dlength_m[timebase.segment_start] == 0).all()),
        })
        scale_rows.append({
            "pass_id": info.pass_id,
            "role": info.role,
            "entry_thickness_mm": rp.p.entry_thickness,
            "exit_thickness_mm": rp.p.exit_thickness,
            "scale_source": rp.p.scale_source,
            "scale_lock_id": rp.p.scale_lock_id,
            "scale_lock_sha256": rp.p.scale_lock_sha256,
            "scale_match_pass_id": rp.p.scale_match_pass_id,
            "scale_match_log_distance": rp.p.scale_match_log_distance,
            "tension_scale": rp.p.tension_scale,
            "thickness_scale": rp.p.thickness_scale,
            "flatness_scale": rp.p.flatness_scale,
            "rollforce_scale": rp.p.rollforce_scale,
            "radial_scale": rp.p.radial_scale,
            "offcenter_scale": rp.p.offcenter_scale,
        })
    pd.DataFrame(rows).to_csv(RESULTS / "time_protocol_numeric_QA.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scale_rows).to_csv(
        RESULTS / "data1_scale_lock_application.csv", index=False, encoding="utf-8-sig"
    )


def write_static_tables() -> None:
    # Locked controller settings from the frozen lock.
    configs = json.loads((ROOT / "config" / "selected_controller_configs.json").read_text(encoding="utf-8"))
    rows = []
    for item in configs:
        row = dict(item)
        row["policy"] = archive.POLICY_NAMES[item["label"]]
        row["normalized_amplitude_bound"] = 0.55
        row["move_limit_reference_dt_s"] = 0.5
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "locked_controller_settings.csv", index=False, encoding="utf-8-sig")

    # OPC UA loopback tables.
    opc = json.loads((ROOT / "outputs" / "opcua" / "results.json").read_text(encoding="utf-8"))
    pd.DataFrame([
        {"Item": "Cycles", "Result": int(opc["cycles"])},
        {"Item": "Server restarts / client reconnects", "Result": f"{int(opc['server_restart_events'])} / {int(opc['client_reconnects'])}"},
        {"Item": "Latency p95 / p99 (ms)", "Result": f"{float(opc['latency_ms_p95']):.3f} / {float(opc['latency_ms_p99']):.3f}"},
        {"Item": "Command-boundary violations", "Result": int(opc["command_boundary_violations"])},
    ]).to_csv(RESULTS / "opcua_loopback_summary.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(ROOT / "outputs" / "opcua" / "results.json", RESULTS / "local_opcua_loopback_results.json")

    for name in ["fault_diagnostic_summary.csv", "fault_diagnostic_confusion_matrix.csv", "fault_event_log.csv"]:
        shutil.copy2(ROOT / "outputs" / "faults" / name, RESULTS / name)
    for name in ["reference_corridor_coverage_sensitivity.csv", "reference_corridor_phase_envelopes.csv", "reference_corridor_policy_effects.csv"]:
        src = ROOT / "outputs" / "corridor" / name
        if src.exists():
            shutil.copy2(src, RESULTS / name)
    corridor_check = ROOT / "outputs" / "corridor" / "coverage_sensitivity_hard_checks.json"
    if corridor_check.exists():
        shutil.copy2(corridor_check, RESULTS / corridor_check.name)
    lock = json.loads((ROOT / "config" / "data1_scale_lock.json").read_text(encoding="utf-8"))
    pd.DataFrame(lock["entries"]).to_csv(
        RESULTS / "data1_scale_lock_catalog.csv", index=False, encoding="utf-8-sig"
    )
    for name in ["source_sequence_relocking_nominal_summary.csv", "source_sequence_relocking_selected_configs.csv"]:
        shutil.copy2(ROOT / "outputs" / "relocking" / name, RESULTS / name)
    shutil.copy2(ROOT / "outputs" / "seed_sensitivity" / "seed_sensitivity_summary.csv", RESULTS / "seed_sensitivity_summary.csv")
    stress = (
        ROOT / "outputs" / "stress_root" /
        "timestamp_matched_stress_clustered_effects.csv"
    )
    if stress.exists():
        shutil.copy2(stress, RESULTS / "stress_clustered_effects.csv")
    command = ROOT / "outputs" / "command_audit" / "data2_command_audit_summary.csv"
    if command.exists():
        shutil.copy2(command, RESULTS / "data2_command_audit_summary.csv")
    energy = (
        ROOT / "outputs" / "sensitivity" / "04_analysis" /
        "command_innovation_energy_audit.csv"
    )
    if energy.exists():
        shutil.copy2(energy, RESULTS / energy.name)
    maturity = (
        ROOT / "outputs" / "canonical" /
        "10_appendix_audit" / "10_maturity_headroom_summary.csv"
    )
    if maturity.exists():
        shutil.copy2(maturity, RESULTS / "maturity_headroom_summary.csv")
    farch = ROOT / "outputs" / "farch_forward_diagnostic"
    for name in [
        "farch_forward_summary.csv",
        "farch_forward_by_pass.csv",
        "farch_forward_by_pass_segment.csv",
        "farch_recorded_envelopes.csv",
        "farch_recorded_envelope_occupancy.csv",
        "farch_recorded_envelope_occupancy_by_pass.csv",
        "farch_forward_audit.json",
    ]:
        src = farch / name
        if src.exists():
            shutil.copy2(src, RESULTS / name)
    replay_distribution = ROOT / "outputs" / "replay_distribution"
    for name in [
        "replay_flatness_distribution_summary.csv",
        "replay_flatness_distribution_stratified.csv",
        "replay_flatness_distribution_audit.json",
    ]:
        src = replay_distribution / name
        if src.exists():
            shutil.copy2(src, RESULTS / name)
    pid_audit = ROOT / "outputs" / "pid_selection_audit"
    for name in [
        "pid_full_grid.csv", "pid_locked_params.json",
        "pid_selection_audit.json",
    ]:
        src = pid_audit / name
        if src.exists():
            shutil.copy2(src, RESULTS / name)


def rebuild_manifest() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if (
            path.is_file()
            and ".git" not in path.parts
            and "outputs" not in path.parts
            and "__pycache__" not in path.parts
            and path.suffix not in {".log", ".gz", ".pyc"}
        ):
            rel = path.relative_to(ROOT).as_posix()
            if rel == "MANIFEST_SHA256.csv":
                continue  # the manifest cannot hash itself
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"relative_path": rel, "bytes": path.stat().st_size, "sha256": digest})
    pd.DataFrame(rows).to_csv(ROOT / "MANIFEST_SHA256.csv", index=False, encoding="utf-8-sig")
    print(f"[manifest] {len(rows)} files hashed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--batch-root", required=False, type=Path, default=None)
    args = parser.parse_args()
    if args.manifest_only:
        rebuild_manifest()
        return 0
    if args.batch_root is None:
        parser.error("--batch-root is required unless --manifest-only is used")
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_canonical_tables()
    write_sensitivity_tables()
    write_expanded_tables()
    write_time_protocol_qa(args.batch_root)
    write_static_tables()
    print("[publish] tables written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
