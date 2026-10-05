"""Rebuild and decompose canonical flatness recorded-band metrics.

Legacy ``S_out``, ``S_excess_mean`` and ``S_excess_cvar95`` fields are kept
for compatibility. They measure two-sided distance from a phase-specific
Data1 recorded operating band; they are not one-sided flatness-quality or
safety metrics. This script exposes the lower- and upper-side components.

Output-guardrail flags describe the transition from row k to row k+1. The
guardrail ledger shifts each flag by one row within a pass/segment before
associating it with an error row. That ledger is descriptive, not causal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tail_statistics import upper_tail_count


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STEPS = (
    ROOT / "outputs" / "canonical" / "03_nominal"
    / "stage_allocation_ablation_step_logs.csv.gz"
)
DEFAULT_METRICS = (
    ROOT / "outputs" / "canonical" / "03_nominal"
    / "stage_allocation_ablation_metrics.csv"
)
DEFAULT_ENVELOPE = (
    ROOT / "config" / "reference_run" / "03_nominal" / "audit"
    / "data1_stage_quantile_envelope_p025_p975.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "recorded_band_decomposition"
DEFAULT_PUBLISH = ROOT / "results"

AGGREGATION_FILE = "one replay pass/file"
AGGREGATION_CONDITION = "files first averaged within thickness transition"
AGGREGATION_SUMMARY = (
    "equal thickness transition; files first averaged within transition"
)
INTERPRETATION = (
    "two-sided distance from the Data1 recorded operating band; "
    "not a one-sided flatness-quality or safety metric"
)
GUARDRAIL_NOTE = (
    "descriptive row composition only; transition flags are shifted from "
    "row k to row k+1 within segment; no causal attribution"
)

FILE_NAME = "recorded_band_decomposition_by_file.csv"
CONDITION_NAME = "recorded_band_decomposition_by_condition.csv"
SUMMARY_NAME = "recorded_band_decomposition_summary.csv"
GUARDRAIL_NAME = "recorded_band_guardrail_row_composition.csv"
AUDIT_NAME = "recorded_band_decomposition_audit.json"


def policy_name(label: str) -> str:
    value = str(label)
    if value.startswith("PID"):
        return "PID"
    if value.startswith("ADRC") or value.startswith("OC-PID"):
        return "OC-PID"
    if value.startswith("C2"):
        return "C2"
    if value.startswith("C7"):
        return "C7-Core"
    raise ValueError(f"Unknown policy label: {label}")


def publication_conditions(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result.loc[result["pass_id"].astype(str).eq("P06"), "condition_id"] = (
        "0.628->0.458 acceleration"
    )
    return result


def top_tail_indices(values: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    """Return deterministic largest-ceil((1-q)*N) indices for 0 <= q < 1.

    Finite pass-level band distances are expected by the replay contract.
    Stable sorting preserves the historical deterministic handling of ties.
    """
    values = np.asarray(values, dtype=float)
    k = upper_tail_count(values.size, quantile)
    if values.size == 0:
        return np.array([], dtype=int)
    return np.argsort(values, kind="stable")[-k:]


def summarize_file(group: pd.DataFrame) -> pd.Series:
    total = group["total_distance"].to_numpy(float)
    below = group["below_distance"].to_numpy(float)
    above = group["above_distance"].to_numpy(float)
    z = group["S_normalized"].to_numpy(float)
    tail = top_tail_indices(total)
    below_mask = below > 0.0
    above_mask = above > 0.0
    inside_mask = ~(below_mask | above_mask)
    return pd.Series({
        "n_rows": int(len(group)),
        "below_count": int(below_mask.sum()),
        "inside_count": int(inside_mask.sum()),
        "above_count": int(above_mask.sum()),
        "below_frequency": float(below_mask.mean()),
        "above_frequency": float(above_mask.mean()),
        "recorded_band_departure_frequency": float((total > 0.0).mean()),
        "below_mean_distance": float(below.mean()),
        "above_mean_distance": float(above.mean()),
        "two_sided_mean_distance": float(total.mean()),
        "tail_rows": int(tail.size),
        "below_cvar95_component": float(below[tail].mean()),
        "above_cvar95_component": float(above[tail].mean()),
        "two_sided_cvar95": float(total[tail].mean()),
        "S_normalized_mean": float(z.mean()),
        "S_normalized_median": float(np.median(z)),
        "S_normalized_rms": float(np.sqrt(np.mean(np.square(z)))),
    })


METRIC_COLUMNS = [
    "below_frequency", "above_frequency",
    "recorded_band_departure_frequency", "below_mean_distance",
    "above_mean_distance", "two_sided_mean_distance",
    "below_cvar95_component", "above_cvar95_component",
    "two_sided_cvar95", "S_normalized_mean", "S_normalized_median",
    "S_normalized_rms",
]


def aggregate_by_condition(file_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (policy, condition), group in file_table.groupby(
        ["policy", "condition_id"], sort=True
    ):
        row: dict[str, object] = {
            "policy": policy, "condition_id": condition,
            "n_files": int(len(group)), "n_rows": int(group["n_rows"].sum()),
            "below_count": int(group["below_count"].sum()),
            "inside_count": int(group["inside_count"].sum()),
            "above_count": int(group["above_count"].sum()),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = float(group[metric].mean())
        row["aggregation"] = AGGREGATION_CONDITION
        row["interpretation"] = INTERPRETATION
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_summary(condition_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, group in condition_table.groupby("policy", sort=True):
        row: dict[str, object] = {
            "policy": policy, "n_conditions": int(len(group)),
            "n_files": int(group["n_files"].sum()),
            "n_rows": int(group["n_rows"].sum()),
            "below_count": int(group["below_count"].sum()),
            "inside_count": int(group["inside_count"].sum()),
            "above_count": int(group["above_count"].sum()),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = float(group[metric].mean())
        row["aggregation"] = AGGREGATION_SUMMARY
        row["interpretation"] = INTERPRETATION
        rows.append(row)
    return pd.DataFrame(rows)


def guardrail_composition(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    scopes = {
        "after_any_channel_guardrail": "aligned_output_clip_any_active",
        "after_S_guardrail": "aligned_output_clip_S_active",
    }
    for (policy, condition), condition_rows in rows.groupby(
        ["policy", "condition_id"], sort=True
    ):
        for scope, flag in scopes.items():
            selected = condition_rows.loc[condition_rows[flag].eq(1)]
            total_condition_rows = int(len(condition_rows))
            n = int(len(selected))
            below = selected["below_distance"].to_numpy(float)
            above = selected["above_distance"].to_numpy(float)
            total = selected["total_distance"].to_numpy(float)
            z = selected["S_normalized"].to_numpy(float)
            below_count = int((below > 0.0).sum())
            above_count = int((above > 0.0).sum())
            inside_count = int(n - below_count - above_count)
            output.append({
                "policy": policy, "condition_id": condition, "row_scope": scope,
                "n_files": int(selected["file"].nunique()) if n else 0,
                "condition_rows": total_condition_rows, "scope_rows": n,
                "scope_fraction_of_condition_rows": (
                    float(n / total_condition_rows) if total_condition_rows else np.nan
                ),
                "below_count": below_count, "inside_count": inside_count,
                "above_count": above_count,
                "below_fraction_within_scope": float(below_count / n) if n else np.nan,
                "inside_fraction_within_scope": float(inside_count / n) if n else np.nan,
                "above_fraction_within_scope": float(above_count / n) if n else np.nan,
                "below_mean_distance_within_scope": float(below.mean()) if n else np.nan,
                "above_mean_distance_within_scope": float(above.mean()) if n else np.nan,
                "two_sided_mean_distance_within_scope": float(total.mean()) if n else np.nan,
                "S_normalized_min": float(z.min()) if n else np.nan,
                "S_normalized_median": float(np.median(z)) if n else np.nan,
                "S_normalized_max": float(z.max()) if n else np.nan,
                "positive_8_boundary_rows": int(np.isclose(z, 8.0, atol=1e-12).sum()),
                "negative_8_boundary_rows": int(np.isclose(z, -8.0, atol=1e-12).sum()),
                "interpretation": GUARDRAIL_NOTE,
            })
    return pd.DataFrame(output)


def max_abs_delta(left: pd.Series, right: pd.Series) -> float:
    return float(np.max(np.abs(left.to_numpy(float) - right.to_numpy(float))))


def build_tables(
    steps_path: Path, metrics_path: Path, envelope_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    step_columns = [
        "k", "segment_id", "phase", "control", "pass_id", "condition_id",
        "file", "seed", "repeat_idx", "S_error", "S_control_error",
        "output_transition_evaluated", "output_clip_any_active",
        "output_clip_S_active",
    ]
    steps = pd.read_csv(steps_path, compression="infer", usecols=step_columns)
    metric_columns = [
        "control", "pass_id", "condition_id", "file", "seed", "repeat_idx",
        "flatness_scale", "output_clip_scale", "S_out", "S_excess_mean",
        "S_excess_cvar95",
    ]
    metrics = pd.read_csv(metrics_path, usecols=metric_columns)
    envelope = pd.read_csv(envelope_path)[
        ["phase", "S_norm_lo", "S_norm_hi"]
    ]
    join_keys = [
        "control", "pass_id", "condition_id", "file", "seed", "repeat_idx"
    ]
    if metrics.duplicated(join_keys).any():
        raise RuntimeError("Canonical pass metrics are not unique on replay identity")
    if (metrics["flatness_scale"] <= 0.0).any():
        raise RuntimeError("Every canonical flatness scale must be positive")
    rows = steps.merge(
        metrics[join_keys + ["flatness_scale", "output_clip_scale"]],
        on=join_keys, how="left", validate="many_to_one",
    )
    if rows[["flatness_scale", "output_clip_scale"]].isna().any().any():
        raise RuntimeError("A canonical step row did not match its pass scale")
    rows = rows.merge(envelope, on="phase", how="left", validate="many_to_one")
    if rows[["S_norm_lo", "S_norm_hi"]].isna().any().any():
        unknown = sorted(rows.loc[rows["S_norm_lo"].isna(), "phase"].unique())
        raise RuntimeError(f"Missing phase envelope rows: {unknown}")
    source_error = rows["S_control_error"].where(
        rows["S_control_error"].notna(), rows["S_error"]
    )
    rows["S_normalized"] = source_error / rows["flatness_scale"]
    rows["below_distance"] = (
        rows["S_norm_lo"] - rows["S_normalized"]
    ).clip(lower=0.0)
    rows["above_distance"] = (
        rows["S_normalized"] - rows["S_norm_hi"]
    ).clip(lower=0.0)
    rows["total_distance"] = rows["below_distance"] + rows["above_distance"]
    rows["policy"] = rows["control"].map(policy_name)

    sort_columns = [
        "control", "pass_id", "file", "seed", "repeat_idx", "segment_id", "k"
    ]
    rows = rows.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    segment_keys = sort_columns[:-1]
    grouped = rows.groupby(segment_keys, sort=False, dropna=False)
    for source, target in [
        ("output_transition_evaluated", "aligned_transition_reached"),
        ("output_clip_any_active", "aligned_output_clip_any_active"),
        ("output_clip_S_active", "aligned_output_clip_S_active"),
    ]:
        rows[target] = grouped[source].shift(1, fill_value=0).astype(int)

    rows = publication_conditions(rows)
    file_keys = ["policy", "condition_id", "pass_id", "file"]
    file_table = (
        rows.groupby(file_keys, sort=True, dropna=False)
        .apply(summarize_file, include_groups=False)
        .reset_index()
    )
    for column in [
        "n_rows", "below_count", "inside_count", "above_count", "tail_rows"
    ]:
        file_table[column] = file_table[column].astype(int)
    file_table["aggregation"] = AGGREGATION_FILE
    file_table["interpretation"] = INTERPRETATION
    condition_table = aggregate_by_condition(file_table)
    summary_table = aggregate_summary(condition_table)
    guardrail_table = guardrail_composition(rows)

    anchor = metrics.copy()
    anchor["policy"] = anchor["control"].map(policy_name)
    anchor = publication_conditions(anchor)
    anchor_keys = ["policy", "condition_id", "pass_id", "file"]
    comparison = file_table.merge(
        anchor[anchor_keys + ["S_out", "S_excess_mean", "S_excess_cvar95"]],
        on=anchor_keys, how="left", validate="one_to_one",
    )
    if comparison[["S_out", "S_excess_mean", "S_excess_cvar95"]].isna().any().any():
        raise RuntimeError("A rebuilt file metric did not match its legacy anchor row")
    file_anchor_deltas = {
        "S_out": max_abs_delta(
            comparison["recorded_band_departure_frequency"], comparison["S_out"]
        ),
        "S_excess_mean": max_abs_delta(
            comparison["two_sided_mean_distance"], comparison["S_excess_mean"]
        ),
        "S_excess_cvar95": max_abs_delta(
            comparison["two_sided_cvar95"], comparison["S_excess_cvar95"]
        ),
    }
    additive_mean_delta = float(np.max(np.abs(
        file_table["two_sided_mean_distance"]
        - file_table["below_mean_distance"] - file_table["above_mean_distance"]
    )))
    additive_cvar_delta = float(np.max(np.abs(
        file_table["two_sided_cvar95"] - file_table["below_cvar95_component"]
        - file_table["above_cvar95_component"]
    )))
    original_guardrail = {
        "transition_evaluated": int(rows["output_transition_evaluated"].sum()),
        "any_active": int(rows["output_clip_any_active"].sum()),
        "S_active": int(rows["output_clip_S_active"].sum()),
    }
    aligned_guardrail = {
        "transition_reached": int(rows["aligned_transition_reached"].sum()),
        "any_active_reached": int(rows["aligned_output_clip_any_active"].sum()),
        "S_active_reached": int(rows["aligned_output_clip_S_active"].sum()),
    }
    segment_first = rows.groupby(segment_keys, sort=False, dropna=False).head(1)
    s_boundary_rows = rows.loc[rows["aligned_output_clip_S_active"].eq(1)]
    positive_boundary = int(np.isclose(
        s_boundary_rows["S_normalized"].to_numpy(float), 8.0, atol=1e-12
    ).sum())
    negative_boundary = int(np.isclose(
        s_boundary_rows["S_normalized"].to_numpy(float), -8.0, atol=1e-12
    ).sum())
    audit: dict[str, object] = {
        "status": "PASS",
        "definition": {
            "below": "max(S_norm_lo - S_normalized, 0)",
            "above": "max(S_normalized - S_norm_hi, 0)",
            "total": "below + above",
            "tail": "mean of the largest ceil(0.05*n) total-distance rows per file",
            "tail_component_rule": (
                "below and above use the same rows selected by total distance"
            ),
        },
        "interpretation": INTERPRETATION,
        "aggregation": AGGREGATION_SUMMARY,
        "row_count": int(len(rows)), "file_count": int(len(file_table)),
        "condition_count": int(condition_table["condition_id"].nunique()),
        "policy_count": int(summary_table["policy"].nunique()),
        "legacy_file_anchor_max_abs_delta": file_anchor_deltas,
        "max_abs_additive_mean_identity_delta": additive_mean_delta,
        "max_abs_additive_cvar_identity_delta": additive_cvar_delta,
        "guardrail_alignment": {
            "rule": "transition flags at k are shifted to reached row k+1 within segment",
            "original_flag_counts": original_guardrail,
            "aligned_reached_row_counts": aligned_guardrail,
            "segment_first_rows_with_aligned_flag": int(
                segment_first[[
                    "aligned_transition_reached", "aligned_output_clip_any_active",
                    "aligned_output_clip_S_active",
                ]].to_numpy(int).sum()
            ),
            "aligned_S_rows_at_positive_8": positive_boundary,
            "aligned_S_rows_at_negative_8": negative_boundary,
            "note": GUARDRAIL_NOTE,
        },
    }
    tolerance = 1e-12
    failures = [
        name for name, value in file_anchor_deltas.items() if value > tolerance
    ]
    if additive_mean_delta > tolerance:
        failures.append("mean additivity")
    if additive_cvar_delta > tolerance:
        failures.append("CVaR additivity")
    if original_guardrail["any_active"] != aligned_guardrail["any_active_reached"]:
        failures.append("any-channel guardrail shift count")
    if original_guardrail["S_active"] != aligned_guardrail["S_active_reached"]:
        failures.append("S guardrail shift count")
    if audit["guardrail_alignment"]["segment_first_rows_with_aligned_flag"] != 0:
        failures.append("segment boundary alignment")
    if positive_boundary + negative_boundary != aligned_guardrail["S_active_reached"]:
        failures.append("S guardrail boundary sign accounting")
    if failures:
        audit["status"] = "FAIL"
        audit["failures"] = failures
        raise RuntimeError(f"Recorded-band audit failed: {failures}")
    return file_table, condition_table, summary_table, guardrail_table, audit


def write_outputs(
    tables: tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]
    ], out_dir: Path, publish_dir: Path | None,
) -> None:
    file_table, condition_table, summary_table, guardrail_table, audit = tables
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, pd.DataFrame] = {
        FILE_NAME: file_table, CONDITION_NAME: condition_table,
        SUMMARY_NAME: summary_table, GUARDRAIL_NAME: guardrail_table,
    }
    for name, frame in payloads.items():
        frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
    (out_dir / AUDIT_NAME).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if publish_dir is not None:
        publish_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in payloads.items():
            frame.to_csv(publish_dir / name, index=False, encoding="utf-8-sig")
        (publish_dir / AUDIT_NAME).write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--pass-metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--envelope", type=Path, default=DEFAULT_ENVELOPE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--publish-dir", type=Path, default=DEFAULT_PUBLISH)
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    for path in [args.steps, args.pass_metrics, args.envelope]:
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    tables = build_tables(args.steps, args.pass_metrics, args.envelope)
    write_outputs(
        tables, args.out_dir, None if args.no_publish else args.publish_dir,
    )
    audit = tables[-1]
    print(
        "[recorded-band] PASS: "
        f"{audit['row_count']} rows, {audit['file_count']} policy-pass files, "
        f"legacy max delta "
        f"{max(audit['legacy_file_anchor_max_abs_delta'].values()):.3g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
