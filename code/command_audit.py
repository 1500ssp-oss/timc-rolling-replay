"""Audit locked replay commands without making a plant-performance claim.

Data1 supplies a controller-specific command support envelope.  Data2 is then
screened for finite, bounded, rate-limited and non-oscillatory proposals.  The
result is an action plausibility screen only; it never estimates process output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from supported_command_motion import (
    reset_command_motion,
    supported_command_deltas,
    supported_total_variation,
    supported_transition_mask,
)


CHANNELS = ("speed", "gap", "shape")
BOUND = 0.55
EPS = 1e-12


def longest_run(values: np.ndarray, weights: np.ndarray, segments: np.ndarray) -> tuple[int, float]:
    best_count = current_count = 0
    best_weight = current_weight = 0.0
    previous_segment = segments[0] if len(segments) else -1
    for hit, weight, segment in zip(values.astype(bool), weights, segments):
        if segment != previous_segment:
            current_count, current_weight = 0, 0.0
            previous_segment = segment
        if hit:
            current_count += 1
            current_weight += float(weight)
            if current_count > best_count:
                best_count, best_weight = current_count, current_weight
        else:
            current_count, current_weight = 0, 0.0
    return best_count, best_weight


def ood_score(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    width = np.maximum(upper - lower, 1e-6)
    below = np.maximum(lower - values, 0.0) / width
    above = np.maximum(values - upper, 0.0) / width
    return np.maximum(below, above).max(axis=1)


def data1_support(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for control, group in frame.groupby("control", sort=True):
        values = group[[f"u_{channel}" for channel in CHANNELS]].to_numpy(float)
        lower = np.nanquantile(values, 0.005, axis=0)
        upper = np.nanquantile(values, 0.995, axis=0)
        rows.append({
            "control": control,
            **{f"u_{channel}_p005": lower[i] for i, channel in enumerate(CHANNELS)},
            **{f"u_{channel}_p995": upper[i] for i, channel in enumerate(CHANNELS)},
        })
    values = frame[[f"u_{channel}" for channel in CHANNELS]].to_numpy(float)
    lower = np.nanquantile(values, 0.005, axis=0)
    upper = np.nanquantile(values, 0.995, axis=0)
    rows.append({
        "control": "__pooled_locked_data1__",
        **{f"u_{channel}_p005": lower[i] for i, channel in enumerate(CHANNELS)},
        **{f"u_{channel}_p995": upper[i] for i, channel in enumerate(CHANNELS)},
    })
    return pd.DataFrame(rows)


def audit_group(group: pd.DataFrame, support: pd.Series, shared_support: pd.Series) -> dict[str, float | int | str]:
    # Archive order, not segment-label order: recurring labels are separate runs.
    group = group.sort_values("k", kind="stable").copy()
    applied = group[[f"u_{channel}" for channel in CHANNELS]].to_numpy(float)
    target = group[[f"u_target_unclipped_{channel}" for channel in CHANNELS]].to_numpy(float)
    preclamp = group[[f"u_preclamp_{channel}" for channel in CHANNELS]].to_numpy(float)
    delta = group[[f"du_{channel}" for channel in CHANNELS]].to_numpy(float)
    dt = group["timestamp_dt_s"].to_numpy(float)
    distance = group["distance_step_m"].to_numpy(float)
    segment = group["segment_id"].to_numpy(int)
    pre_projection_tv = supported_total_variation(target, segment)
    post_projection_tv = supported_total_variation(applied, segment)
    reset_motion = reset_command_motion(applied, segment)

    lower = np.array([support[f"u_{channel}_p005"] for channel in CHANNELS], dtype=float)
    upper = np.array([support[f"u_{channel}_p995"] for channel in CHANNELS], dtype=float)
    score = ood_score(applied, lower, upper)
    flagged = score > 0.0
    shared_lower = np.array([shared_support[f"u_{channel}_p005"] for channel in CHANNELS], dtype=float)
    shared_upper = np.array([shared_support[f"u_{channel}_p995"] for channel in CHANNELS], dtype=float)
    shared_score = ood_score(applied, shared_lower, shared_upper)
    shared_flagged = shared_score > 0.0

    components = []
    for prefix, multiplier in (("u_mpc", group["mpc_share"].to_numpy(float)[:, None]),
                               ("u_ff", group["ff_share"].to_numpy(float)[:, None]),
                               ("u_pid", 1.0),
                               ("u_adrc", 1.0)):
        raw = group[[f"{prefix}_{channel}" for channel in CHANNELS]].to_numpy(float)
        components.append(raw * multiplier)
    component_norm_sum = sum(np.linalg.norm(component, axis=1) for component in components)
    cancellation = np.clip(1.0 - np.linalg.norm(target, axis=1) / np.maximum(component_norm_sum, EPS), 0.0, 1.0)

    reversal_count = 0
    for channel_i in range(len(CHANNELS)):
        previous = 0.0
        previous_segment = segment[0] if len(segment) else 0
        for current, current_segment in zip(delta[:, channel_i], segment):
            if current_segment != previous_segment:
                previous = 0.0
                previous_segment = current_segment
            if abs(current) > 1e-10 and abs(previous) > 1e-10 and np.sign(current) != np.sign(previous):
                reversal_count += 1
            if abs(current) > 1e-10:
                previous = current

    # Slew diagnostics share TV's supported-transition convention. Reset-start
    # actions remain in logged du and reset-motion fields, not in peak slew.
    transition_dt = dt[1:][supported_transition_mask(segment, len(applied))]
    valid_dt = np.isfinite(transition_dt) & (transition_dt > 0.0)
    changes = supported_command_deltas(applied, segment)
    rate = np.abs(changes[valid_dt]) / transition_dt[valid_dt, None]
    finite = np.isfinite(np.column_stack([applied, target, preclamp, delta])).all(axis=1)
    bound_violation = np.max(np.abs(applied), axis=1) > BOUND + 1e-10
    rate_limited = group["sat_flag"].to_numpy(int) > 0
    longest_sat_samples, longest_sat_s = longest_run(rate_limited, dt, segment)
    _, longest_sat_m = longest_run(rate_limited, distance, segment)
    longest_ood_samples, longest_ood_s = longest_run(flagged, dt, segment)
    _, longest_ood_m = longest_run(flagged, distance, segment)

    return {
        "control": str(group["control"].iat[0]),
        "pass_id": str(group["pass_id"].iat[0]),
        "file": str(group["file"].iat[0]),
        "n_steps": int(len(group)),
        "duration_s": float(dt.sum()),
        "distance_m": float(distance.sum()),
        "invalid_command_count": int((~finite).sum()),
        "boundary_violation_count": int(bound_violation.sum()),
        "action_rms": float(np.sqrt(np.mean(applied ** 2))),
        "peak_action_abs": float(np.max(np.abs(applied))),
        "pre_projection_tv": pre_projection_tv,
        "post_projection_tv": post_projection_tv,
        "TV_t_per_s": float(post_projection_tv / max(dt.sum(), EPS)),
        "TV_L_per_100m": float(100.0 * post_projection_tv / distance.sum()) if distance.sum() > 0.0 else float("nan"),
        **reset_motion,
        "peak_slew_rate_per_s": float(rate.max(initial=0.0)),
        "saturation_ratio": float(rate_limited.mean()),
        "longest_saturation_samples": longest_sat_samples,
        "longest_saturation_s": float(longest_sat_s),
        "longest_saturation_m": float(longest_sat_m),
        "channel_cancellation_ratio_mean": float(cancellation.mean()),
        "channel_cancellation_ratio_p95": float(np.quantile(cancellation, 0.95)),
        "command_reversal_count": int(reversal_count),
        "command_reversals_per_100m": float(100.0 * reversal_count / max(distance.sum(), EPS)),
        "action_ood_ratio": float(flagged.mean()),
        "action_ood_max": float(score.max()),
        "action_ood_p95": float(np.quantile(score, 0.95)),
        "longest_ood_samples": longest_ood_samples,
        "longest_ood_s": float(longest_ood_s),
        "longest_ood_m": float(longest_ood_m),
        "pooled_action_ood_ratio": float(shared_flagged.mean()),
        "pooled_action_ood_max": float(shared_score.max()),
        "pooled_action_ood_p95": float(np.quantile(shared_score, 0.95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data1-logs", required=True)
    parser.add_argument("--data2-logs", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data1 = pd.read_csv(args.data1_logs, compression="gzip")
    data2 = pd.read_csv(args.data2_logs, compression="gzip")
    support = data1_support(data1)
    support.to_csv(out_dir / "data1_locked_command_support.csv", index=False, encoding="utf-8-sig")
    support_by_control = support.set_index("control")
    shared_support = support_by_control.loc["__pooled_locked_data1__"]

    rows = []
    for (control, pass_id), group in data2.groupby(["control", "pass_id"], sort=True):
        rows.append(audit_group(group, support_by_control.loc[control], shared_support))
    detail = pd.DataFrame(rows)
    summary = detail.groupby("control", as_index=False).agg(
        n_files=("pass_id", "nunique"), n_steps=("n_steps", "sum"), duration_s=("duration_s", "sum"), distance_m=("distance_m", "sum"),
        invalid_command_count=("invalid_command_count", "sum"), boundary_violation_count=("boundary_violation_count", "sum"),
        action_rms=("action_rms", "mean"), peak_action_abs=("peak_action_abs", "max"),
        pre_projection_tv=("pre_projection_tv", "mean"), post_projection_tv=("post_projection_tv", "mean"),
        startup_motion=("startup_motion", "mean"), restart_motion=("restart_motion", "mean"), startup_restart_motion=("startup_restart_motion", "mean"),
        TV_t_per_s=("TV_t_per_s", "mean"), TV_L_per_100m=("TV_L_per_100m", "mean"), peak_slew_rate_per_s=("peak_slew_rate_per_s", "max"),
        saturation_ratio=("saturation_ratio", "mean"), longest_saturation_samples=("longest_saturation_samples", "max"),
        longest_saturation_s=("longest_saturation_s", "max"), longest_saturation_m=("longest_saturation_m", "max"),
        channel_cancellation_ratio_mean=("channel_cancellation_ratio_mean", "mean"), channel_cancellation_ratio_p95=("channel_cancellation_ratio_p95", "max"),
        command_reversal_count=("command_reversal_count", "sum"), command_reversals_per_100m=("command_reversals_per_100m", "mean"),
        action_ood_ratio=("action_ood_ratio", "mean"), action_ood_max=("action_ood_max", "max"), action_ood_p95=("action_ood_p95", "max"),
        longest_ood_samples=("longest_ood_samples", "max"), longest_ood_s=("longest_ood_s", "max"), longest_ood_m=("longest_ood_m", "max"),
        pooled_action_ood_ratio=("pooled_action_ood_ratio", "mean"), pooled_action_ood_max=("pooled_action_ood_max", "max"), pooled_action_ood_p95=("pooled_action_ood_p95", "max"),
    )
    detail.to_csv(out_dir / "data2_command_audit_by_file.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "data2_command_audit_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "method.json").write_text(json.dumps({
        "name": "Surrogate-free command-effect plausibility screening",
        "scope": "This audit does not estimate plant-level closed-loop performance.",
        "data1_support": "Primary envelope pools all locked controllers on Data1; a controller-specific 0.5th to 99.5th percentile envelope is retained as a shift diagnostic.",
        "bound": BOUND,
        "ood_rule": "A sample is OOD when any applied command lies outside the stated Data1 envelope; excursion is normalized by that envelope width.",
        "limits": "The support is controller-generated replay support, not a measured plant actuator-label distribution.",
        "command_motion": "TV uses adjacent applied commands within contiguous supported segments; segment labels are not grouped across runs. Peak slew uses those transitions with finite positive control intervals and is zero when none exist. Startup and restart moves from zero are separate diagnostics. Logged du retains the original projected moves, including reset-start actions.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
