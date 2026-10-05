"""Data1 F_arch zero-innovation forward discrepancy diagnostic.

This script evaluates the released archived-response function (``plant_step``)
as a rolling-origin, open-loop forward discrepancy diagnostic on Data1.  It is
deliberately separate from the publication pipeline and does not modify any
locked configuration or publication result.

For every eligible origin, the state is reconstructed from recorded current
and lagged outputs plus the hidden coordinates obtained by replaying from the
start of the same supported segment.  The subsequent rollout fixes

    u = 0, ext = 0, proc = 0, innovation = 0

and evaluates horizons 1, 5, 10 and 20 without allowing a target to cross a
segment boundary.  The numerical output guardrail is evaluated both at the
released scale 8 and with an infinite scale.  This is a forward discrepancy
diagnostic; it is not physical or causal validation of a production plant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import batch_archive
import protocol
from segmented_archived_timebase import derive_segmented_archived_timebase


SCRIPT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_ROOT.parent
DEFAULT_OUT_DIR = PACKAGE_ROOT / "outputs" / "farch_forward_diagnostic"
DEFAULT_FROZEN_ENVELOPE = (
    PACKAGE_ROOT
    / "config"
    / "reference_run"
    / "03_nominal"
    / "audit"
    / "data1_stage_quantile_envelope_p025_p975.csv"
)
DATA1_BATCHES = {"I", "IV", "VII"}
EXPECTED_DATA1_PASSES = 13
DEFAULT_HORIZONS = (1, 5, 10, 20)
CHANNELS = (
    ("T", 0, "T_norm"),
    ("h", 1, "h_norm"),
    ("S", 2, "S_norm"),
)
GUARDRAILS = (("scale_8", 8.0), ("infinite", math.inf))
ZERO_U = np.zeros(3, dtype=float)
ZERO_EXT = np.zeros(5, dtype=float)
ZERO_PROC = np.zeros(6, dtype=float)
MACHINE_EPS = float(np.finfo(np.float64).eps)


def portable_path(path: Path) -> str:
    """Return a package-relative provenance path without workstation details."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PACKAGE_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_horizons(text: str) -> tuple[int, ...]:
    try:
        horizons = tuple(sorted({int(token.strip()) for token in text.split(",") if token.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Horizons must be comma-separated positive integers") from exc
    if not horizons or any(value <= 0 for value in horizons):
        raise argparse.ArgumentTypeError("Horizons must be positive integers")
    if 1 not in horizons:
        raise argparse.ArgumentTypeError("Horizon 1 is required for the innovations hard check")
    return horizons


def json_ready(value: Any) -> Any:
    """Convert NumPy/path objects and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def zero_step_scale8(suite: Any, x: np.ndarray, p: Any, k: int, steps: int) -> np.ndarray:
    """Call the exact signature used by ``build_replay_innovations``."""
    return suite.plant_step(
        x,
        ZERO_U,
        p,
        k,
        steps,
        ZERO_EXT,
        ZERO_PROC,
        allocation=True,
    )


def zero_step_unclipped(suite: Any, x: np.ndarray, p: Any, k: int, steps: int) -> np.ndarray:
    return suite.plant_step(
        x,
        ZERO_U,
        p,
        k,
        steps,
        ZERO_EXT,
        ZERO_PROC,
        allocation=True,
        output_clip_scale=math.inf,
    )


def reconstruct_recorded_states(
    suite: Any,
    rp: Any,
    segment_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild the exact teacher-forced state used by innovation extraction."""
    y = np.asarray(rp.y_real, dtype=float)
    n = len(y)
    states = np.full((n, 9), np.nan, dtype=float)
    stored_values: list[np.ndarray] = []
    rebuilt_values: list[np.ndarray] = []
    x = np.r_[y[0], y[0], np.zeros(3, dtype=float)]

    for k in range(n):
        segment_start = k == 0 or segment_ids[k] != segment_ids[k - 1]
        if segment_start:
            x = np.r_[y[k], y[k], np.zeros(3, dtype=float)]
        states[k] = x
        if k >= n - 1 or segment_ids[k + 1] != segment_ids[k]:
            continue
        prediction = zero_step_scale8(suite, x, rp.p, k, n)
        rebuilt = y[k + 1] - prediction[:3]
        stored = np.asarray(rp.innovations[k], dtype=float)
        rebuilt_values.append(rebuilt.copy())
        stored_values.append(stored.copy())
        x = prediction.copy()
        x[:3] = y[k + 1]
        x[3:6] = y[k]

    if stored_values:
        stored_array = np.vstack(stored_values)
        rebuilt_array = np.vstack(rebuilt_values)
    else:
        stored_array = np.empty((0, 3), dtype=float)
        rebuilt_array = np.empty((0, 3), dtype=float)
    scale = max(
        1.0,
        float(np.max(np.abs(stored_array))) if stored_array.size else 0.0,
        float(np.max(np.abs(rebuilt_array))) if rebuilt_array.size else 0.0,
    )
    tolerance = 64.0 * MACHINE_EPS * scale
    difference = rebuilt_array - stored_array
    exact = bool(np.array_equal(rebuilt_array, stored_array))
    within_machine_precision = bool(
        np.all(np.abs(difference) <= tolerance)
    )
    sign_mismatch = int(
        np.count_nonzero(
            np.signbit(rebuilt_array) != np.signbit(stored_array)
        )
    )
    check = {
        "eligible_transitions": int(len(stored_array)),
        "eligible_scalar_values": int(stored_array.size),
        "bitwise_equal": exact,
        "within_machine_precision": within_machine_precision,
        "max_abs_difference": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "machine_precision_tolerance": float(tolerance),
        "sign_mismatch_count": sign_mismatch,
    }
    if not within_machine_precision or sign_mismatch:
        raise AssertionError(
            f"{rp.pass_id}: reconstructed one-step discrepancy does not match stored "
            f"innovations (max_abs={check['max_abs_difference']:.17g}, "
            f"tol={tolerance:.17g}, sign_mismatches={sign_mismatch})"
        )
    if not np.isfinite(states).all():
        raise AssertionError(f"{rp.pass_id}: non-finite reconstructed state")
    return states, check


def phase_labels(suite: Any, rp: Any) -> np.ndarray:
    n = len(rp.y_real)
    return np.asarray(
        [suite.phase_name(min(k, n - 2), n, rp.p) for k in range(n)],
        dtype=object,
    )


def recorded_normalized_rows(suite: Any, passes: list[Any]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for rp in passes:
        scales = protocol.scale_vec(rp.p)
        normalized = np.asarray(rp.y_real, dtype=float) / scales.reshape(1, 3)
        rows.append(
            pd.DataFrame(
                {
                    "pass_id": rp.pass_id,
                    "phase": phase_labels(suite, rp),
                    "T_norm": normalized[:, 0],
                    "h_norm": normalized[:, 1],
                    "S_norm": normalized[:, 2],
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    if not np.isfinite(result[[item[2] for item in CHANNELS]].to_numpy(float)).all():
        raise AssertionError("Non-finite normalized recorded Data1 value")
    return result


def quantile_envelope_wide(recorded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phase, group in recorded.groupby("phase", sort=True):
        row: dict[str, Any] = {"phase": str(phase)}
        for _, _, norm_col in CHANNELS:
            row[f"{norm_col}_lo"] = float(group[norm_col].quantile(0.025))
            row[f"{norm_col}_hi"] = float(group[norm_col].quantile(0.975))
            row[f"{norm_col}_median"] = float(group[norm_col].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("phase", kind="mergesort").reset_index(drop=True)


def envelope_wide_to_long(
    envelope: pd.DataFrame,
    scope: str,
    held_out_pass_id: str,
    recorded_training: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pass_count = int(recorded_training["pass_id"].nunique())
    for _, entry in envelope.iterrows():
        phase = str(entry["phase"])
        phase_count = int(recorded_training["phase"].eq(phase).sum())
        for channel, _, norm_col in CHANNELS:
            rows.append(
                {
                    "envelope_scope": scope,
                    "held_out_pass_id": held_out_pass_id,
                    "phase": phase,
                    "channel": channel,
                    "lower_norm": float(entry[f"{norm_col}_lo"]),
                    "upper_norm": float(entry[f"{norm_col}_hi"]),
                    "median_norm": float(entry[f"{norm_col}_median"]),
                    "training_rows_in_phase": phase_count,
                    "training_pass_count": pass_count,
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def build_recorded_envelopes(
    suite: Any,
    passes: list[Any],
    frozen_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    recorded = recorded_normalized_rows(suite, passes)
    recomputed_all = quantile_envelope_wide(recorded)
    frozen = pd.read_csv(frozen_path).sort_values("phase", kind="mergesort").reset_index(drop=True)
    required = ["phase"] + [
        f"{norm_col}_{suffix}"
        for _, _, norm_col in CHANNELS
        for suffix in ("lo", "hi", "median")
    ]
    missing = sorted(set(required) - set(frozen.columns))
    if missing:
        raise ValueError(f"Frozen envelope is missing columns: {missing}")
    frozen = frozen[required]
    if list(frozen["phase"].astype(str)) != list(recomputed_all["phase"].astype(str)):
        raise AssertionError("Frozen and recomputed envelope phases differ")
    numeric_columns = required[1:]
    envelope_difference = (
        frozen[numeric_columns].to_numpy(float)
        - recomputed_all[numeric_columns].to_numpy(float)
    )
    # Quantile medians can differ by a few 1e-11 across supported pandas
    # versions because an even-sized median averages its two central values.
    # The frozen file remains authoritative; this check detects substantive
    # data/normalizer drift while recording the observed numerical difference.
    envelope_tolerance = max(
        1e-10,
        128.0
        * MACHINE_EPS
        * max(
            1.0,
            float(np.max(np.abs(frozen[numeric_columns].to_numpy(float)))),
        ),
    )
    envelope_match = bool(np.all(np.abs(envelope_difference) <= envelope_tolerance))
    if not envelope_match:
        raise AssertionError(
            "Recomputed all-Data1 envelope does not reproduce the frozen envelope: "
            f"max_abs={np.max(np.abs(envelope_difference)):.17g}, "
            f"tol={envelope_tolerance:.17g}"
        )

    envelope_frames = [
        envelope_wide_to_long(
            frozen,
            "frozen_all_data1",
            "",
            recorded,
            portable_path(frozen_path),
        )
    ]
    for held_out in sorted(recorded["pass_id"].unique()):
        training = recorded.loc[~recorded["pass_id"].eq(held_out)].copy()
        lopo = quantile_envelope_wide(training)
        envelope_frames.append(
            envelope_wide_to_long(
                lopo,
                "leave_one_pass_out",
                str(held_out),
                training,
                "recomputed from the other 12 Data1 passes",
            )
        )
    envelopes = pd.concat(envelope_frames, ignore_index=True)
    check = {
        "frozen_path": portable_path(frozen_path),
        "frozen_sha256": sha256_file(frozen_path),
        "recomputed_matches_frozen": envelope_match,
        "max_abs_difference": float(np.max(np.abs(envelope_difference))),
        "comparison_tolerance": float(envelope_tolerance),
        "recorded_rows": int(len(recorded)),
        "recorded_passes": int(recorded["pass_id"].nunique()),
        "phases": sorted(recorded["phase"].astype(str).unique().tolist()),
        "lopo_folds": int(recorded["pass_id"].nunique()),
    }
    return recorded, envelopes, check


def selected_origins(segment_ids: np.ndarray, maximum_per_segment: int) -> set[int] | None:
    if maximum_per_segment <= 0:
        return None
    selected: set[int] = set()
    for segment in np.unique(segment_ids):
        indices = np.flatnonzero(segment_ids == segment)
        candidates = indices[:-1]
        if len(candidates) <= maximum_per_segment:
            selected.update(int(value) for value in candidates)
            continue
        positions = np.linspace(0, len(candidates) - 1, maximum_per_segment)
        chosen = np.unique(np.rint(positions).astype(int))
        selected.update(int(candidates[position]) for position in chosen)
    return selected


def rollout_pass(
    suite: Any,
    rp: Any,
    inventory_row: pd.Series,
    horizons: tuple[int, ...],
    maximum_origins_per_segment: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y = np.asarray(rp.y_real, dtype=float)
    n = len(y)
    timebase = derive_segmented_archived_timebase(
        rp.df,
        protocol.DISTANCE_STEP_REFERENCE_M,
    )
    segment_ids = np.asarray(timebase.segment_id, dtype=int)
    states, reconstruction_check = reconstruct_recorded_states(suite, rp, segment_ids)
    phases = phase_labels(suite, rp)
    scales = protocol.scale_vec(rp.p)
    guardrail_limit = 8.0 * scales
    origin_filter = selected_origins(segment_ids, maximum_origins_per_segment)
    rows: list[dict[str, Any]] = []

    for origin_k in range(n - 1):
        if origin_filter is not None and origin_k not in origin_filter:
            continue
        valid_horizons = tuple(
            horizon
            for horizon in horizons
            if origin_k + horizon < n
            and segment_ids[origin_k + horizon] == segment_ids[origin_k]
        )
        if not valid_horizons:
            continue
        maximum_horizon = max(valid_horizons)
        for guardrail_label, guardrail_scale in GUARDRAILS:
            x = states[origin_k].copy()
            active_channels = np.zeros(3, dtype=bool)
            active_step_counts = np.zeros(3, dtype=int)
            crossing_channels = np.zeros(3, dtype=bool)
            crossing_step_counts = np.zeros(3, dtype=int)
            for step in range(1, maximum_horizon + 1):
                global_k = origin_k + step - 1
                if math.isfinite(guardrail_scale):
                    # The first call preserves the exact arithmetic/signature of
                    # the released innovations path.  The unclipped companion
                    # call only diagnoses whether that call clipped an output.
                    x_next = zero_step_scale8(suite, x, rp.p, global_k, n)
                    unguarded_from_same_state = zero_step_unclipped(
                        suite, x, rp.p, global_k, n
                    )
                    step_active = np.abs(unguarded_from_same_state[:3]) > guardrail_limit
                    active_channels |= step_active
                    active_step_counts += step_active.astype(int)
                    crossing_channels |= step_active
                    crossing_step_counts += step_active.astype(int)
                else:
                    x_next = zero_step_unclipped(suite, x, rp.p, global_k, n)
                    step_crossing = np.abs(x_next[:3]) > guardrail_limit
                    crossing_channels |= step_crossing
                    crossing_step_counts += step_crossing.astype(int)
                x = x_next
                if step not in valid_horizons:
                    continue
                target_k = origin_k + step
                prediction = x[:3].copy()
                target = y[target_k]
                discrepancy = target - prediction
                occupancy = np.abs(prediction) / guardrail_limit
                target_occupancy = np.abs(target) / guardrail_limit
                row: dict[str, Any] = {
                    "pass_id": rp.pass_id,
                    "batch": str(inventory_row["batch"]),
                    "condition_id": rp.condition_id,
                    "trial_id": rp.trial_id,
                    "file": rp.file,
                    "source_sha256": str(inventory_row["sha256"]),
                    "segment_id": int(segment_ids[origin_k]),
                    "origin_k": int(origin_k),
                    "target_k": int(target_k),
                    "horizon_updates": int(step),
                    "target_phase": str(phases[target_k]),
                    "guardrail_label": guardrail_label,
                    "guardrail_scale": float(guardrail_scale),
                    "guardrail_applied": bool(math.isfinite(guardrail_scale)),
                    "rollout_guardrail_any_active": bool(np.any(active_channels)),
                    "rollout_guardrail_active_channel_count": int(
                        np.count_nonzero(active_step_counts > 0)
                    ),
                    "eight_scale_boundary_any_crossed": bool(np.any(crossing_channels)),
                    "target_same_segment": bool(
                        segment_ids[target_k] == segment_ids[origin_k]
                    ),
                }
                for channel, index, _ in CHANNELS:
                    row[f"{channel}_scale"] = float(scales[index])
                    row[f"{channel}_prediction"] = float(prediction[index])
                    row[f"{channel}_target"] = float(target[index])
                    row[f"{channel}_discrepancy"] = float(discrepancy[index])
                    row[f"{channel}_discrepancy_norm"] = float(
                        discrepancy[index] / scales[index]
                    )
                    row[f"{channel}_guardrail_occupancy"] = float(occupancy[index])
                    row[f"{channel}_target_guardrail_occupancy"] = float(
                        target_occupancy[index]
                    )
                    row[f"{channel}_rollout_guardrail_active"] = bool(
                        active_channels[index]
                    )
                    row[f"{channel}_rollout_guardrail_step_count"] = int(
                        active_step_counts[index]
                    )
                    row[f"{channel}_eight_scale_boundary_crossed"] = bool(
                        crossing_channels[index]
                    )
                    row[f"{channel}_eight_scale_boundary_step_count"] = int(
                        crossing_step_counts[index]
                    )
                rows.append(row)

    return rows, {
        "pass_id": rp.pass_id,
        "rows": n,
        "segments": int(segment_ids.max() + 1) if len(segment_ids) else 0,
        "evaluated_origins": int(len({row["origin_k"] for row in rows})),
        "prediction_rows": int(len(rows)),
        "timebase_source_counts": {
            str(label): int(np.count_nonzero(timebase.source == label))
            for label in np.unique(timebase.source)
        },
        "state_reconstruction_h1_check": reconstruction_check,
    }


def predictions_to_long(predictions: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "pass_id",
        "batch",
        "condition_id",
        "segment_id",
        "origin_k",
        "target_k",
        "horizon_updates",
        "target_phase",
        "guardrail_label",
        "guardrail_scale",
        "guardrail_applied",
        "rollout_guardrail_any_active",
        "eight_scale_boundary_any_crossed",
    ]
    frames: list[pd.DataFrame] = []
    for channel, _, _ in CHANNELS:
        frame = predictions[base_columns].copy()
        frame["channel"] = channel
        frame["scale"] = predictions[f"{channel}_scale"].to_numpy(float)
        frame["prediction"] = predictions[f"{channel}_prediction"].to_numpy(float)
        frame["target"] = predictions[f"{channel}_target"].to_numpy(float)
        frame["discrepancy"] = predictions[f"{channel}_discrepancy"].to_numpy(float)
        frame["discrepancy_norm"] = predictions[
            f"{channel}_discrepancy_norm"
        ].to_numpy(float)
        frame["prediction_norm"] = frame["prediction"] / frame["scale"]
        frame["target_norm"] = frame["target"] / frame["scale"]
        frame["guardrail_occupancy"] = predictions[
            f"{channel}_guardrail_occupancy"
        ].to_numpy(float)
        frame["target_guardrail_occupancy"] = predictions[
            f"{channel}_target_guardrail_occupancy"
        ].to_numpy(float)
        frame["rollout_guardrail_active"] = predictions[
            f"{channel}_rollout_guardrail_active"
        ].to_numpy(bool)
        frame["rollout_guardrail_step_count"] = predictions[
            f"{channel}_rollout_guardrail_step_count"
        ].to_numpy(int)
        frame["eight_scale_boundary_crossed"] = predictions[
            f"{channel}_eight_scale_boundary_crossed"
        ].to_numpy(bool)
        frame["eight_scale_boundary_step_count"] = predictions[
            f"{channel}_eight_scale_boundary_step_count"
        ].to_numpy(int)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    numeric = result[
        [
            "prediction",
            "target",
            "discrepancy",
            "discrepancy_norm",
            "guardrail_occupancy",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise AssertionError("Non-finite forward prediction or discrepancy")
    return result


def correlation_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def metric_row(group: pd.DataFrame) -> dict[str, Any]:
    discrepancy = group["discrepancy"].to_numpy(float)
    discrepancy_norm = group["discrepancy_norm"].to_numpy(float)
    abs_norm = np.abs(discrepancy_norm)
    prediction_norm = group["prediction_norm"].to_numpy(float)
    target_norm = group["target_norm"].to_numpy(float)
    target_centered = target_norm - float(np.mean(target_norm))
    denominator = float(np.sum(np.square(target_centered)))
    r2_norm = (
        float(1.0 - np.sum(np.square(discrepancy_norm)) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    guardrail_applied = bool(group["guardrail_applied"].iloc[0])
    applied_steps = int(group["horizon_updates"].sum())
    activation_rate = (
        float(group["rollout_guardrail_active"].mean())
        if guardrail_applied
        else float("nan")
    )
    activation_step_rate = (
        float(group["rollout_guardrail_step_count"].sum() / max(applied_steps, 1))
        if guardrail_applied
        else float("nan")
    )
    return {
        "n_origins": int(len(group)),
        "n_passes": int(group["pass_id"].nunique()),
        "n_segments": int(group[["pass_id", "segment_id"]].drop_duplicates().shape[0]),
        "discrepancy_mean": float(np.mean(discrepancy)),
        "mae": float(np.mean(np.abs(discrepancy))),
        "rmse": float(np.sqrt(np.mean(np.square(discrepancy)))),
        "discrepancy_norm_mean": float(np.mean(discrepancy_norm)),
        "mae_norm": float(np.mean(abs_norm)),
        "rmse_norm": float(np.sqrt(np.mean(np.square(discrepancy_norm)))),
        "median_abs_discrepancy_norm": float(np.median(abs_norm)),
        "p95_abs_discrepancy_norm": float(np.quantile(abs_norm, 0.95)),
        "max_abs_discrepancy_norm": float(np.max(abs_norm)),
        "correlation_norm": correlation_or_nan(prediction_norm, target_norm),
        "r2_norm": r2_norm,
        "guardrail_trajectory_activation_rate": activation_rate,
        "guardrail_any_channel_trajectory_activation_rate": (
            float(group["rollout_guardrail_any_active"].mean())
            if guardrail_applied
            else float("nan")
        ),
        "guardrail_step_channel_activation_rate": activation_step_rate,
        "eight_scale_boundary_crossing_rate": float(
            group["eight_scale_boundary_crossed"].mean()
        ),
        "prediction_guardrail_occupancy_p95": float(
            group["guardrail_occupancy"].quantile(0.95)
        ),
        "prediction_guardrail_occupancy_max": float(
            group["guardrail_occupancy"].max()
        ),
        "prediction_beyond_8scale_rate": float(
            (group["guardrail_occupancy"] > 1.0 + 1e-12).mean()
        ),
        "recorded_target_guardrail_occupancy_p95": float(
            group["target_guardrail_occupancy"].quantile(0.95)
        ),
        "recorded_target_guardrail_occupancy_max": float(
            group["target_guardrail_occupancy"].max()
        ),
        "recorded_target_beyond_8scale_rate": float(
            (group["target_guardrail_occupancy"] > 1.0 + 1e-12).mean()
        ),
    }


def grouped_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in frame.groupby(grouper, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_columns, keys)}
        row.update(metric_row(group))
        if "pass_id" in group_columns:
            row["condition_id"] = str(group["condition_id"].iloc[0])
            row["batch"] = str(group["batch"].iloc[0])
        if "segment_id" in group_columns:
            row["first_origin_k"] = int(group["origin_k"].min())
            row["last_origin_k"] = int(group["origin_k"].max())
            row["first_target_k"] = int(group["target_k"].min())
            row["last_target_k"] = int(group["target_k"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def add_equal_pass_summary(
    pooled: pd.DataFrame,
    by_pass: pd.DataFrame,
) -> pd.DataFrame:
    pooled = pooled.copy()
    pooled.insert(0, "aggregation", "pooled_origins")
    key_columns = ["guardrail_label", "guardrail_scale", "horizon_updates", "channel"]
    excluded = set(key_columns) | {"pass_id", "condition_id", "batch"}
    metric_columns = [
        column
        for column in by_pass.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(by_pass[column])
    ]
    equal_rows: list[dict[str, Any]] = []
    for keys, group in by_pass.groupby(key_columns, sort=True):
        row = {column: key for column, key in zip(key_columns, keys)}
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.mean()) if values.notna().any() else float("nan")
        row["n_origins"] = int(group["n_origins"].sum())
        row["n_passes"] = int(group["pass_id"].nunique())
        row["n_segments"] = int(group["n_segments"].sum())
        row["aggregation"] = "equal_pass_mean"
        equal_rows.append(row)
    equal = pd.DataFrame(equal_rows)
    ordered = list(pooled.columns)
    for column in ordered:
        if column not in equal:
            equal[column] = np.nan
    return pd.concat([pooled, equal[ordered]], ignore_index=True)


def occupancy_metric_row(group: pd.DataFrame) -> dict[str, Any]:
    prediction_excess = group["prediction_envelope_excess"].to_numpy(float)
    target_excess = group["target_envelope_excess"].to_numpy(float)
    return {
        "n_origins": int(len(group)),
        "n_passes": int(group["pass_id"].nunique()),
        "n_segments": int(group[["pass_id", "segment_id"]].drop_duplicates().shape[0]),
        "prediction_inside_rate": float(group["prediction_inside"].mean()),
        "recorded_target_inside_rate": float(group["target_inside"].mean()),
        "prediction_outside_excess_mean": float(np.mean(prediction_excess)),
        "prediction_outside_excess_p95": float(np.quantile(prediction_excess, 0.95)),
        "prediction_outside_excess_max": float(np.max(prediction_excess)),
        "recorded_target_outside_excess_mean": float(np.mean(target_excess)),
        "recorded_target_outside_excess_p95": float(np.quantile(target_excess, 0.95)),
        "recorded_target_outside_excess_max": float(np.max(target_excess)),
        "inside_status_agreement_rate": float(
            (group["prediction_inside"] == group["target_inside"]).mean()
        ),
    }


def occupancy_grouped(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_columns, keys)}
        row.update(occupancy_metric_row(group))
        if "pass_id" in group_columns:
            row["condition_id"] = str(group["condition_id"].iloc[0])
            row["batch"] = str(group["batch"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_envelope_occupancy(
    long_predictions: pd.DataFrame,
    envelopes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames: list[pd.DataFrame] = []
    pass_frames: list[pd.DataFrame] = []
    base_keys = [
        "guardrail_label",
        "guardrail_scale",
        "horizon_updates",
        "channel",
    ]
    for scope in ("frozen_all_data1", "leave_one_pass_out"):
        envelope = envelopes.loc[envelopes["envelope_scope"].eq(scope)].copy()
        if scope == "frozen_all_data1":
            merge_keys_left = ["target_phase", "channel"]
            merge_keys_right = ["phase", "channel"]
        else:
            merge_keys_left = ["pass_id", "target_phase", "channel"]
            merge_keys_right = ["held_out_pass_id", "phase", "channel"]
        evaluation = long_predictions.merge(
            envelope,
            left_on=merge_keys_left,
            right_on=merge_keys_right,
            how="left",
            validate="many_to_one",
        )
        if evaluation[["lower_norm", "upper_norm"]].isna().any().any():
            raise AssertionError(f"Missing envelope bound for scope {scope}")
        evaluation["prediction_inside"] = (
            (evaluation["prediction_norm"] >= evaluation["lower_norm"])
            & (evaluation["prediction_norm"] <= evaluation["upper_norm"])
        )
        evaluation["target_inside"] = (
            (evaluation["target_norm"] >= evaluation["lower_norm"])
            & (evaluation["target_norm"] <= evaluation["upper_norm"])
        )
        evaluation["prediction_envelope_excess"] = np.maximum.reduce(
            [
                evaluation["lower_norm"].to_numpy(float)
                - evaluation["prediction_norm"].to_numpy(float),
                evaluation["prediction_norm"].to_numpy(float)
                - evaluation["upper_norm"].to_numpy(float),
                np.zeros(len(evaluation), dtype=float),
            ]
        )
        evaluation["target_envelope_excess"] = np.maximum.reduce(
            [
                evaluation["lower_norm"].to_numpy(float)
                - evaluation["target_norm"].to_numpy(float),
                evaluation["target_norm"].to_numpy(float)
                - evaluation["upper_norm"].to_numpy(float),
                np.zeros(len(evaluation), dtype=float),
            ]
        )
        summary = occupancy_grouped(evaluation, base_keys)
        summary.insert(0, "envelope_scope", scope)
        by_pass = occupancy_grouped(evaluation, ["pass_id", *base_keys])
        by_pass.insert(0, "envelope_scope", scope)
        summary_frames.append(summary)
        pass_frames.append(by_pass)
    return (
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(pass_frames, ignore_index=True),
    )


def add_equal_pass_occupancy(
    pooled: pd.DataFrame,
    by_pass: pd.DataFrame,
) -> pd.DataFrame:
    pooled = pooled.copy()
    pooled.insert(0, "aggregation", "pooled_origins")
    key_columns = [
        "envelope_scope",
        "guardrail_label",
        "guardrail_scale",
        "horizon_updates",
        "channel",
    ]
    excluded = set(key_columns) | {"pass_id", "condition_id", "batch"}
    metric_columns = [
        column
        for column in by_pass.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(by_pass[column])
    ]
    equal_rows: list[dict[str, Any]] = []
    for keys, group in by_pass.groupby(key_columns, sort=True):
        row = {column: key for column, key in zip(key_columns, keys)}
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.mean()) if values.notna().any() else float("nan")
        row["n_origins"] = int(group["n_origins"].sum())
        row["n_passes"] = int(group["pass_id"].nunique())
        row["n_segments"] = int(group["n_segments"].sum())
        row["aggregation"] = "equal_pass_mean"
        equal_rows.append(row)
    equal = pd.DataFrame(equal_rows)
    ordered = list(pooled.columns)
    for column in ordered:
        if column not in equal:
            equal[column] = np.nan
    return pd.concat([pooled, equal[ordered]], ignore_index=True)


def hard_check_h1_forward_rows(
    predictions: pd.DataFrame,
    pass_map: dict[str, Any],
) -> dict[str, Any]:
    h1 = predictions.loc[
        predictions["guardrail_label"].eq("scale_8")
        & predictions["horizon_updates"].eq(1)
    ].sort_values(["pass_id", "origin_k"], kind="mergesort")
    expected: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    for row in h1.itertuples(index=False):
        rp = pass_map[str(row.pass_id)]
        expected.append(np.asarray(rp.innovations[int(row.origin_k)], dtype=float))
        observed.append(
            np.array(
                [row.T_discrepancy, row.h_discrepancy, row.S_discrepancy],
                dtype=float,
            )
        )
    expected_array = np.vstack(expected) if expected else np.empty((0, 3), dtype=float)
    observed_array = np.vstack(observed) if observed else np.empty((0, 3), dtype=float)
    difference = observed_array - expected_array
    magnitude = max(
        1.0,
        float(np.max(np.abs(expected_array))) if expected_array.size else 0.0,
        float(np.max(np.abs(observed_array))) if observed_array.size else 0.0,
    )
    tolerance = 64.0 * MACHINE_EPS * magnitude
    sign_mismatch = int(
        np.count_nonzero(np.signbit(observed_array) != np.signbit(expected_array))
    )
    check = {
        "definition": "recorded_target_minus_F_arch_prediction",
        "eligible_transitions_checked": int(len(h1)),
        "scalar_values_checked": int(expected_array.size),
        "bitwise_equal": bool(np.array_equal(observed_array, expected_array)),
        "within_machine_precision": bool(np.all(np.abs(difference) <= tolerance)),
        "max_abs_difference": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "machine_precision_tolerance": float(tolerance),
        "sign_mismatch_count": sign_mismatch,
    }
    if not check["within_machine_precision"] or sign_mismatch:
        raise AssertionError(
            "Rolling-origin h=1 discrepancy failed the stored innovations hard check: "
            f"max_abs={check['max_abs_difference']:.17g}, tol={tolerance:.17g}, "
            f"sign_mismatches={sign_mismatch}"
        )
    return check


def write_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        compression=compression,
    )


def output_record(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }


def load_data1(batch_root: Path) -> tuple[Any, Any, list[Any], pd.DataFrame]:
    suite, replay = protocol.load_suite_modules()
    batch_archive.BATCH_ROOT = batch_root.resolve()
    passes, inventory = batch_archive.load_batch_passes(
        protocol,
        suite,
        replay,
        included_batches=DATA1_BATCHES,
    )
    passes = sorted(passes, key=lambda item: item.pass_id)
    inventory = inventory.sort_values("pass_id", kind="mergesort").reset_index(drop=True)
    if len(passes) != EXPECTED_DATA1_PASSES:
        raise AssertionError(f"Expected 13 Data1 passes, found {len(passes)}")
    if {rp.pass_id for rp in passes} != set(batch_archive.DATA1_IDS):
        raise AssertionError("Loaded pass identities do not match the frozen Data1 set")
    return suite, replay, passes, inventory


def file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {portable_path(path): sha256_file(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data1 F_arch zero-innovation rolling-origin forward discrepancy diagnostic"
    )
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--frozen-envelope",
        type=Path,
        default=DEFAULT_FROZEN_ENVELOPE,
    )
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=DEFAULT_HORIZONS,
        help="Comma-separated update horizons; h=1 is mandatory (default: 1,5,10,20)",
    )
    parser.add_argument(
        "--pass-limit",
        type=int,
        default=0,
        help="Evaluate only the first N Data1 passes (smoke tests only; 0 means all 13)",
    )
    parser.add_argument(
        "--max-origins-per-segment",
        type=int,
        default=0,
        help="Deterministic evenly spaced origin cap per segment (smoke tests only; 0 means all)",
    )
    parser.add_argument(
        "--write-row-detail",
        action="store_true",
        help="Also write the production-derived rolling-origin prediction rows as gzip CSV",
    )
    args = parser.parse_args()
    if args.pass_limit < 0 or args.max_origins_per_segment < 0:
        parser.error("--pass-limit and --max-origins-per-segment must be non-negative")

    started = time.perf_counter()
    out_dir = args.out_dir.resolve()
    frozen_path = args.frozen_envelope.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[load] resolving the hash-locked Data1 archive", flush=True)
    suite, replay, all_passes, inventory = load_data1(args.batch_root)
    _, envelopes, envelope_check = build_recorded_envelopes(
        suite,
        all_passes,
        frozen_path,
    )
    evaluation_passes = all_passes[: args.pass_limit or None]
    inventory_by_pass = inventory.set_index("pass_id")
    print(
        f"[setup] evaluation_passes={len(evaluation_passes)} horizons={args.horizons} "
        f"origin_cap={args.max_origins_per_segment or 'all'}",
        flush=True,
    )

    prediction_rows: list[dict[str, Any]] = []
    pass_audits: list[dict[str, Any]] = []
    for index, rp in enumerate(evaluation_passes, start=1):
        rows, pass_audit = rollout_pass(
            suite,
            rp,
            inventory_by_pass.loc[rp.pass_id],
            tuple(args.horizons),
            args.max_origins_per_segment,
        )
        prediction_rows.extend(rows)
        pass_audits.append(pass_audit)
        print(
            f"[forward] {index:02d}/{len(evaluation_passes):02d} {rp.pass_id}: "
            f"rows={len(rows):,}",
            flush=True,
        )
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise AssertionError("No eligible rolling-origin predictions were generated")
    if not predictions["target_same_segment"].all():
        raise AssertionError("At least one forward target crosses a segment boundary")
    expected_pairs = {
        (label, horizon)
        for label, _ in GUARDRAILS
        for horizon in args.horizons
    }
    observed_pairs = set(
        zip(predictions["guardrail_label"], predictions["horizon_updates"])
    )
    if observed_pairs != expected_pairs:
        raise AssertionError(
            f"Missing guardrail/horizon result: expected={expected_pairs}, observed={observed_pairs}"
        )

    pass_map = {rp.pass_id: rp for rp in evaluation_passes}
    h1_check = hard_check_h1_forward_rows(predictions, pass_map)
    reconstructed_eligible_transitions = int(
        sum(
            item["state_reconstruction_h1_check"]["eligible_transitions"]
            for item in pass_audits
        )
    )
    complete_h1_coverage = (
        h1_check["eligible_transitions_checked"]
        == reconstructed_eligible_transitions
    )
    h1_check["eligible_transitions_from_state_reconstruction"] = (
        reconstructed_eligible_transitions
    )
    h1_check["complete_coverage"] = complete_h1_coverage
    h1_check["complete_coverage_required"] = (
        args.max_origins_per_segment == 0
    )
    if args.max_origins_per_segment == 0 and not complete_h1_coverage:
        raise AssertionError(
            "Rolling-origin h=1 innovations check did not cover every legal "
            f"transition: checked={h1_check['eligible_transitions_checked']}, "
            f"expected={reconstructed_eligible_transitions}"
        )
    long_predictions = predictions_to_long(predictions)
    metric_keys = ["guardrail_label", "guardrail_scale", "horizon_updates", "channel"]
    by_pass = grouped_metrics(long_predictions, ["pass_id", *metric_keys])
    by_segment = grouped_metrics(
        long_predictions,
        ["pass_id", "segment_id", *metric_keys],
    )
    pooled = grouped_metrics(long_predictions, metric_keys)
    summary = add_equal_pass_summary(pooled, by_pass)
    occupancy, occupancy_by_pass = evaluate_envelope_occupancy(
        long_predictions,
        envelopes,
    )
    occupancy = add_equal_pass_occupancy(occupancy, occupancy_by_pass)

    output_frames: list[tuple[Path, pd.DataFrame]] = [
        (out_dir / "farch_forward_summary.csv", summary),
        (out_dir / "farch_forward_by_pass.csv", by_pass),
        (out_dir / "farch_forward_by_pass_segment.csv", by_segment),
        (out_dir / "farch_recorded_envelopes.csv", envelopes),
        (out_dir / "farch_recorded_envelope_occupancy.csv", occupancy),
        (
            out_dir / "farch_recorded_envelope_occupancy_by_pass.csv",
            occupancy_by_pass,
        ),
    ]
    output_manifest: dict[str, Any] = {}
    for path, frame in output_frames:
        write_csv(frame, path)
        output_manifest[path.name] = output_record(path, frame)
    if args.write_row_detail:
        row_path = out_dir / "farch_forward_row_detail.csv.gz"
        write_csv(predictions, row_path, compression="gzip")
        output_manifest[row_path.name] = output_record(row_path, predictions)

    dependencies = [
        Path(__file__).resolve(),
        SCRIPT_ROOT / "protocol.py",
        SCRIPT_ROOT / "batch_archive.py",
        SCRIPT_ROOT / "segmented_archived_timebase.py",
        SCRIPT_ROOT / "engine_core" / "15_realdata_closed_loop_suite.py",
        SCRIPT_ROOT / "engine_core" / "20_independent_replay_validation.py",
        PACKAGE_ROOT / "config" / "data1_scale_lock.json",
        PACKAGE_ROOT / "data_map" / "production_batch_map.csv",
        frozen_path,
    ]
    runtime_before_audit = float(time.perf_counter() - started)
    audit = {
        "schema_version": 1,
        "diagnostic_id": "data1_farch_zero_innovation_forward_discrepancy_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation_boundary": (
            "Rolling-origin forward discrepancy diagnostic only; not physical or causal "
            "validation of a production plant."
        ),
        "state_origin": (
            "recorded current/lag outputs at every eligible origin plus hidden state "
            "reconstructed from the start of the same supported segment"
        ),
        "rollout": {
            "u": [0.0, 0.0, 0.0],
            "ext": [0.0] * 5,
            "proc": [0.0] * 6,
            "innovation": [0.0, 0.0, 0.0],
            "allocation": True,
            "model_variant": "plant_step default hybrid",
            "horizons_updates": list(args.horizons),
            "guardrail_scales": [8.0, "infinite"],
            "target_boundary_rule": "origin and target must share segment_id",
            "primary_design": "rolling_origin",
        },
        "timebase": {
            "implementation": "segmented_archived_timebase.derive_segmented_archived_timebase",
            "distance_step_reference_m": protocol.DISTANCE_STEP_REFERENCE_M,
            "targets_cross_segment": False,
        },
        "inputs": {
            "batch_root": "<BATCH_ROOT>",
            "data1_batches": sorted(DATA1_BATCHES),
            "all_data1_pass_ids": [rp.pass_id for rp in all_passes],
            "evaluated_pass_ids": [rp.pass_id for rp in evaluation_passes],
            "pass_limit": int(args.pass_limit),
            "max_origins_per_segment": int(args.max_origins_per_segment),
            "inventory": inventory.to_dict(orient="records"),
        },
        "hard_checks": {
            "rolling_origin_h1_vs_stored_innovations": h1_check,
            "per_pass_state_reconstruction": [
                item["state_reconstruction_h1_check"] for item in pass_audits
            ],
            "frozen_recorded_envelope": envelope_check,
            "all_predictions_finite": True,
            "all_targets_within_segment": True,
            "all_required_guardrail_horizon_pairs_present": True,
        },
        "counts": {
            "data1_passes_loaded": len(all_passes),
            "passes_evaluated": len(evaluation_passes),
            "prediction_rows": len(predictions),
            "long_channel_rows": len(long_predictions),
            "prediction_rows_by_guardrail_horizon": [
                {
                    "guardrail_label": str(keys[0]),
                    "horizon_updates": int(keys[1]),
                    "rows": int(len(group)),
                }
                for keys, group in predictions.groupby(
                    ["guardrail_label", "horizon_updates"], sort=True
                )
            ],
            "pass_audits": pass_audits,
        },
        "recorded_envelopes": {
            "quantiles": [0.025, 0.975],
            "phase_specific": True,
            "normalization": "each recorded/predicted value divided by its pass's frozen Data1 scale",
            "scopes": {
                "frozen_all_data1": "frozen canonical all-Data1 phase envelope",
                "leave_one_pass_out": (
                    "held-out pass excluded; phase envelope rebuilt from the other 12 "
                    "Data1 passes"
                ),
            },
            "occupancy_object": "forward prediction at the recorded target phase/time",
            "recorded_target_occupancy_also_reported": True,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "dependency_sha256": file_hashes(dependencies),
        "outputs": output_manifest,
        "runtime_seconds_before_audit_write": runtime_before_audit,
    }
    audit_path = out_dir / "farch_forward_audit.json"
    audit_path.write_text(
        json.dumps(json_ready(audit), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"[done] output={out_dir} predictions={len(predictions):,} "
        f"runtime={time.perf_counter() - started:.2f}s",
        flush=True,
    )
    print(
        f"[check] h1_bitwise_equal={h1_check['bitwise_equal']} "
        f"max_abs_difference={h1_check['max_abs_difference']:.17g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
