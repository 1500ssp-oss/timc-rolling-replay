"""Segment-aware time and distance reconstruction for archived-update replay.

Timestamp gaps are retained for audit, but they are not passed through the
controller or converted into fictitious observed strip length.  Stateful
components are reset by the caller whenever ``segment_id`` changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DISTANCE_STEP_RATIO_LOW = 0.40
DISTANCE_STEP_RATIO_HIGH = 3.00
TIMESTAMP_DISTANCE_RATIO_LOW = 0.50
TIMESTAMP_DISTANCE_RATIO_HIGH = 2.00


@dataclass(frozen=True)
class ArchivedTimebaseV2:
    elapsed_control_s: np.ndarray
    dt_control_s: np.ndarray
    dt_gap_audit_s: np.ndarray
    observed_dlength_m: np.ndarray
    gap_distance_unknown_m: np.ndarray
    segment_id: np.ndarray
    segment_start: np.ndarray
    source: np.ndarray
    raw_timestamp_consistent: np.ndarray
    valid_observed_distance: np.ndarray

    @property
    def elapsed_s(self) -> np.ndarray:
        return self.elapsed_control_s

    @property
    def effective_dt_s(self) -> np.ndarray:
        return self.dt_control_s

    @property
    def distance_step_m(self) -> np.ndarray:
        return self.observed_dlength_m


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def derive_segmented_archived_timebase(
    frame: pd.DataFrame,
    distance_reference_m: float,
    gap_ratio_low: float = DISTANCE_STEP_RATIO_LOW,
    gap_ratio_high: float = DISTANCE_STEP_RATIO_HIGH,
    local_window: int = 5,
) -> ArchivedTimebaseV2:
    """Build a segment-local control clock and an observed-distance measure.

    The first row of every segment receives the median of the first up to five
    supported intervals in that segment.  The original positive timestamp gap
    remains available in ``dt_gap_audit_s``.  Unsupported length jumps and all
    segment-start rows contribute zero observed distance.
    """
    n = len(frame)
    if n == 0:
        raise ValueError("Cannot derive a timebase for an empty frame.")
    if local_window < 1:
        raise ValueError("local_window must be positive.")

    timestamp = _numeric(frame, "raw_timestamp")
    speed_series = pd.Series(_numeric(frame, "speed_avg"))
    speed = speed_series.interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
    length = _numeric(frame, "strip_length_actual")

    raw_dt = np.diff(timestamp, prepend=np.nan)
    observed_step = np.abs(np.diff(length, prepend=np.nan))
    valid_distance = (
        np.isfinite(observed_step)
        & (observed_step >= gap_ratio_low * distance_reference_m)
        & (observed_step <= gap_ratio_high * distance_reference_m)
    )
    distance_dt = np.full(n, np.nan, dtype=float)
    positive_speed = np.isfinite(speed) & (speed > 1e-9)
    distance_dt[valid_distance & positive_speed] = (
        observed_step[valid_distance & positive_speed]
        * 60.0
        / speed[valid_distance & positive_speed]
    )

    raw_chronological = np.isfinite(raw_dt) & (raw_dt > 0.0)
    raw_distance = speed * raw_dt / 60.0
    consistency_ratio = raw_distance / np.maximum(observed_step, 1e-12)
    raw_consistent = (
        raw_chronological
        & valid_distance
        & np.isfinite(consistency_ratio)
        & (consistency_ratio >= TIMESTAMP_DISTANCE_RATIO_LOW)
        & (consistency_ratio <= TIMESTAMP_DISTANCE_RATIO_HIGH)
    )

    # Any row unsupported by an observed length increment starts a new replay
    # segment.  A valid length increment can still reconstruct a local dt even
    # when its raw timestamp is inconsistent.
    break_segment = (~valid_distance) & (~raw_consistent)
    break_segment[0] = False
    segment_id = np.cumsum(break_segment.astype(int))
    segment_start = np.r_[True, segment_id[1:] != segment_id[:-1]]

    supported = raw_consistent | (valid_distance & np.isfinite(distance_dt) & (distance_dt > 0.0))
    candidate_dt = np.where(raw_consistent, raw_dt, distance_dt).astype(float)
    supported_values = candidate_dt[supported & np.isfinite(candidate_dt) & (candidate_dt > 0.0)]
    if supported_values.size == 0:
        fallback = raw_dt[raw_chronological]
        if fallback.size == 0:
            raise ValueError("No supported positive interval is available for this pass.")
        global_dt = float(np.median(fallback))
    else:
        global_dt = float(np.median(supported_values))

    dt_control = candidate_dt.copy()
    source = np.where(raw_consistent, "raw_timestamp_consistent", "distance_over_speed").astype(object)
    unique_segments = np.unique(segment_id)
    for segment in unique_segments:
        indices = np.flatnonzero(segment_id == segment)
        start = int(indices[0])
        candidates = [
            int(i)
            for i in indices[1:]
            if supported[i] and np.isfinite(candidate_dt[i]) and candidate_dt[i] > 0.0
        ][:local_window]
        local_dt = float(np.median(candidate_dt[candidates])) if candidates else global_dt
        dt_control[start] = local_dt
        source[start] = "segment_start_local_median" if candidates else "segment_start_global_median"
        unsupported_inside = indices[
            ~np.isfinite(dt_control[indices]) | (dt_control[indices] <= 0.0)
        ]
        if unsupported_inside.size:
            dt_control[unsupported_inside] = local_dt
            source[unsupported_inside] = "segment_local_median_fallback"

    if not np.all(np.isfinite(dt_control) & (dt_control > 0.0)):
        raise ValueError("Segment-local control time reconstruction failed.")

    observed_dlength = np.where(valid_distance, observed_step, 0.0).astype(float)
    observed_dlength[segment_start] = 0.0

    dt_gap_audit = np.where(raw_chronological, raw_dt, np.nan).astype(float)
    gap_distance_unknown = np.zeros(n, dtype=float)
    timed_gap = segment_start & raw_chronological & positive_speed
    gap_distance_unknown[timed_gap] = speed[timed_gap] * raw_dt[timed_gap] / 60.0
    untimed_gap = segment_start & ~timed_gap & np.isfinite(observed_step)
    gap_distance_unknown[untimed_gap] = observed_step[untimed_gap]
    gap_distance_unknown[0] = 0.0

    elapsed = np.cumsum(dt_control) - dt_control[0]
    return ArchivedTimebaseV2(
        elapsed_control_s=elapsed,
        dt_control_s=dt_control,
        dt_gap_audit_s=dt_gap_audit,
        observed_dlength_m=observed_dlength,
        gap_distance_unknown_m=gap_distance_unknown,
        segment_id=segment_id,
        segment_start=segment_start,
        source=source,
        raw_timestamp_consistent=raw_consistent,
        valid_observed_distance=valid_distance,
    )
