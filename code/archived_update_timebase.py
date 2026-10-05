"""Shared archived-update timebase reconstruction.

This module deliberately assigns no global sampling period. Predictive models
operate on the archived-update index; only physical-time quantities consume the
row-wise effective elapsed time reconstructed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArchivedTimebase:
    elapsed_s: np.ndarray
    effective_dt_s: np.ndarray
    distance_step_m: np.ndarray
    segment_id: np.ndarray
    source: np.ndarray


def derive_archived_timebase(
    frame: pd.DataFrame,
    distance_reference_m: float,
    gap_ratio_low: float = 0.40,
    gap_ratio_high: float = 3.00,
) -> ArchivedTimebase:
    """Reconcile raw timestamps with distance/speed without a fixed dt fallback."""
    n = len(frame)
    if n == 0:
        raise ValueError("Cannot derive a timebase for an empty frame.")
    timestamp = pd.to_numeric(frame.get("raw_timestamp"), errors="coerce").to_numpy(float)
    speed = pd.to_numeric(frame.get("speed_avg"), errors="coerce").interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
    raw_dt = np.diff(timestamp, prepend=np.nan)
    if "strip_length_actual" in frame:
        length = pd.to_numeric(frame["strip_length_actual"], errors="coerce").to_numpy(float)
        distance = np.abs(np.diff(length, prepend=np.nan))
    else:
        distance = np.full(n, np.nan, dtype=float)

    valid_distance = (distance >= gap_ratio_low * distance_reference_m) & (distance <= gap_ratio_high * distance_reference_m)
    distance_dt = distance * 60.0 / np.maximum(speed, 1e-9)
    raw_distance = speed * raw_dt / 60.0
    consistency_ratio = raw_distance / np.maximum(distance, 1e-9)
    raw_consistent = (
        np.isfinite(raw_dt)
        & (raw_dt > 0.0)
        & valid_distance
        & np.isfinite(consistency_ratio)
        & (consistency_ratio >= 0.5)
        & (consistency_ratio <= 2.0)
    )
    raw_chronological = np.isfinite(raw_dt) & (raw_dt > 0.0)
    data_candidates = np.concatenate((raw_dt[raw_chronological], distance_dt[valid_distance & np.isfinite(distance_dt) & (distance_dt > 0.0)]))
    if not data_candidates.size:
        raise ValueError("No positive timestamp or distance/speed interval is available for this pass.")
    initial_dt = float(np.median(data_candidates))

    dt = np.where(raw_consistent, raw_dt, np.where(valid_distance, distance_dt, np.where(raw_chronological, raw_dt, initial_dt))).astype(float)
    source = np.where(
        raw_consistent,
        "raw_timestamp_consistent",
        np.where(valid_distance, "distance_over_speed", np.where(raw_chronological, "raw_timestamp_unreconciled", "data_derived_initial")),
    ).astype(object)
    dt[0] = initial_dt
    source[0] = "data_derived_initial"
    if not np.all(np.isfinite(dt) & (dt > 0.0)):
        raise ValueError("Effective time reconstruction produced a non-positive interval.")

    # A chronological but distance-inconsistent jump is still a discontinuity:
    # neither admissible source supports carrying controller state across it.
    break_segment = (~valid_distance) & (~raw_consistent)
    break_segment[0] = False
    segment = np.cumsum(break_segment.astype(int))
    distance_step = np.where(valid_distance, distance, speed * dt / 60.0)
    elapsed = np.cumsum(dt) - dt[0]
    return ArchivedTimebase(elapsed, dt, distance_step, segment, source)
