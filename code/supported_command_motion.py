"""Command motion on ordered, contiguous supported archive segments.

Only adjacent rows with an unchanged segment label form an observed transition.
A label that recurs later does not reconnect separate runs. Commands at each
run's first row are moves from the reset zero command, reported separately.
"""
from __future__ import annotations

import numpy as np


def supported_transition_mask(segment_ids, n_rows: int) -> np.ndarray:
    """Return a mask over adjacent transitions in the supplied row order.

    Without segment labels the archive is one continuous run, preserving the
    ordinary adjacent-difference convention. Never sort or group by labels.
    """
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    if segment_ids is None:
        return np.ones(max(n_rows - 1, 0), dtype=bool)
    labels = np.asarray(segment_ids)
    if labels.shape != (n_rows,):
        raise ValueError("segment_ids must have one label per command row")
    return labels[1:] == labels[:-1]


def _command_array(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("commands must be a row-by-channel array")
    return values


def supported_command_deltas(values, segment_ids=None) -> np.ndarray:
    """Return signed changes for supported adjacent transitions only."""
    values = _command_array(values)
    mask = supported_transition_mask(segment_ids, len(values))
    return np.diff(values, axis=0)[mask]


def supported_total_variation(values, segment_ids=None) -> float:
    """Three-channel (or general multichannel) supported L1 command motion."""
    return float(np.abs(supported_command_deltas(values, segment_ids)).sum())


def reset_command_motion(values, segment_ids=None) -> dict[str, float]:
    """Separate first-file startup and later restart moves from reset zero.

    These are applied-command magnitudes at contiguous run starts. They are
    never a difference between the end of one run and the start of another.
    """
    values = _command_array(values)
    mask = supported_transition_mask(segment_ids, len(values))
    if not len(values):
        return {"startup_motion": 0.0, "restart_motion": 0.0, "startup_restart_motion": 0.0}
    startup = float(np.abs(values[0]).sum())
    restart = float(np.abs(values[1:][~mask]).sum())
    return {"startup_motion": startup, "restart_motion": restart, "startup_restart_motion": startup + restart}
