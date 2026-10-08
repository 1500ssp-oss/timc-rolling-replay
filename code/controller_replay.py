from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from segmented_archived_timebase import derive_segmented_archived_timebase
from supported_command_motion import reset_command_motion, supported_total_variation
from timc_paths import package_path


ROOT = Path(__file__).resolve().parent
FINAL_ROOT = ROOT / "outputs"
PROTOCOL_PATH = ROOT / "protocol.py"
RNG_SEED = 20260622
ARCHIVED_UPDATE_HORIZON = 10
# Kept solely to convert documented legacy per-update move limits to a physical
# slew rate. It is not a sampling period or predictor time step.
LEGACY_MOVE_LIMIT_REFERENCE_DT_S = 0.50
MODEL_PREDICTION_CACHE: dict[tuple[str, str, int, float], np.ndarray] = {}
INNOVATION_INJECTION = np.vstack([np.eye(3), np.zeros((6, 3))])


def load_protocol():
    spec = importlib.util.spec_from_file_location("protocol_stage_alloc", PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load protocol module from {PROTOCOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["protocol_stage_alloc"] = module
    spec.loader.exec_module(module)
    return module


protocol = load_protocol()


@dataclass(frozen=True)
class ExposureConfig:
    block: str
    label: str
    scenario_id: str
    base_control: str = "C7"
    stage_mode: str = "full"
    allocation_mode: str = "full"
    mpc_share: float = 0.10
    ff_share: float = 0.42
    use_pid: bool = True
    # None resolves to True (the historical default); every locked/sensitivity
    # configuration must set this explicitly so the policy description in the
    # manuscript matches the executed filter exactly.
    use_filter: bool | None = None
    gate_enabled: bool = True
    no_gate_release_gain: bool = False
    clamp_scale: float = 1.0
    amplitude_bound: float | None = None
    dynamic_penalty: bool = True
    force_common_outer_limits: bool = False
    outer_du_scale: float = 1.0
    pid_gain_scale: float = 1.0
    adrc_disturbance_scale: float = 1.0
    stage_source: str = "position"
    speed_slope_threshold: float = 0.0
    legacy_move_limit_reference_dt_s: float = LEGACY_MOVE_LIMIT_REFERENCE_DT_S
    timebase: str = "archived_update"
    distance_step_reference_m: float = 0.78
    gap_ratio_low: float = 0.40
    gap_ratio_high: float = 3.00
    outer_du_hard_scale: float = 2.0
    predictor_bank_path: str = ""
    residual_gain: float = 0.82
    command_gain_scale: float = 1.0
    cross_coupling_scale: float = 1.0
    response_variant: str = "hybrid"
    output_clip_scale: float = 8.0
    anti_windup_mode: str = "vectorwise"
    audit_energy: bool = False
    measurement_noise_level: float = 0.0
    reference_kind: str = "none"
    seed_group: str = ""
    comparison_base: str = ""
    actuator_delay_updates: int = 0
    actuator_deadzone: float = 0.0
    measurement_delay_updates: int = 0
    fault_mode: str = "none"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class LockedModelBank:
    """Frozen archive-driven predictor: persistence for T/S and GRU for h.

    At observation/control row k, the GRU window ends at k and forecasts the
    next archived row k+1. The prepared features are the recorded archive,
    not a policy's counterfactual thickness trajectory. Thus this is an
    exogenous, one-archived-update forecast, not a closed-loop predictor.
    """
    def __init__(self, path: str):
        from model_lopo import SeqModel
        payload = torch.load(package_path(path), map_location="cpu", weights_only=False)
        # Frozen inference stays on CPU to reproduce the locked manuscript run.
        # GPU is reserved for deep-model retraining, where it materially helps.
        self.device = torch.device("cpu")
        self.features = list(payload["features"])
        self.window = int(payload["window"])
        self.mu = np.asarray(payload["mu"], dtype=np.float32)
        self.sd = np.asarray(payload["sd"], dtype=np.float32)
        self.model = SeqModel(str(payload["kind"]), len(self.features)).to(self.device).eval()
        self.model.load_state_dict(payload["state_dict"])

    def prepare(
        self,
        frame: pd.DataFrame,
        dt: np.ndarray,
        distance: np.ndarray,
        segment_ids: np.ndarray | None = None,
    ) -> None:
        """Cache forecasts by their information/decision row, not target row.

        cached[k] = h_hat(k+1 | k), using only archive features through row k.
        As in model_lopo.sequences, eligibility also requires the row just
        before the WINDOW inputs to belong to the same contiguous segment.
        The next target must continue that segment. No target value is read.
        Segment starts, short segments and the last archive row have no eligible
        next-row forecast and fall back to persistence in predict().
        """
        work = frame.copy()
        n = len(work)
        dt = np.asarray(dt, dtype=float)
        distance = np.asarray(distance, dtype=float)
        if dt.shape != (n,) or distance.shape != (n,):
            raise ValueError("dt and distance must each have one value per archive row")
        work["effective_dt_s"] = dt
        work["distance_step_m"] = distance
        if segment_ids is None:
            segment_ids = np.zeros(n, dtype=int)
        segment_ids = np.asarray(segment_ids, dtype=int)
        if segment_ids.shape != (n,):
            raise ValueError("segment_ids must have one value per archive row")
        # Use contiguous runs even if an externally supplied segment ID recurs.
        runs = np.zeros(n, dtype=int)
        if n > 1:
            runs[1:] = np.cumsum(segment_ids[1:] != segment_ids[:-1])
        # Preserve the frozen training preprocessing; only past observations
        # are used for missingness, exactly as in model_lopo.sequences.
        values = work[self.features].apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0).to_numpy(np.float32)
        windows, indices = [], []
        window = self.window
        for k in range(window, n - 1):
            # Training: inputs k-window+1..k, target k+1. Cache under k so
            # the controller consumes the forecast at its information time.
            if not np.all(runs[k - window : k + 2] == runs[k]):
                continue
            windows.append(values[k - window + 1 : k + 1])
            indices.append(k)
        self.cached = np.full(n, np.nan, dtype=float)
        self.target_rows = np.full(n, -1, dtype=int)
        if windows:
            x = (np.asarray(windows, dtype=np.float32) - self.mu) / self.sd
            with torch.no_grad():
                pred = self.model(torch.tensor(x, device=self.device)).detach().cpu().numpy()
            info_rows = np.asarray(indices, dtype=int)
            self.cached[info_rows] = pred
            self.target_rows[info_rows] = info_rows + 1

    def predict(self, k: int, current: np.ndarray, horizon: int) -> np.ndarray:
        """Repeat h_hat(k+1|k) over the planning stages, with T/S held current.

        The stages do not add forecast steps. Invalid archive indices raise
        instead of silently wrapping negative indices or borrowing another row.
        """
        if not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer archive row index")
        if k < 0 or (hasattr(self, "cached") and k >= len(self.cached)):
            raise IndexError("k lies outside the prepared archive")
        thickness = self.cached[k] if hasattr(self, "cached") else np.nan
        if not np.isfinite(thickness):
            return np.tile(current, (horizon, 1))
        pred = np.tile(current, (horizon, 1))
        pred[:, 1] = thickness
        return pred


def stage_weights_custom(
    suite,
    k: int,
    steps: int,
    p: Any,
    mode: str,
    phase_override: str | None = None,
) -> np.ndarray:
    accel = np.array([8.5, 4.8, 4.0], dtype=float)
    steady = np.array([5.0, 8.6, 5.2], dtype=float)
    decel = np.array([5.2, 5.2, 8.8], dtype=float)
    fixed = np.array([5.5, 6.0, 5.5], dtype=float)
    uniform = np.array([6.0, 6.0, 6.0], dtype=float)
    phase = phase_override or suite.phase_name(k, steps, p)
    if mode == "fixed":
        return fixed
    if mode == "uniform":
        return uniform
    if mode == "full":
        return {"accel": accel, "steady": steady, "decel": decel}.get(phase, steady)
    if mode == "shuffled":
        return {"accel": steady, "steady": decel, "decel": accel}.get(phase, steady)
    if mode == "reversed":
        return {"accel": decel, "steady": accel, "decel": steady}.get(phase, steady)
    raise ValueError(f"Unknown stage_mode={mode}")


def replay_timebase(frame: pd.DataFrame, config: ExposureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return elapsed time, positive control dt, distance step, and segment ids.

    Timestamp mode preserves the native logged updates while preventing PID,
    rate limits, and controller state from bridging invalid timestamp/distance
    gaps. The learned predictor remains sample-domain and is labeled as such.
    """
    n = len(frame)
    if config.timebase != "archived_update":
        raise ValueError(f"Unknown timebase={config.timebase}")
    timebase = derive_segmented_archived_timebase(
        frame,
        config.distance_step_reference_m,
        config.gap_ratio_low,
        config.gap_ratio_high,
    )
    return (
        timebase.elapsed_s,
        timebase.effective_dt_s,
        timebase.distance_step_m,
        timebase.segment_id,
        timebase.source,
    )


def speed_phase_labels(frame: pd.DataFrame, dt_s: float | np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Offline full-file speed-phase annotation.

    The centered smoothing below deliberately uses the full speed profile: the
    manuscript discloses phase labels as offline full-file annotations, not as
    causal online controller inputs.  Do not reuse this function inside a
    causal online loop.
    """
    speed = pd.to_numeric(frame.get("speed_avg", pd.Series(index=frame.index, dtype=float)), errors="coerce")
    speed = speed.interpolate(limit_direction="both").fillna(0.0)
    values = speed.to_numpy(dtype=float)
    if np.isscalar(dt_s):
        # Offline full-file annotation (see docstring).
        smooth = speed.rolling(window=9, center=True, min_periods=1).median().to_numpy(dtype=float)
        slope = np.gradient(smooth, max(float(dt_s), 1e-9))
    else:
        dt = np.asarray(dt_s, dtype=float)
        elapsed = np.cumsum(dt) - dt[0]
        uniform_dt = max(float(np.median(dt[np.isfinite(dt) & (dt > 0.0)])), 1e-6)
        uniform_time = np.arange(0.0, max(float(elapsed[-1]), uniform_dt) + uniform_dt, uniform_dt)
        uniform_speed = np.interp(uniform_time, elapsed, values)
        width = max(3, int(round(4.0 / uniform_dt)))
        # Offline full-file annotation (see docstring).
        smooth_uniform = pd.Series(uniform_speed).rolling(window=width, center=True, min_periods=1).median().to_numpy(dtype=float)
        slope = np.interp(elapsed, uniform_time, np.gradient(smooth_uniform, uniform_dt))
        smooth = np.interp(elapsed, uniform_time, smooth_uniform)
    if threshold <= 0.0:
        centre = slope[max(0, len(slope) // 4) : max(1, 3 * len(slope) // 4)]
        med = float(np.median(centre)) if len(centre) else 0.0
        mad = float(np.median(np.abs(centre - med))) if len(centre) else 0.0
        threshold = max(3.0 * 1.4826 * mad, 0.005 * max(float(np.nanmedian(np.abs(smooth))), 1.0))
    labels = np.full(len(slope), "steady", dtype=object)
    labels[slope > threshold] = "accel"
    labels[slope < -threshold] = "decel"
    return labels, slope


def response_matrix_custom(suite, p: Any, horizon: int, allocation_mode: str) -> np.ndarray:
    if allocation_mode in {"full", "full_alloc"}:
        return suite.response_matrix(p, horizon, True)
    if allocation_mode in {"diag", "no_offdiag"}:
        return suite.response_matrix(p, horizon, False)
    if allocation_mode == "wrong_sign":
        resp = suite.response_matrix(p, horizon, True).copy()
        mask = np.ones((3, 3), dtype=bool)
        np.fill_diagonal(mask, False)
        resp[:, mask] *= -1.0
        return resp
    raise ValueError(f"Unknown allocation_mode={allocation_mode}")


def solve_mpc_custom(
    suite,
    base_pred: np.ndarray,
    weights: np.ndarray,
    p: Any,
    u_prev: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    allocation_mode: str,
    dynamic_penalty: bool,
    penalty_scale: float = 1.0,
) -> np.ndarray:
    horizon = len(base_pred)
    resp = response_matrix_custom(suite, p, horizon, allocation_mode)
    scales = np.array([p.tension_scale, p.thickness_scale, p.flatness_scale])
    rows, rhs = [], []
    for t in range(horizon):
        for j in range(3):
            row = resp[t, j, :] / max(scales[j], 1e-9)
            rows.append(np.sqrt(weights[j]) * row)
            rhs.append(np.sqrt(weights[j]) * base_pred[t, j] / max(scales[j], 1e-9))
    a = np.vstack(rows)
    b = np.asarray(rhs)
    penalty = np.array([22.0, 26.0, 24.0]) if dynamic_penalty else np.array([16.0, 20.0, 18.0])
    h = a.T @ a + np.diag(penalty * penalty_scale)
    g = a.T @ b
    # The strictly positive diagonal move penalty makes h positive definite;
    # the audited formulation therefore has no singular-Hessian branch.
    du = -np.linalg.solve(h, g)
    if allocation_mode in {"diag", "no_offdiag"}:
        du_lim = np.array([0.095, 0.080, 0.095])
    else:
        du_lim = np.array([0.070, 0.060, 0.070])
    return np.clip(u_prev + np.clip(du, -du_lim, du_lim), u_min, u_max)


def reference_signal(kind: str, k: int, n: int, scale: np.ndarray) -> np.ndarray:
    return protocol.reference_signal(kind, k, n, scale)


def norm3(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(values, dtype=float), axis=1)


def total_variation_from_cols(df: pd.DataFrame, cols: list[str]) -> float:
    if df.empty:
        return float("nan")
    arr = df[cols].to_numpy(dtype=float)
    segment_ids = df["segment_id"].to_numpy() if "segment_id" in df else None
    return supported_total_variation(arr, segment_ids)


def add_mechanism_metrics(metrics: dict[str, Any], df: pd.DataFrame, p: Any) -> dict[str, Any]:
    scales = np.array([p.tension_scale, p.thickness_scale, p.flatness_scale], dtype=float)
    metrics["preclamp_total_variation"] = total_variation_from_cols(df, ["u_preclamp_speed", "u_preclamp_gap", "u_preclamp_shape"])
    metrics["postclamp_total_variation"] = total_variation_from_cols(df, ["u_speed", "u_gap", "u_shape"])
    metrics["mpc_raw_total_variation"] = total_variation_from_cols(df, ["u_mpc_speed", "u_mpc_gap", "u_mpc_shape"])
    metrics["mpc_component_total_variation"] = total_variation_from_cols(df, ["u_mpc_component_speed", "u_mpc_component_gap", "u_mpc_component_shape"])
    metrics.update(reset_command_motion(
        df[["u_speed", "u_gap", "u_shape"]].to_numpy(dtype=float),
        df["segment_id"].to_numpy() if "segment_id" in df else None,
    ))
    metrics["mean_norm_u_mpc_raw"] = float(norm3(df[["u_mpc_speed", "u_mpc_gap", "u_mpc_shape"]].to_numpy()).mean())
    metrics["mean_norm_u_mpc_component"] = float(norm3(df[["u_mpc_component_speed", "u_mpc_component_gap", "u_mpc_component_shape"]].to_numpy()).mean())
    metrics["mean_norm_u_ff"] = float(norm3(df[["u_ff_speed", "u_ff_gap", "u_ff_shape"]].to_numpy()).mean())
    metrics["mean_norm_u_pid"] = float(norm3(df[["u_pid_speed", "u_pid_gap", "u_pid_shape"]].to_numpy()).mean())
    metrics["mean_norm_u_adrc"] = float(norm3(df[["u_adrc_speed", "u_adrc_gap", "u_adrc_shape"]].to_numpy()).mean())
    metrics["mean_norm_u_preclamp"] = float(norm3(df[["u_preclamp_speed", "u_preclamp_gap", "u_preclamp_shape"]].to_numpy()).mean())
    metrics["mean_norm_u_applied"] = float(norm3(df[["u_speed", "u_gap", "u_shape"]].to_numpy()).mean())
    metrics["sat_ratio_logged"] = float(df["sat_flag"].mean()) if "sat_flag" in df else float("nan")
    amp_cols = ["amplitude_active_speed", "amplitude_active_gap", "amplitude_active_shape"]
    if all(col in df for col in amp_cols):
        amp = df[amp_cols].to_numpy(dtype=bool)
        metrics["amplitude_projection_ratio"] = float(np.any(amp, axis=1).mean())
        for idx, channel in enumerate(["speed", "gap", "shape"]):
            metrics[f"amplitude_active_{channel}_ratio"] = float(amp[:, idx].mean())
    if "raw_to_applied_L2" in df:
        metrics["raw_to_applied_L2_mean"] = float(df["raw_to_applied_L2"].mean())
        metrics["raw_to_applied_L2_p95"] = float(df["raw_to_applied_L2"].quantile(0.95))
    if "prediction_alpha" in df:
        metrics["prediction_alpha_mean"] = float(df["prediction_alpha"].mean())
        metrics["prediction_alpha_active_ratio"] = float((df["prediction_alpha"] > 0.0).mean())
        metrics["prediction_confidence_mean"] = float(df["prediction_confidence"].mean())
        metrics["prediction_support_mean"] = float(df["prediction_support"].mean())

    def phase_rms(phase: str, col: str, scale: float) -> float:
        subset = df[df["phase"].eq(phase)]
        if subset.empty:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(subset[col].to_numpy(dtype=float) / max(scale, 1e-9)))))

    metrics["accel_T_RMSn"] = phase_rms("accel", "T_error", scales[0])
    metrics["steady_h_RMSn"] = phase_rms("steady", "h_error", scales[1])
    metrics["decel_S_RMSn"] = phase_rms("decel", "S_error", scales[2])

    for phase_col, prefix in [("phase", "position"), ("speed_phase", "speed")]:
        if phase_col not in df.columns:
            continue
        for phase in ["accel", "steady", "decel"]:
            subset = df[df[phase_col].eq(phase)]
            metrics[f"{prefix}_{phase}_coverage"] = float(len(subset) / max(len(df), 1))
            for idx, target in enumerate(["T", "h", "S"]):
                if subset.empty:
                    value = float("nan")
                else:
                    values = subset[f"{target}_error"].to_numpy(dtype=float) / max(scales[idx], 1e-9)
                    value = float(np.sqrt(np.mean(np.square(values))))
                metrics[f"{prefix}_{phase}_{target}_RMSn"] = value

        stage_values = [
            metrics.get(f"{prefix}_{phase}_{target}_RMSn", float("nan"))
            for phase in ["accel", "steady", "decel"]
            for target in ["T", "h", "S"]
        ]
        finite = np.asarray([v for v in stage_values if np.isfinite(v)], dtype=float)
        metrics[f"{prefix}_stage_balanced_RMSn"] = float(finite.mean()) if finite.size else float("nan")

    off_sq = []
    for _, row in df.iterrows():
        vals = np.array([row["T_error"], row["h_error"], row["S_error"]], dtype=float) / scales
        if row["phase"] == "accel":
            off_sq.extend([vals[1] ** 2, vals[2] ** 2])
        elif row["phase"] == "steady":
            off_sq.extend([vals[0] ** 2, vals[2] ** 2])
        else:
            off_sq.extend([vals[0] ** 2, vals[1] ** 2])
    metrics["offtarget_RMSn"] = float(np.sqrt(np.mean(off_sq))) if off_sq else float("nan")
    return metrics


def run_exposure_control(
    suite,
    rp,
    config: ExposureConfig,
    seed: int,
    locked_pid: Any | None,
    repeat_idx: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    p = rp.p
    scales = protocol.scale_vec(p)
    y_real = rp.y_real
    innovations = rp.innovations
    n = len(y_real)
    base_flags = suite.control_flags(config.base_control)
    prediction_allocation = config.allocation_mode not in {"diag", "no_offdiag"}
    use_filter = config.use_filter if config.use_filter is not None else True

    if config.clamp_scale >= 90.0 and config.amplitude_bound is None:
        command_limit = 99.0
        move_scale = 99.0
    else:
        command_limit = float(config.amplitude_bound) if config.amplitude_bound is not None else 0.55 * config.clamp_scale
        move_scale = config.clamp_scale
    u_min = np.array([-command_limit, -command_limit, -command_limit], dtype=float)
    u_max = np.array([command_limit, command_limit, command_limit], dtype=float)

    active_filter = "ekf" if (use_filter or base_flags["ekf_baseline"]) else "raw"

    elapsed_s, dt_s, distance_step_m, segment_ids, time_source = replay_timebase(rp.df, config)
    # Causal segment-local median residual: only past/current rows from the
    # same supported segment enter the measurement played back at row k.
    hf = np.zeros_like(y_real, dtype=float)
    for segment_id in np.unique(segment_ids):
        index = np.flatnonzero(segment_ids == segment_id)
        segment_values = y_real[index]
        trailing = (
            pd.DataFrame(segment_values)
            .rolling(window=7, center=False, min_periods=1)
            .median()
            .to_numpy()
        )
        hf[index] = segment_values - trailing
    u = np.zeros(3, dtype=float)
    if config.base_control == "C0" and locked_pid is not None:
        pid = protocol.make_locked_pid(suite, p, float(dt_s[0]), locked_pid, extra_move_scale=min(config.clamp_scale, 99.0))
    else:
        pid = suite.make_pid(config.base_control, p, float(dt_s[0]))
    pid.kp *= config.pid_gain_scale
    pid.ki *= config.pid_gain_scale
    pid.kd *= config.pid_gain_scale
    pid.anti_windup_mode = config.anti_windup_mode
    speed_phases, speed_slopes = speed_phase_labels(rp.df, dt_s, config.speed_slope_threshold)
    model_bank = LockedModelBank(config.predictor_bank_path) if config.predictor_bank_path else None
    if model_bank is not None:
        cache_key = (config.predictor_bank_path, rp.file, len(rp.df), float(np.round(dt_s.sum(), 6)))
        if cache_key in MODEL_PREDICTION_CACHE:
            model_bank.cached = MODEL_PREDICTION_CACHE[cache_key]
        else:
            model_bank.prepare(rp.df, dt_s, distance_step_m, segment_ids)
            MODEL_PREDICTION_CACHE[cache_key] = model_bank.cached
    x = np.r_[y_real[0], y_real[0], np.zeros(3, dtype=float)]
    filt = None
    solve_ms: list[float] = []
    filter_ms: list[float] = []
    sat_count = 0
    invalid_command_count = 0
    fallback_count = 0
    constraint_count = 0
    output_clip_count = 0
    output_clip_channel_count = np.zeros(3, dtype=int)
    output_clip_eligible_transition_count = 0
    command_energy: list[np.ndarray] = []
    innovation_energy: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    command_history: list[np.ndarray] = []
    measurement_history: list[np.ndarray] = []
    last_valid_z = suite.h_func(x).copy()

    for k in range(n):
        segment_start = k == 0 or segment_ids[k] != segment_ids[k - 1]
        if segment_start:
            # A stateful controller/filter must never bridge a timestamp or
            # distance discontinuity in the archived production record.
            pid.reset()
            u = np.zeros(3, dtype=float)
            x = np.r_[y_real[k], y_real[k], np.zeros(3, dtype=float)]
            x0 = x + np.r_[0.08 * scales * rng.normal(size=3), np.zeros(6, dtype=float)]
            filt = suite.build_filter(active_filter, x0, p, n, "S5", seed + 73 + k, 80)
        ref = reference_signal(config.reference_kind, k, n, scales)
        # The nominal archived measurement is deterministic and causal.  Any
        # synthetic measurement noise is an explicit scenario setting below;
        # its scale is tied to the frozen Data1 catalogue rather than to a
        # full-pass statistic that would use future rows.
        z = suite.h_func(x) + 0.25 * hf[k]
        if config.measurement_noise_level > 0.0:
            z += rng.normal(0.0, config.measurement_noise_level * scales, size=3)
        measurement_history.append(z.copy())
        if config.measurement_delay_updates > 0:
            z = measurement_history[max(0, len(measurement_history) - config.measurement_delay_updates - 1)].copy()
        fault_active = int(0.45 * n) <= k < int(0.55 * n)
        if fault_active and config.fault_mode == "nonfinite":
            z[:] = np.nan
        elif fault_active and config.fault_mode == "input_channel_scale_extreme":
            z[0] *= 1000.0
        if config.fault_mode != "none":
            if not np.all(np.isfinite(z)) or np.any(np.abs(z) > 20.0 * scales):
                fallback_count += 1
                z = last_valid_z.copy()
            else:
                last_valid_z = z.copy()
        filter_start = time.perf_counter()
        if use_filter or base_flags["ekf_baseline"]:
            x_est, residual = filt.step(u, z, k, allocation=prediction_allocation)
        else:
            x_est, residual = filt.step(u, z, k, allocation=True)
        if config.base_control == "C7" and config.gate_enabled:
            rho = 0.10
            x_est[:3] = rho * x_est[:3] + (1.0 - rho) * z
        filter_ms.append((time.perf_counter() - filter_start) * 1000.0)

        y = suite.h_func(x_est)
        y_for_control = y - ref
        start = time.perf_counter()
        prediction_branch_requested = abs(config.mpc_share) > 1e-12 or abs(config.ff_share) > 1e-12 or config.no_gate_release_gain
        if prediction_branch_requested or model_bank is not None:
            pred = suite.predict_stage_narx(
                x_est,
                p,
                k,
                n,
                ARCHIVED_UPDATE_HORIZON,
                str(base_flags["predictor"]),
                allocation=prediction_allocation,
            )
        else:
            pred = np.tile(y, (ARCHIVED_UPDATE_HORIZON, 1))
        if model_bank is not None:
            pred = model_bank.predict(k, y, ARCHIVED_UPDATE_HORIZON)
        prediction_alpha = float(getattr(model_bank, "last_alpha", 1.0)) if model_bank is not None else float(prediction_branch_requested)
        prediction_confidence = float(getattr(model_bank, "last_confidence", 1.0)) if model_bank is not None else float(prediction_branch_requested)
        prediction_support = float(getattr(model_bank, "last_support", 1.0)) if model_bank is not None else float(prediction_branch_requested)
        pred_for_control = pred - ref.reshape(1, 3)
        stage_phase = str(speed_phases[k]) if config.stage_source == "speed" else None
        weights = stage_weights_custom(suite, k, n, p, config.stage_mode, phase_override=stage_phase)

        u_mpc = np.zeros(3, dtype=float)
        u_ff = np.zeros(3, dtype=float)
        u_pid = np.zeros(3, dtype=float)
        u_adrc = np.zeros(3, dtype=float)
        u_target_parts = np.zeros(3, dtype=float)
        mpc_active = bool(base_flags["mpc"])
        if mpc_active and prediction_branch_requested:
            u_mpc = solve_mpc_custom(
                suite,
                pred_for_control,
                weights,
                p,
                u,
                u_min,
                u_max,
                config.allocation_mode,
                config.dynamic_penalty,
            )
            u_ff = suite.feedforward_from_prediction(pred_for_control, p)
            mpc_share = config.mpc_share
            ff_share = config.ff_share
            if config.no_gate_release_gain:
                mpc_share = max(mpc_share, 0.16)
                ff_share = max(ff_share, 0.70)
            # A custom predictor bank can expose a confidence/support gate.
            # The gate attenuates the prediction branch only; feedback remains
            # continuously active and is never switched off by model failure.
            u_target_parts += prediction_alpha * (mpc_share * u_mpc + ff_share * u_ff)
        elif (not mpc_active) and str(base_flags["predictor"]) != "persistence" and config.base_control in {"C2", "C3"}:
            u_ff = suite.feedforward_from_prediction(pred_for_control, p)
            u_target_parts += prediction_alpha * u_ff

        if config.use_pid and base_flags["pid"]:
            u_pid = pid.step(-y_for_control, dt=float(dt_s[k]))
            u_target_parts += u_pid

        if config.base_control == "ADRC":
            u_adrc = config.adrc_disturbance_scale * np.array(
                [
                    0.20 * x_est[6] / max(p.rollforce_scale, 1e-9),
                    -0.16 * x_est[7],
                    -0.18 * x_est[8],
                ],
                dtype=float,
            )
            u_target_parts += u_adrc

        u_target_unclipped = u_target_parts.copy()
        u_preclamp = np.clip(u_target_unclipped, u_min, u_max)
        amplitude_active = ~np.isclose(u_target_unclipped, u_preclamp)
        if not np.all(np.isfinite(u_preclamp)):
            invalid_command_count += 1
            u_preclamp = np.nan_to_num(u_preclamp, nan=0.0, posinf=command_limit, neginf=-command_limit)
        solve_ms.append((time.perf_counter() - start) * 1000.0)

        du_base = (
            np.array([0.065, 0.052, 0.065], dtype=float)
            if (mpc_active or config.force_common_outer_limits)
            else np.array([0.090, 0.075, 0.090], dtype=float)
        )
        # Historical per-update limits are converted through the explicitly
        # named legacy reference, then enforced with each row's effective dt.
        du_nominal = du_base * move_scale * config.outer_du_scale
        if config.timebase == "archived_update":
            rate_per_s = du_nominal / max(float(config.legacy_move_limit_reference_dt_s), 1e-9)
            du_hard = du_nominal * config.outer_du_hard_scale
            du_lim = np.minimum(rate_per_s * float(dt_s[k]), du_hard)
        else:
            du_lim = du_nominal
        du = np.clip(u_preclamp - u, -du_lim, du_lim)
        sat_flag = bool(np.any(np.abs(u_preclamp - u) > du_lim + 1e-12))
        if sat_flag:
            sat_count += 1
        u = np.clip(u + du, u_min, u_max)
        if not np.all(np.isfinite(u)):
            invalid_command_count += 1
            u = np.nan_to_num(u, nan=0.0, posinf=command_limit, neginf=-command_limit)
        command_history.append(u.copy())
        delay = max(0, int(config.actuator_delay_updates))
        u_effective = command_history[max(0, len(command_history) - delay - 1)].copy()
        if config.actuator_deadzone > 0.0:
            u_effective[np.abs(u_effective) < config.actuator_deadzone] = 0.0

        phase = suite.phase_name(k, n, p)
        control_error = x[:3] - ref
        rows.append(
            {
                "k": k,
                "time_s": float(elapsed_s[k]),
                "timestamp_dt_s": float(dt_s[k]),
                "distance_step_m": float(distance_step_m[k]),
                "segment_id": int(segment_ids[k]),
                "timebase": config.timebase,
                "effective_time_source": str(time_source[k]),
                "phase": phase,
                "speed_phase": str(speed_phases[k]),
                "speed_slope": float(speed_slopes[k]),
                "control": config.label,
                "base_control": config.base_control,
                "block": config.block,
                "scenario": config.scenario_id,
                "seed": seed,
                "repeat_idx": repeat_idx,
                "pass_id": rp.pass_id,
                "condition_id": rp.condition_id,
                "trial_id": rp.trial_id,
                "file": rp.file,
                "stage_mode": config.stage_mode,
                "allocation_mode": config.allocation_mode,
                "mpc_share": config.mpc_share,
                "ff_share": config.ff_share,
                "clamp_scale": config.clamp_scale,
                "amplitude_bound": command_limit,
                "T_error": control_error[0],
                "h_error": control_error[1],
                "S_error": control_error[2],
                "T_control_error": control_error[0],
                "h_control_error": control_error[1],
                "S_control_error": control_error[2],
                "T_est": y[0],
                "h_est": y[1],
                "S_est": y[2],
                "u_mpc_speed": u_mpc[0],
                "u_mpc_gap": u_mpc[1],
                "u_mpc_shape": u_mpc[2],
                "u_mpc_component_speed": config.mpc_share * u_mpc[0],
                "u_mpc_component_gap": config.mpc_share * u_mpc[1],
                "u_mpc_component_shape": config.mpc_share * u_mpc[2],
                "u_ff_speed": u_ff[0],
                "u_ff_gap": u_ff[1],
                "u_ff_shape": u_ff[2],
                "u_pid_speed": u_pid[0],
                "u_pid_gap": u_pid[1],
                "u_pid_shape": u_pid[2],
                "u_adrc_speed": u_adrc[0],
                "u_adrc_gap": u_adrc[1],
                "u_adrc_shape": u_adrc[2],
                "u_target_unclipped_speed": u_target_unclipped[0],
                "u_target_unclipped_gap": u_target_unclipped[1],
                "u_target_unclipped_shape": u_target_unclipped[2],
                "u_preclamp_speed": u_preclamp[0],
                "u_preclamp_gap": u_preclamp[1],
                "u_preclamp_shape": u_preclamp[2],
                "amplitude_active_speed": int(amplitude_active[0]),
                "amplitude_active_gap": int(amplitude_active[1]),
                "amplitude_active_shape": int(amplitude_active[2]),
                "u_speed": u[0],
                "u_gap": u[1],
                "u_shape": u[2],
                "u_effective_speed": u_effective[0],
                "u_effective_gap": u_effective[1],
                "u_effective_shape": u_effective[2],
                "du_speed": du[0],
                "du_gap": du[1],
                "du_shape": du[2],
                "sat_flag": int(sat_flag),
                "raw_to_applied_L2": float(np.linalg.norm(u_target_unclipped - u)),
                "residual_norm": residual,
                "mpc_active": int(mpc_active),
                "prediction_alpha": prediction_alpha,
                "prediction_confidence": prediction_confidence,
                "prediction_support": prediction_support,
                "output_transition_evaluated": 0,
                "output_clip_any_active": 0,
                "output_clip_T_active": 0,
                "output_clip_h_active": 0,
                "output_clip_S_active": 0,
            }
        )
        if k < n - 1 and segment_ids[k + 1] == segment_ids[k]:
            output_clip_eligible_transition_count += 1
            rows[-1]["output_transition_evaluated"] = 1
            step_kwargs = {
                "allocation": True,
                "model_variant": config.response_variant,
                "command_gain_scale": config.command_gain_scale,
                "cross_coupling_scale": config.cross_coupling_scale,
                "output_clip_scale": config.output_clip_scale,
            }
            innovation = config.residual_gain * innovations[k]
            nominal_response = (
                config.response_variant == "hybrid"
                and config.command_gain_scale == 1.0
                and config.cross_coupling_scale == 1.0
                and config.output_clip_scale == 8.0
                and config.actuator_delay_updates == 0
                and config.actuator_deadzone == 0.0
            )
            if nominal_response:
                # Preserve the frozen nominal arithmetic exactly.  The more
                # general branch below is used only when a response factor is
                # deliberately varied by a sensitivity experiment.
                x_model = suite.plant_step(
                    x, u, p, k, n, np.zeros(5), np.zeros(6), allocation=True
                )
                x_next = x_model.copy()
                x_next[:3] += innovation
            else:
                x_model = suite.plant_step(
                    x, u_effective, p, k, n, np.zeros(5), np.zeros(6), **step_kwargs
                )
                x_next = x_model + INNOVATION_INJECTION @ innovation
            clip_limit = config.output_clip_scale * scales
            if np.isfinite(config.output_clip_scale):
                clipped_channels = np.abs(x_next[:3]) > clip_limit + 1e-12
                at_model_limit = np.isclose(np.abs(x_model[:3]), clip_limit, rtol=0.0, atol=1e-10)
                active_clip = clipped_channels | at_model_limit
                output_clip_count += int(np.any(active_clip))
                output_clip_channel_count += active_clip.astype(int)
                rows[-1]["output_clip_any_active"] = int(np.any(active_clip))
                rows[-1]["output_clip_T_active"] = int(active_clip[0])
                rows[-1]["output_clip_h_active"] = int(active_clip[1])
                rows[-1]["output_clip_S_active"] = int(active_clip[2])
                x_next[:3] = np.clip(x_next[:3], -clip_limit, clip_limit)
            if config.audit_energy:
                x_zero = suite.plant_step(x, np.zeros(3), p, k, n, np.zeros(5), np.zeros(6), **step_kwargs)
                command_energy.append(np.abs(x_model[:3] - x_zero[:3]) / np.maximum(scales, 1e-9))
                innovation_energy.append(np.abs(innovation) / np.maximum(scales, 1e-9))
            x = x_next
            constraint_count += int(np.any(np.abs(suite.h_func(x)) > 3.0 * scales))

    diag = {
        "control": config.label,
        "base_control": config.base_control,
        "seed": seed,
        "repeat_idx": repeat_idx,
        "scenario": config.scenario_id,
        "active_filter": active_filter,
        "sat_count": sat_count,
        "sat_ratio": float(sat_count / max(n, 1)),
        "constraint_count": int(constraint_count),
        "constraint_ratio": float(constraint_count / max(output_clip_eligible_transition_count, 1)),
        "output_clip_count": int(output_clip_count),
        "output_clip_eligible_transition_count": int(output_clip_eligible_transition_count),
        "output_clip_ratio": float(output_clip_count / max(output_clip_eligible_transition_count, 1)),
        "output_clip_T_count": int(output_clip_channel_count[0]),
        "output_clip_h_count": int(output_clip_channel_count[1]),
        "output_clip_S_count": int(output_clip_channel_count[2]),
        "output_clip_T_ratio": float(output_clip_channel_count[0] / max(output_clip_eligible_transition_count, 1)),
        "output_clip_h_ratio": float(output_clip_channel_count[1] / max(output_clip_eligible_transition_count, 1)),
        "output_clip_S_ratio": float(output_clip_channel_count[2] / max(output_clip_eligible_transition_count, 1)),
        "pid_aw_vector_freeze_count": int(getattr(pid, "vector_freeze_count", 0)),
        "pid_aw_speed_count": int(getattr(pid, "channel_freeze_count", np.zeros(3, dtype=int))[0]),
        "pid_aw_gap_count": int(getattr(pid, "channel_freeze_count", np.zeros(3, dtype=int))[1]),
        "pid_aw_shape_count": int(getattr(pid, "channel_freeze_count", np.zeros(3, dtype=int))[2]),
        "fallback_count": fallback_count,
        "invalid_command_count": invalid_command_count,
        "filter_ms_mean": float(np.mean(filter_ms)) if filter_ms else 0.0,
        "filter_ms_p95": float(np.percentile(filter_ms, 95)) if filter_ms else 0.0,
        "solve_ms_mean": float(np.mean(solve_ms)) if solve_ms else 0.0,
        "solve_ms_p95": float(np.percentile(solve_ms, 95)) if solve_ms else 0.0,
        "solve_ms_max": float(np.max(solve_ms)) if solve_ms else 0.0,
        "p95_compute_ms": float(np.percentile(np.asarray(filter_ms) + np.asarray(solve_ms), 95)) if filter_ms and solve_ms else 0.0,
        "timebase": config.timebase,
        "segments": int(segment_ids.max() + 1) if len(segment_ids) else 0,
        "median_timestamp_dt_s": float(np.median(dt_s)) if len(dt_s) else float("nan"),
        "median_distance_step_m": float(np.median(distance_step_m)) if len(distance_step_m) else float("nan"),
        "predictor_device": str(getattr(model_bank, "device", "none")) if model_bank is not None else "none",
    }
    if command_energy:
        command_arr = np.vstack(command_energy)
        innovation_arr = np.vstack(innovation_energy)
        for idx, channel in enumerate(["T", "h", "S"]):
            diag[f"command_effect_{channel}_median"] = float(np.median(command_arr[:, idx]))
            diag[f"command_effect_{channel}_p95"] = float(np.percentile(command_arr[:, idx], 95))
            diag[f"innovation_{channel}_median"] = float(np.median(innovation_arr[:, idx]))
            diag[f"innovation_{channel}_p95"] = float(np.percentile(innovation_arr[:, idx], 95))
    return pd.DataFrame(rows), diag


def suite_metrics_with_config(suite, rp, df: pd.DataFrame, diag: dict[str, Any], config: ExposureConfig, envelope: pd.DataFrame) -> dict[str, Any]:
    scenario = protocol.ScenarioSpec(
        config.scenario_id,
        config.block,
        residual_gain=config.residual_gain,
        measurement_noise_level=config.measurement_noise_level,
        move_limit_scale=config.clamp_scale,
        reference_kind=config.reference_kind,
        ablation=config.label,
    )
    metrics = protocol.suite_metrics(suite, rp, df, diag, config.base_control, scenario)
    metrics.update(protocol.envelope_metrics_for_curve(suite, rp, df, envelope))
    metrics.update(add_mechanism_metrics({}, df, rp.p))
    for key, value in diag.items():
        if key.startswith(("output_clip_", "pid_aw_", "command_effect_", "innovation_")):
            metrics[key] = value
    duration_s = float(df["timestamp_dt_s"].sum()) if "timestamp_dt_s" in df else float("nan")
    distance_m = float(df["distance_step_m"].sum()) if "distance_step_m" in df else float("nan")
    raw_tv = float(metrics.get("postclamp_total_variation", metrics.get("control_total_variation", float("nan"))))
    metrics["TV_t_per_s"] = raw_tv / max(duration_s, 1e-9)
    metrics["TV_L_per_100m"] = 100.0 * raw_tv / max(distance_m, 1e-9) if np.isfinite(distance_m) and distance_m > 0.0 else float("nan")
    metrics["duration_s"] = duration_s
    metrics["distance_m"] = distance_m
    metrics.update(
        {
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
        }
    )
    metrics.update(
        {
            "block": config.block,
            "scenario": config.scenario_id,
            "control": config.label,
            "base_control": config.base_control,
            "stage_mode": config.stage_mode,
            "allocation_mode": config.allocation_mode,
            "mpc_share": config.mpc_share,
            "ff_share": config.ff_share,
            "clamp_scale": config.clamp_scale,
            "amplitude_bound": config.amplitude_bound if config.amplitude_bound is not None else 0.55 * config.clamp_scale,
            "use_pid": int(config.use_pid),
            "use_filter": int(config.use_filter if config.use_filter is not None else True),
            "gate_enabled": int(config.gate_enabled),
            "dynamic_penalty": int(config.dynamic_penalty),
            "force_common_outer_limits": int(config.force_common_outer_limits),
            "outer_du_scale": config.outer_du_scale,
            "pid_gain_scale": config.pid_gain_scale,
            "adrc_disturbance_scale": config.adrc_disturbance_scale,
            "stage_source": config.stage_source,
            "archived_update_horizon": ARCHIVED_UPDATE_HORIZON,
            "legacy_move_limit_reference_dt_s": config.legacy_move_limit_reference_dt_s,
            "timebase": config.timebase,
            "distance_step_reference_m": config.distance_step_reference_m,
            "gap_ratio_low": config.gap_ratio_low,
            "gap_ratio_high": config.gap_ratio_high,
            "outer_du_hard_scale": config.outer_du_hard_scale,
            "predictor_bank_path": config.predictor_bank_path,
            "residual_gain": config.residual_gain,
            "command_gain_scale": config.command_gain_scale,
            "cross_coupling_scale": config.cross_coupling_scale,
            "response_variant": config.response_variant,
            "output_clip_scale": config.output_clip_scale,
            "anti_windup_mode": config.anti_windup_mode,
            "audit_energy": int(config.audit_energy),
            "measurement_noise_level": config.measurement_noise_level,
            "reference_kind": config.reference_kind,
            "comparison_base": config.comparison_base,
            "actuator_delay_updates": config.actuator_delay_updates,
            "actuator_deadzone": config.actuator_deadzone,
            "measurement_delay_updates": config.measurement_delay_updates,
            "fault_mode": config.fault_mode,
            "repeat_idx": int(diag["repeat_idx"]),
        }
    )
    return metrics


def build_configs() -> list[ExposureConfig]:
    configs: list[ExposureConfig] = []

    def add(block: str, label: str, **kwargs: Any) -> None:
        scenario_id = kwargs.pop("scenario_id", block)
        seed_group = kwargs.pop("seed_group", scenario_id)
        comparison_base = kwargs.pop("comparison_base", "")
        configs.append(
            ExposureConfig(
                block=block,
                label=label,
                scenario_id=scenario_id,
                seed_group=seed_group,
                comparison_base=comparison_base,
                **kwargs,
            )
        )

    # Final deployment interpretation: preserve the conservative controller structure.
    add("deployment", "C7-Full", comparison_base="C7-Full")
    add("deployment", "C7-NoGate", gate_enabled=False, no_gate_release_gain=True, comparison_base="C7-Full")
    add("deployment", "C7-NoStage", stage_mode="fixed", dynamic_penalty=False, comparison_base="C7-Full")
    add("deployment", "C7-NoAlloc", allocation_mode="diag", comparison_base="C7-Full")
    add("deployment", "C7-NoFilter", use_filter=False, comparison_base="C7-Full")
    add("deployment", "C7-NoPID", use_pid=False, comparison_base="C7-Full")
    add("deployment", "C7-NoConstraint", clamp_scale=99.0, comparison_base="C7-Full")
    add("deployment", "C7-TightConstraint", clamp_scale=0.60, comparison_base="C7-Full")
    add("deployment", "PID-only", base_control="C0", mpc_share=0.0, ff_share=0.0, use_filter=False, comparison_base="C7-Full")

    # Fairness audit: identical outer command and move limits for the practical baselines.
    add("fairness", "C7-Locked", force_common_outer_limits=True, comparison_base="PID-Matched")
    add("fairness", "PID-Matched", base_control="C0", mpc_share=0.0, ff_share=0.0, use_filter=False, force_common_outer_limits=True, comparison_base="PID-Matched")
    add("fairness", "ADRC-Matched", base_control="ADRC", mpc_share=0.0, ff_share=0.0, force_common_outer_limits=True, comparison_base="PID-Matched")

    # Data1-only tuning grid. All candidates share the C7 outer safety wrapper and seeds.
    tuning_base = "C7-Data1-Reference"
    add("fairness_tuning", tuning_base, force_common_outer_limits=True, scenario_id="fairness_tuning", seed_group="fairness_tuning", comparison_base=tuning_base)
    for gain in [0.80, 1.00, 1.20]:
        for du_scale in [0.50, 0.65, 0.80, 1.00, 1.20, 1.40]:
            add(
                "fairness_tuning",
                f"PID-g{gain:.2f}-du{du_scale:.2f}",
                scenario_id="fairness_tuning",
                seed_group="fairness_tuning",
                base_control="C0",
                mpc_share=0.0,
                ff_share=0.0,
                use_filter=False,
                force_common_outer_limits=True,
                pid_gain_scale=gain,
                outer_du_scale=du_scale,
                comparison_base=tuning_base,
            )
            add(
                "fairness_tuning",
                f"ADRC-g{gain:.2f}-du{du_scale:.2f}",
                scenario_id="fairness_tuning",
                seed_group="fairness_tuning",
                base_control="ADRC",
                mpc_share=0.0,
                ff_share=0.0,
                force_common_outer_limits=True,
                pid_gain_scale=gain,
                outer_du_scale=du_scale,
                comparison_base=tuning_base,
            )

    # Incremental C7 ladder under the same outer safety wrapper.
    ladder_base = "C7-L0-PID"
    add("c7_ladder", ladder_base, mpc_share=0.0, ff_share=0.0, use_filter=False, gate_enabled=False, stage_mode="fixed", allocation_mode="diag", force_common_outer_limits=True, comparison_base=ladder_base)
    add("c7_ladder", "C7-L1-PID-FF", mpc_share=0.0, ff_share=0.42, use_filter=False, gate_enabled=False, stage_mode="fixed", allocation_mode="diag", force_common_outer_limits=True, comparison_base=ladder_base)
    add("c7_ladder", "C7-L2-PID-FF-MPC", mpc_share=0.10, ff_share=0.42, use_filter=False, gate_enabled=False, stage_mode="fixed", allocation_mode="diag", force_common_outer_limits=True, comparison_base=ladder_base)
    add("c7_ladder", "C7-L3-Gate-EKF", mpc_share=0.10, ff_share=0.42, use_filter=True, gate_enabled=True, stage_mode="fixed", allocation_mode="diag", force_common_outer_limits=True, comparison_base=ladder_base)
    add("c7_ladder", "C7-L4-SpeedStage", mpc_share=0.10, ff_share=0.42, use_filter=True, gate_enabled=True, stage_mode="full", stage_source="speed", allocation_mode="diag", force_common_outer_limits=True, comparison_base=ladder_base)
    add("c7_ladder", "C7-L5-Full", mpc_share=0.10, ff_share=0.42, use_filter=True, gate_enabled=True, stage_mode="full", stage_source="speed", allocation_mode="full", force_common_outer_limits=True, comparison_base=ladder_base)

    # Mechanism exposure: isolate stage weights with a visible MPC proposal and relaxed move clamp.
    for mode, label in [
        ("full", "Stage-Full"),
        ("fixed", "Stage-Fixed"),
        ("uniform", "Stage-Uniform"),
        ("shuffled", "Stage-Shuffled"),
        ("reversed", "Stage-Reversed"),
    ]:
        add(
            "stage_mechanism",
            label,
            scenario_id="stage_lam040_clamp15",
            seed_group="stage_lam040_clamp15",
            stage_mode=mode,
            mpc_share=0.40,
            clamp_scale=1.5,
            dynamic_penalty=True,
            comparison_base="Stage-Full",
        )

    # Mechanism exposure: compare full, diagonal and wrong-sign allocation at two MPC shares.
    for lam in [0.40, 0.60]:
        sid = f"allocation_lam{int(lam * 100):03d}_clamp15"
        add("allocation_mechanism", f"Alloc-Full-lam{int(lam*100):02d}", scenario_id=sid, seed_group=sid, mpc_share=lam, clamp_scale=1.5, allocation_mode="full", comparison_base=f"Alloc-Full-lam{int(lam*100):02d}")
        add("allocation_mechanism", f"Alloc-Diag-lam{int(lam*100):02d}", scenario_id=sid, seed_group=sid, mpc_share=lam, clamp_scale=1.5, allocation_mode="diag", comparison_base=f"Alloc-Full-lam{int(lam*100):02d}")
        add("allocation_mechanism", f"Alloc-WrongSign-lam{int(lam*100):02d}", scenario_id=sid, seed_group=sid, mpc_share=lam, clamp_scale=1.5, allocation_mode="wrong_sign", comparison_base=f"Alloc-Full-lam{int(lam*100):02d}")

    # Clamp masking: pair Full/NoStage/NoAlloc under progressively relaxed final move clamps.
    for clamp in [1.0, 1.5, 2.0]:
        sid = f"clamp_masking_scale_{str(clamp).replace('.', 'p')}"
        add("clamp_masking", f"Clamp{clamp:g}-Full", scenario_id=sid, seed_group=sid, clamp_scale=clamp, comparison_base=f"Clamp{clamp:g}-Full")
        add("clamp_masking", f"Clamp{clamp:g}-NoStage", scenario_id=sid, seed_group=sid, clamp_scale=clamp, stage_mode="fixed", dynamic_penalty=True, comparison_base=f"Clamp{clamp:g}-Full")
        add("clamp_masking", f"Clamp{clamp:g}-NoAlloc", scenario_id=sid, seed_group=sid, clamp_scale=clamp, allocation_mode="diag", comparison_base=f"Clamp{clamp:g}-Full")
        add("clamp_masking", f"Clamp{clamp:g}-NoStageNoAlloc", scenario_id=sid, seed_group=sid, clamp_scale=clamp, stage_mode="fixed", allocation_mode="diag", dynamic_penalty=True, comparison_base=f"Clamp{clamp:g}-Full")
    return configs


def condition_weighted_summary(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g0 in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        per_condition = g0.groupby("condition_id", dropna=False)[metric_cols].mean(numeric_only=True)
        row["n_rows"] = int(len(g0))
        row["n_conditions"] = int(per_condition.shape[0])
        for metric in metric_cols:
            if metric in per_condition.columns:
                row[f"{metric}_cw_mean"] = float(per_condition[metric].mean())
                row[f"{metric}_cw_median"] = float(per_condition[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_effect_summary(metrics: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    configs = metrics[["block", "scenario", "control", "comparison_base"]].drop_duplicates()
    rows = []
    metric_cols = [
        "composite_normalized_RMS",
        "control_total_variation",
        "S_out",
        "preclamp_total_variation",
        "sat_ratio_logged",
        "amplitude_projection_ratio",
        "amplitude_active_speed_ratio",
        "amplitude_active_gap_ratio",
        "amplitude_active_shape_ratio",
        "raw_to_applied_L2_mean",
        "raw_to_applied_L2_p95",
        "output_clip_ratio",
        "pid_aw_vector_freeze_count",
        "pid_aw_speed_count",
        "pid_aw_gap_count",
        "pid_aw_shape_count",
        "accel_T_RMSn",
        "steady_h_RMSn",
        "decel_S_RMSn",
        "offtarget_RMSn",
    ]
    for _, cfg in configs.iterrows():
        base = str(cfg["comparison_base"])
        control = str(cfg["control"])
        if not base or control == base:
            continue
        subset = metrics[
            metrics["block"].eq(cfg["block"])
            & metrics["scenario"].eq(cfg["scenario"])
            & metrics["control"].isin([base, control])
        ].copy()
        wide = subset.pivot_table(index=["pass_id", "repeat_idx"], columns="control", values=metric_cols, aggfunc="mean")
        if wide.empty:
            continue
        row = {
            "block": cfg["block"],
            "scenario": cfg["scenario"],
            "control": control,
            "baseline": base,
        }
        for metric in metric_cols:
            try:
                paired = wide[metric][[base, control]].dropna()
            except Exception:
                continue
            if paired.empty:
                continue
            diff = paired[control] - paired[base]
            row[f"{metric}_diff_mean"] = float(diff.mean())
            row[f"{metric}_diff_median"] = float(diff.median())
            denom = float(paired[base].mean())
            row[f"{metric}_rel_diff_pct"] = float(100.0 * diff.mean() / denom) if abs(denom) > 1e-12 else float("nan")

        step_sub = steps[
            steps["block"].eq(cfg["block"])
            & steps["scenario"].eq(cfg["scenario"])
            & steps["control"].isin([base, control])
        ].copy()
        if not step_sub.empty:
            idx = ["pass_id", "repeat_idx", "k"]
            base_steps = step_sub[step_sub["control"].eq(base)].set_index(idx)
            other_steps = step_sub[step_sub["control"].eq(control)].set_index(idx)
            common = base_steps.index.intersection(other_steps.index)
            if len(common) > 0:
                b = base_steps.loc[common, ["u_mpc_speed", "u_mpc_gap", "u_mpc_shape"]].to_numpy(dtype=float)
                a = other_steps.loc[common, ["u_mpc_speed", "u_mpc_gap", "u_mpc_shape"]].to_numpy(dtype=float)
                delta = a - b
                row["E_mpc_vs_base"] = float(np.linalg.norm(delta, axis=1).mean())
                dot = np.sum(a * b, axis=1)
                denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                valid = denom > 1e-12
                if np.any(valid):
                    cos = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
                    row["mpc_angle_deg_vs_base"] = float(np.degrees(np.arccos(cos)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def save_figures(out_dir: Path, deployment: pd.DataFrame, paired: pd.DataFrame) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "axes.grid": True,
            "grid.color": "#E6E8F0",
        }
    )

    dep = deployment.sort_values("control_total_variation_cw_mean", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    colors = ["#A3BEFA" if c == "C7-Full" else "#E2E5EA" for c in dep["control"]]
    ax.barh(dep["control"], dep["control_total_variation_cw_mean"], color=colors, edgecolor="#464C55", linewidth=1.1)
    ax.invert_yaxis()
    ax.set_xlabel("Post-clamp control total variation")
    ax.set_title("Deployment ablation under locked Data2 replay")
    ax.set_axisbelow(True)
    for _, r in dep.iterrows():
        txt = f"TV {r['control_total_variation_cw_mean']:.1f}; RMS {r['composite_normalized_RMS_cw_mean']:.2f}; S_out {r['S_out_cw_mean']:.3f}; sat {r['sat_ratio_logged_cw_mean']:.2f}"
        ax.text(r["control_total_variation_cw_mean"] + max(dep["control_total_variation_cw_mean"]) * 0.012, r["control"], txt, va="center", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "deployment_ablation_tv.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / "deployment_ablation_tv.svg", bbox_inches="tight")
    plt.close(fig)

    if paired.empty or "E_mpc_vs_base" not in paired.columns:
        return
    mech = paired[paired["block"].isin(["stage_mechanism", "allocation_mechanism", "clamp_masking"])].copy()
    mech = mech[mech["E_mpc_vs_base"].notna()].copy()
    mech = mech.sort_values(["block", "scenario", "E_mpc_vs_base"], ascending=[True, True, False])
    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    labels = [f"{r.block}: {r.control}" for r in mech.itertuples()]
    vals = mech["E_mpc_vs_base"].to_numpy(dtype=float)
    ax.barh(labels, vals, color="#CEDFFE", edgecolor="#2E4780", linewidth=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean ||u_MPC(variant) - u_MPC(baseline)||")
    ax.set_title("Mechanism-exposure proposal differences")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "stage_allocation_mechanism_effects.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / "stage_allocation_mechanism_effects.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(FINAL_ROOT / "stage_allocation_ablation"))
    parser.add_argument("--full-root", default=str(FINAL_ROOT))
    parser.add_argument("--dataset", choices=["Data1", "Data2"], default="Data2")
    parser.add_argument("--blocks", default=None, help="Comma-separated config blocks to run")
    parser.add_argument("--config-json", default=None, help="Optional JSON list of serialized ExposureConfig objects")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--data2-pass-limit", type=int, default=None)
    parser.add_argument("--config-limit", type=int, default=None)
    parser.add_argument("--timebase", choices=["archived_update"], default=None, help="Override the archived-update timebase for every selected configuration")
    parser.add_argument("--distance-step-reference-m", type=float, default=0.78)
    parser.add_argument("--no-step-log", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_root))
    full_root = Path(args.full_root).resolve()
    ensure_dir(out_dir / "audit")
    t0 = time.time()
    suite, replay = protocol.load_suite_modules()
    data1_passes = protocol.load_passes(suite, replay, protocol.DATA1_DIR, "Data1")
    data2_passes = protocol.load_passes(suite, replay, protocol.DATA2_DIR, "Data2")
    if args.data2_pass_limit is not None:
        data2_passes = data2_passes[: args.data2_pass_limit]
    envelope_path = full_root / "audit" / "data1_stage_quantile_envelope_p025_p975.csv"
    if envelope_path.exists():
        envelope = pd.read_csv(envelope_path)
    else:
        envelope = protocol.build_quantile_envelope(suite, data1_passes, out_dir / "audit")
    locked_pid = protocol.select_data1_pid_legacy_disabled(
        suite, data1_passes, full_root / "pid_data1_grid"
    )

    if args.config_json:
        payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        configs = [ExposureConfig(**item) for item in payload]
    else:
        configs = build_configs()
    if args.blocks:
        allowed_blocks = {item.strip() for item in args.blocks.split(",") if item.strip()}
        configs = [config for config in configs if config.block in allowed_blocks]
    if args.config_limit is not None:
        configs = configs[: args.config_limit]
    if args.timebase is not None:
        configs = [replace(config, timebase=args.timebase, distance_step_reference_m=args.distance_step_reference_m) for config in configs]
    eval_passes = data1_passes if args.dataset == "Data1" else data2_passes
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "source_protocol": str(PROTOCOL_PATH),
        "timebase": "archived_update_no_fixed_sampling_period",
        "archived_update_horizon": ARCHIVED_UPDATE_HORIZON,
        "legacy_move_limit_reference_dt_s": LEGACY_MOVE_LIMIT_REFERENCE_DT_S,
        "data1_count": len(data1_passes),
        "data2_count": len(data2_passes),
        "evaluation_dataset": args.dataset,
        "evaluation_pass_count": len(eval_passes),
        "repeats": args.repeats,
        "configs": [asdict(c) for c in configs],
        "locked_pid": asdict(locked_pid),
        "purpose": "Two-layer deployment and mechanism-exposure ablation for stage weighting and channel allocation.",
    }
    (out_dir / "stage_allocation_ablation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_rows: list[dict[str, Any]] = []
    step_frames: list[pd.DataFrame] = []
    total = len(configs) * len(eval_passes) * args.repeats
    run_i = 0
    print(f"[setup] out={out_dir}", flush=True)
    print(f"[setup] configs={len(configs)} dataset={args.dataset} passes={len(eval_passes)} repeats={args.repeats} total={total}", flush=True)
    for cfg in configs:
        for rp_i, rp in enumerate(eval_passes):
            for rep in range(args.repeats):
                run_i += 1
                if run_i == 1 or run_i == total or run_i % 50 == 0:
                    print(f"[run] {run_i}/{total} {cfg.label} {rp.pass_id} elapsed={time.time()-t0:,.1f}s", flush=True)
                # The seed depends only on the pass and repeat index.  Every
                # policy and scenario therefore receives the identical random
                # measurement-noise and segment-start realization for a given
                # pass, so paired policy differences cannot originate from
                # different noise draws.  Multi-seed stability is reported
                # separately (see run_seed_sensitivity in publish_tables.py).
                seed = RNG_SEED + rep + 2000 * (rp_i + 1)
                curve, diag = run_exposure_control(suite, rp, cfg, seed, locked_pid, rep)
                metric_rows.append(suite_metrics_with_config(suite, rp, curve, diag, cfg, envelope))
                if not args.no_step_log:
                    step_frames.append(curve)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "stage_allocation_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    if step_frames:
        steps = pd.concat(step_frames, ignore_index=True, sort=False)
        steps.to_csv(out_dir / "stage_allocation_ablation_step_logs.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    else:
        steps = pd.DataFrame()

    metric_cols = [
        "composite_normalized_RMS",
        "control_total_variation",
        "S_out",
        "h_out",
        "control_delta_rms",
        "startup_motion",
        "restart_motion",
        "startup_restart_motion",
        "P95_abs_S",
        "delta_u_over_peak_action",
        "preclamp_total_variation",
        "postclamp_total_variation",
        "mpc_raw_total_variation",
        "mpc_component_total_variation",
        "mean_norm_u_mpc_raw",
        "mean_norm_u_mpc_component",
        "mean_norm_u_ff",
        "mean_norm_u_pid",
        "mean_norm_u_adrc",
        "mean_norm_u_preclamp",
        "mean_norm_u_applied",
        "sat_ratio_logged",
        "accel_T_RMSn",
        "steady_h_RMSn",
        "decel_S_RMSn",
        "offtarget_RMSn",
        "position_stage_balanced_RMSn",
        "speed_stage_balanced_RMSn",
        "S_excess_mean",
        "S_excess_p95",
        "S_excess_cvar95",
        "TV_t_per_s",
        "TV_L_per_100m",
        "duration_s",
        "distance_m",
        "p95_compute_ms",
        "invalid_command_count",
    ]
    summary = condition_weighted_summary(
        metrics,
        ["block", "scenario", "control", "comparison_base", "timebase", "stage_mode", "stage_source", "allocation_mode", "mpc_share", "clamp_scale", "force_common_outer_limits", "outer_du_scale", "pid_gain_scale", "adrc_disturbance_scale"],
        metric_cols,
    )
    summary.to_csv(out_dir / "stage_allocation_ablation_condition_weighted_summary.csv", index=False, encoding="utf-8-sig")
    paired = paired_effect_summary(metrics, steps) if not steps.empty else pd.DataFrame()
    paired.to_csv(out_dir / "stage_allocation_ablation_paired_effects.csv", index=False, encoding="utf-8-sig")

    deployment = summary[summary["block"].eq("deployment")].copy()
    save_figures(out_dir, deployment, paired)
    print(f"[done] metrics={out_dir / 'stage_allocation_ablation_metrics.csv'}", flush=True)
    print(f"[done] summary={out_dir / 'stage_allocation_ablation_condition_weighted_summary.csv'}", flush=True)
    print(f"[done] paired={out_dir / 'stage_allocation_ablation_paired_effects.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "The direct legacy controller_replay CLI is disabled. Use "
        "code/run_pipeline.py or code/batch_sweep.py so locks, manifests and "
        "fresh-output checks are enforced."
    )
