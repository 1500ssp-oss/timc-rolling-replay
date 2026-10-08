from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from archived_update_timebase import ArchivedTimebase, derive_archived_timebase
from segmented_archived_timebase import derive_segmented_archived_timebase
from scale_lock import load_scale_lock, manifest_fields, scale_override_and_provenance
from tail_statistics import upper_tail_count
from supported_command_motion import supported_command_deltas
from timc_paths import data_directory, scale_lock_path


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "engine_core"
SCALE_LOCK_PATH = scale_lock_path()
DATA1_DIR = data_directory("Data1")
DATA2_DIR = data_directory("Data2")
PLAN_DOC = ROOT / "documentation" / "controller_design_protocol.docx"
GENERAL_PROTOCOL_DOC = ROOT / "documentation" / "general_replay_protocol.docx"
PROTOCOL_EXTRACT_DIR = ROOT / "documentation"

EXPECTED_DATA1_PASSES = 13
EXPECTED_DATA2_PASSES = 10
DATA2_REPEATS_REQUIRED = 5
RNG_SEED = 20260622
DISTANCE_STEP_REFERENCE_M = 0.81298828125
ARCHIVED_UPDATE_HORIZON = 10
# Historical move limits were documented per legacy logged update. This is a
# conversion reference only, never a replay sampling period.
LEGACY_MOVE_LIMIT_REFERENCE_DT_S = 0.50


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_suite_modules():
    if not PROJECT_DIR.exists():
        raise FileNotFoundError(PROJECT_DIR)
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    suite = load_module("suite15_protocol", PROJECT_DIR / "15_realdata_closed_loop_suite.py")
    replay = load_module("replay20_protocol", PROJECT_DIR / "20_independent_replay_validation.py")
    return suite, replay


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def elapsed_s(t0: float) -> str:
    return f"{time.time() - t0:,.1f}s"


def stable_hash_mod(text: str, modulus: int = 100000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


@dataclass(frozen=True)
class PidParams:
    kp_scale: float
    ki_scale: float
    kd_scale: float
    move_limit_scale: float

    @property
    def key(self) -> str:
        return (
            f"kp{self.kp_scale:g}_ki{self.ki_scale:g}_"
            f"kd{self.kd_scale:g}_ml{self.move_limit_scale:g}"
        )


@dataclass(frozen=True)
class ReplayPass:
    source: str
    pass_id: str
    condition_id: str
    trial_id: str
    file: str
    df: pd.DataFrame
    p: Any
    y_real: np.ndarray
    innovations: np.ndarray


def scale_vec(p) -> np.ndarray:
    return np.array([p.tension_scale, p.thickness_scale, p.flatness_scale], dtype=float)


def archived_timebase(rp: ReplayPass) -> ArchivedTimebase:
    return derive_archived_timebase(rp.df, DISTANCE_STEP_REFERENCE_M)


@dataclass(frozen=True)
class ScenarioSpec:
    scenario: str
    group: str
    residual_gain: float = 0.82
    measurement_noise_level: float = 0.0
    move_limit_scale: float = 1.0
    reference_kind: str = "none"
    predictor_override: str | None = None
    ablation: str | None = None


def protocol_paths() -> dict[str, str]:
    return {
        "redesign_plan_docx": str(PLAN_DOC),
        "general_protocol_docx": str(GENERAL_PROTOCOL_DOC),
        "redesign_plan_extract": str(PROTOCOL_EXTRACT_DIR / "c7_redesign_plan.txt"),
        "general_protocol_extract": str(PROTOCOL_EXTRACT_DIR / "general_protocol.txt"),
    }


def csv_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.csv"), key=lambda p: p.name.lower())


def condition_id_from_file(file_name: str) -> str:
    stem = Path(str(file_name)).stem
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem).strip()
    return stem


def load_passes(suite, replay, data_dir: Path, source: str) -> list[ReplayPass]:
    dataset = suite.load_real_process_data(str(data_dir))
    frames = list(dataset.frames)
    scale_lock = load_scale_lock(SCALE_LOCK_PATH)
    passes: list[ReplayPass] = []
    for idx, df in enumerate(frames, start=1):
        entry = float(df["entry_nominal_from_file"].iloc[0])
        exit_ = float(df["exit_nominal_from_file"].iloc[0])
        scale_override, provenance = scale_override_and_provenance(
            scale_lock, entry, exit_
        )
        # With an override, make_real_pass never reads this pass's response
        # columns to derive normalizers.  Data2 values therefore cannot tune
        # gains, emulator magnitudes, or reported normalized metrics.
        p = suite.make_real_pass(
            df, scale_override=scale_override, scale_provenance=provenance
        )
        # Frames may be sorted by parsed gauge rather than filename.  Preserve
        # the loader's own pass identity instead of zipping against a separate
        # alphabetical file list.
        file_name = Path(str(p.pass_file)).name
        y_real = replay.measured_outputs(df)
        # Innovations must respect the same segment boundaries as the replay:
        # hidden states are reset at every unsupported gap, exactly as in
        # run_exposure_control.
        timebase = derive_segmented_archived_timebase(df, DISTANCE_STEP_REFERENCE_M)
        innovations_payload = replay.build_replay_innovations(
            suite, y_real, p, segment_ids=timebase.segment_id
        )
        innovations = innovations_payload[0] if isinstance(innovations_payload, tuple) else innovations_payload
        passes.append(
            ReplayPass(
                source=source,
                pass_id=f"{source}_P{idx:02d}",
                condition_id=condition_id_from_file(file_name),
                trial_id=Path(str(file_name)).stem,
                file=file_name,
                df=df,
                p=p,
                y_real=y_real,
                innovations=innovations,
            )
        )
    return passes


def scale_lock_manifest() -> dict[str, Any]:
    payload = load_scale_lock(SCALE_LOCK_PATH)
    return manifest_fields(payload) | {"scale_lock_path": "config/data1_scale_lock.json"}


def base_pid_vectors(p) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # These gains are locked before Data2 use.  Kd is intentionally available
    # because the protocol asks for a Kd scale grid; kd_scale=0 remains present.
    scales = scale_vec(p)
    kp_base = np.array([-0.62 / scales[0], 0.58 / scales[1], 0.62 / scales[2]], dtype=float)
    ki_base = np.array([-0.10 / scales[0], 0.08 / scales[1], 0.08 / scales[2]], dtype=float)
    kd_base = np.array([-0.018 / scales[0], 0.014 / scales[1], 0.014 / scales[2]], dtype=float)
    base_limit = np.array([0.40, 0.36, 0.40], dtype=float)
    return kp_base, ki_base, kd_base, base_limit


def make_locked_pid(suite, p, dt: float, params: PidParams, extra_move_scale: float = 1.0):
    kp, ki, kd, base_limit = base_pid_vectors(p)
    move_scale = params.move_limit_scale * extra_move_scale
    return suite.PID(
        kp * params.kp_scale,
        ki * params.ki_scale,
        kd * params.kd_scale,
        dt,
        -base_limit * move_scale,
        base_limit * move_scale,
    )


@contextlib.contextmanager
def patched_make_pid(suite, params: PidParams | None, extra_move_scale: float = 1.0):
    original = suite.make_pid

    if params is None:
        yield
        return

    def _factory(control_id, p, dt):
        if control_id == "C0":
            return make_locked_pid(suite, p, dt, params, extra_move_scale=extra_move_scale)
        return original(control_id, p, dt)

    suite.make_pid = _factory
    try:
        yield
    finally:
        suite.make_pid = original


def pid_candidate_grid() -> list[PidParams]:
    kp_scales = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    ki_scales = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    kd_scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    move_scales = [0.6, 0.8, 1.0, 1.2, 1.4]
    return [
        PidParams(kp, ki, kd, ml)
        for kp, ki, kd, ml in itertools.product(kp_scales, ki_scales, kd_scales, move_scales)
    ]


def fast_pid_replay_rmsc(suite, rp: ReplayPass, params: PidParams, residual_gain: float = 0.82) -> float:
    """Disabled legacy shortcut for the Data1 PID selection.

    The shortcut historically differed from the released replay in its
    timebase, segment handling, measurement path, seed-dependent noise,
    outer projection and definition of composite RMS.  Keeping a second
    implementation would make it too easy to reintroduce those differences.
    PID candidates are now scored only through
    ``controller_replay.run_exposure_control`` by
    ``run_pid_selection_audit.py``.
    """
    del suite, rp, params, residual_gain
    raise RuntimeError(
        "fast_pid_replay_rmsc is a disabled legacy shortcut; use "
        "run_pid_selection_audit.py, which executes the canonical PID replay."
    )


def select_data1_pid_legacy_disabled(
    suite,
    data1_passes: list[ReplayPass],
    out_dir: Path,
    max_candidates: int | None = None,
) -> PidParams:
    """Disabled legacy selector; use the canonical-replay audit entry point."""
    del suite, data1_passes, out_dir, max_candidates
    raise RuntimeError(
        "select_data1_pid_legacy_disabled is not an executable selector; execute "
        "code/run_pid_selection_audit.py so candidates share the exact "
        "batch-sweep measurement, seed, timebase and projection path."
    )


def reference_signal(kind: str, k: int, n: int, scale: np.ndarray) -> np.ndarray:
    if kind == "none":
        return np.zeros(3, dtype=float)
    start = int(0.45 * n)
    if k < start:
        return np.zeros(3, dtype=float)
    if kind == "h_step":
        return np.array([0.0, 0.22 * scale[1], 0.0], dtype=float)
    if kind == "S_step":
        return np.array([0.0, 0.0, -0.22 * scale[2]], dtype=float)
    if kind == "joint_step":
        return np.array([0.0, 0.16 * scale[1], -0.16 * scale[2]], dtype=float)
    raise ValueError(f"Unknown reference kind: {kind}")


def apply_ablation_flags(flags: dict[str, Any], ablation: str | None) -> dict[str, Any]:
    flags = dict(flags)
    if not ablation or ablation == "C7-Full":
        return flags
    if ablation == "C7-NoStage":
        flags["dynamic_weights"] = False
    elif ablation == "C7-NoAlloc":
        flags["allocation"] = False
    elif ablation == "C7-NoFilter":
        flags["filter"] = False
    elif ablation == "C7-NoPID":
        flags["pid"] = False
    elif ablation == "C7-NoGate":
        flags["no_gate"] = True
    elif ablation == "C7-NoConstraint":
        flags["no_constraint"] = True
    elif ablation == "C7-TightConstraint":
        flags["tight_constraint"] = True
    return flags


def run_replay_control_protocol(
    # Auxiliary scenario path. The publication pipeline executes
    # controller_replay.run_exposure_control via batch_sweep.py. This helper
    # uses the same causal trailing residual and pass-only seed convention.
    suite,
    rp: ReplayPass,
    control_id: str,
    seed: int,
    scenario: ScenarioSpec,
    locked_pid: PidParams | None = None,
    label_override: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raise RuntimeError(
        "run_replay_control_protocol is a disabled auxiliary implementation; "
        "use controller_replay.run_exposure_control through batch_sweep.py."
    )
    overrides = dict(overrides or {})
    rng = np.random.default_rng(seed)
    p = rp.p
    scales = scale_vec(p)
    y_real = rp.y_real
    innovations = rp.innovations
    n = len(y_real)
    base_flags = suite.control_flags(control_id)
    flags = apply_ablation_flags(base_flags, scenario.ablation)
    if scenario.predictor_override:
        flags["predictor"] = scenario.predictor_override
    override_flags = {
        "use_filter": "filter",
        "use_pid": "pid",
        "dynamic_weights": "dynamic_weights",
        "allocation": "allocation",
    }
    for override_name, flag_name in override_flags.items():
        if override_name in overrides:
            flags[flag_name] = bool(overrides[override_name])
    timebase = archived_timebase(rp)
    effective_dt = timebase.effective_dt_s
    controller_dt_s = float(effective_dt[0])
    if flags.get("tight_constraint"):
        effective_move_scale = min(scenario.move_limit_scale, 0.60)
    elif flags.get("no_constraint"):
        effective_move_scale = 99.0
    else:
        effective_move_scale = scenario.move_limit_scale
    if locked_pid is not None and control_id == "C0":
        pid = make_locked_pid(suite, p, controller_dt_s, locked_pid, extra_move_scale=effective_move_scale)
    else:
        pid = suite.make_pid(control_id, p, controller_dt_s)

    x = np.r_[y_real[0], y_real[0], np.zeros(3, dtype=float)]
    x0 = x + np.r_[0.08 * scales * rng.normal(size=3), np.zeros(6, dtype=float)]
    active_filter = "ekf" if flags["ekf_baseline"] else "ekf" if flags["filter"] else "raw"
    filt = suite.build_filter(active_filter, x0, p, n, "S5", seed + 73, 80)
    limit = float(overrides.get("u_limit", 0.55 * effective_move_scale))
    if flags.get("no_constraint"):
        limit = 99.0
    u_min = np.array([-limit, -limit, -limit], dtype=float)
    u_max = np.array([limit, limit, limit], dtype=float)
    u = np.zeros(3, dtype=float)
    # Causal trailing-median residual aligned with the publication replay in
    # controller_replay.run_exposure_control.
    hf = y_real - pd.DataFrame(y_real).rolling(window=7, center=False, min_periods=1).median().to_numpy()
    noise_scale = np.maximum(np.std(hf, axis=0), 0.010 * scales)
    solve_ms: list[float] = []
    filter_ms: list[float] = []
    sat_count = 0
    constraint_count = 0
    invalid_command_count = 0
    fallback_count = 0
    rows: list[dict[str, Any]] = []
    phase_labels = overrides.get("phase_labels")
    stage_weights_override = overrides.get("stage_weights")
    measurement_delay = max(0, int(overrides.get("measurement_delay_samples", 0)))
    dropout_length = max(0, int(overrides.get("dropout_length", 0)))
    dropout_start = int(float(overrides.get("dropout_start_ratio", 0.45)) * n)
    dropout_end = min(n, dropout_start + dropout_length)
    dropout_mode = str(overrides.get("dropout_mode", "hold"))
    freeze_channel = overrides.get("freeze_channel")
    freeze_length = max(0, int(overrides.get("freeze_length", 0)))
    freeze_start = int(float(overrides.get("freeze_start_ratio", 0.55)) * n)
    freeze_end = min(n, freeze_start + freeze_length)
    z_history: list[np.ndarray] = []
    last_valid_z = suite.h_func(x).copy()
    if "timestamp_raw" in rp.df.columns:
        raw_timestamp = pd.to_numeric(rp.df["timestamp_raw"], errors="coerce").to_numpy(dtype=float)
        first_valid = raw_timestamp[np.isfinite(raw_timestamp)]
        raw_time_s = raw_timestamp - first_valid[0] if first_valid.size else timebase.elapsed_s
    else:
        raw_time_s = timebase.elapsed_s

    for k in range(n):
        if k == 0 or timebase.segment_id[k] != timebase.segment_id[k - 1]:
            pid.reset()
            u = np.zeros(3, dtype=float)
            x = np.r_[y_real[k], y_real[k], np.zeros(6, dtype=float)]
        ref = reference_signal(scenario.reference_kind, k, n, scales)
        z = suite.h_func(x) + 0.25 * hf[k] + rng.normal(0.0, 0.015 * noise_scale)
        if scenario.measurement_noise_level > 0.0:
            z += rng.normal(0.0, scenario.measurement_noise_level * scales, size=3)
        z_history.append(z.copy())
        if measurement_delay > 0:
            source_idx = max(0, len(z_history) - measurement_delay - 1)
            z = z_history[source_idx].copy()
        if dropout_start <= k < dropout_end:
            if dropout_mode == "nonfinite":
                z[:] = np.nan
            else:
                z = last_valid_z.copy()
        if not np.all(np.isfinite(z)):
            fallback_count += 1
            z = last_valid_z.copy()
        else:
            last_valid_z = z.copy()
        filter_start = time.perf_counter()
        if flags["filter"] or flags["ekf_baseline"]:
            x_est, residual = filt.step(u, z, k, allocation=bool(flags["allocation"]))
        else:
            x_est, residual = filt.step(u, z, k, allocation=True)
        if control_id == "C7" and not flags.get("no_gate"):
            rho = 0.10
            x_est[:3] = rho * x_est[:3] + (1.0 - rho) * z
        filter_ms.append((time.perf_counter() - filter_start) * 1000.0)

        y = suite.h_func(x_est)
        y_for_control = y - ref
        start = time.perf_counter()
        pred = suite.predict_stage_narx(
            x_est, p, k, n, ARCHIVED_UPDATE_HORIZON, str(flags["predictor"]), allocation=bool(flags["allocation"])
        )
        pred_for_control = pred - ref.reshape(1, 3)
        u_target = np.zeros(3, dtype=float)
        weights = suite.stage_control_weights(k, n, p, bool(flags["dynamic_weights"]))
        if scenario.reference_kind != "none":
            weights = weights * np.array([0.9, 1.15, 1.15])
        mpc_active = False
        if flags["mpc"]:
            mpc_active = True
            penalty = {
                "LSTM-MPC": 0.88,
                "DMC-KF": 1.20,
                "ATT-MPC": 0.95,
                "ROBUST-MPC": 1.65,
                "ADAPTIVE-MPC": 1.05,
            }.get(control_id, 1.0)
            u_mpc = suite.solve_mpc(
                pred_for_control,
                weights,
                p,
                u,
                u_min,
                u_max,
                bool(flags["allocation"]),
                bool(flags["dynamic_weights"]),
                penalty_scale=penalty,
            )
            ff = suite.feedforward_from_prediction(pred_for_control, p)
            if control_id == "C7":
                ff_gain = 0.42
                mpc_gain = 0.10
                if flags.get("no_gate"):
                    ff_gain = 0.70
                    mpc_gain = 0.16
                u_target += mpc_gain * u_mpc + ff_gain * ff
            elif control_id in ["LSTM-MPC", "ATT-MPC"]:
                u_target += 0.16 * u_mpc + 0.32 * ff
            elif control_id == "DMC-KF":
                u_target += 0.22 * u_mpc + 0.16 * ff
            elif control_id in ["ROBUST-MPC", "ADAPTIVE-MPC"]:
                u_target += 0.25 * u_mpc + 0.28 * ff
            else:
                u_target += 0.20 * u_mpc + 0.22 * ff
        elif str(flags["predictor"]) != "persistence" and control_id in {"C2", "C3"}:
            u_target = np.clip(suite.feedforward_from_prediction(pred_for_control, p), u_min, u_max)

        if control_id == "ADRC":
            disturbance_est = np.array(
                [
                    0.20 * x_est[6] / max(p.rollforce_scale, 1e-9),
                    -0.16 * x_est[7],
                    -0.18 * x_est[8],
                ],
                dtype=float,
            )
            u_target = np.clip(u_target + disturbance_est, u_min, u_max)
        elif control_id == "ADAPTIVE-MPC":
            adapt = 1.0 + 0.25 * np.tanh(np.linalg.norm(y_for_control / scales))
            u_target = np.clip(adapt * u_target, u_min, u_max)
        if flags["pid"]:
            u_target = np.clip(u_target + pid.step(-y_for_control, dt=float(effective_dt[k])), u_min, u_max)

        if not np.all(np.isfinite(u_target)):
            invalid_command_count += 1
            u_target = np.nan_to_num(u_target, nan=0.0, posinf=limit, neginf=-limit)
        solve_ms.append((time.perf_counter() - start) * 1000.0)

        legacy_step = np.array([0.065, 0.052, 0.065], dtype=float) if flags["mpc"] else np.array([0.090, 0.075, 0.090], dtype=float)
        legacy_step *= effective_move_scale if not flags.get("no_constraint") else 99.0
        rate_per_s = legacy_step / LEGACY_MOVE_LIMIT_REFERENCE_DT_S
        du_lim = np.minimum(rate_per_s * float(effective_dt[k]), legacy_step * 2.0)
        du = np.clip(u_target - u, -du_lim, du_lim)
        if np.any(np.abs(u_target - u) > du_lim + 1e-12):
            sat_count += 1
        u = np.clip(u + du, u_min, u_max)
        if not np.all(np.isfinite(u)):
            invalid_command_count += 1
            u = np.nan_to_num(u, nan=0.0, posinf=limit, neginf=-limit)
        phase = suite.phase_name(k, n, p)
        control_error = x[:3] - ref
        rows.append(
            {
                "k": k,
                "time_s": float(timebase.elapsed_s[k]),
                "effective_dt_s": float(effective_dt[k]),
                "effective_time_source": str(timebase.source[k]),
                "phase": phase,
                "scenario": scenario.scenario,
                "control": label_override or control_id,
                "base_control": control_id,
                "seed": seed,
                "reference_kind": scenario.reference_kind,
                "T_state": x[0],
                "h_state": x[1],
                "S_state": x[2],
                "T_ref": ref[0],
                "h_ref": ref[1],
                "S_ref": ref[2],
                "T_error": control_error[0],
                "h_error": control_error[1],
                "S_error": control_error[2],
                "T_control_error": control_error[0],
                "h_control_error": control_error[1],
                "S_control_error": control_error[2],
                "T_est": y[0],
                "h_est": y[1],
                "S_est": y[2],
                "u_speed": u[0],
                "u_gap": u[1],
                "u_shape": u[2],
                "du_speed": du[0],
                "du_gap": du[1],
                "du_shape": du[2],
                "residual_norm": residual,
                "mpc_active": int(mpc_active),
            }
        )
        if k < n - 1:
            x_next = suite.plant_step(x, u, p, k, n, np.zeros(5), np.zeros(6), allocation=True)
            x_next[:3] += scenario.residual_gain * innovations[k]
            x_next[:3] = np.clip(x_next[:3], -8.0 * scales, 8.0 * scales)
            x = x_next
            constraint_count += int(np.any(np.abs(suite.h_func(x)) > 3.0 * scales))
    diag = {
        "control": label_override or control_id,
        "base_control": control_id,
        "seed": seed,
        "scenario": scenario.scenario,
        "active_filter": active_filter,
        "sat_count": sat_count,
        "sat_ratio": float(sat_count / max(n, 1)),
        "constraint_count": int(constraint_count),
        "constraint_ratio": float(constraint_count / max(n - 1, 1)),
        "fallback_count": fallback_count,
        "invalid_command_count": invalid_command_count,
        "filter_ms_mean": float(np.mean(filter_ms)) if filter_ms else 0.0,
        "filter_ms_p95": float(np.percentile(filter_ms, 95)) if filter_ms else 0.0,
        "solve_ms_mean": float(np.mean(solve_ms)) if solve_ms else 0.0,
        "solve_ms_p95": float(np.percentile(solve_ms, 95)) if solve_ms else 0.0,
        "solve_ms_max": float(np.max(solve_ms)) if solve_ms else 0.0,
        "p95_compute_ms": float(np.percentile(np.asarray(filter_ms) + np.asarray(solve_ms), 95)) if filter_ms and solve_ms else 0.0,
        "residual_gain": scenario.residual_gain,
        "measurement_noise_level": scenario.measurement_noise_level,
        "move_limit_scale": scenario.move_limit_scale,
        "reference_kind": scenario.reference_kind,
        "predictor_override": scenario.predictor_override,
        "ablation": scenario.ablation,
    }
    return pd.DataFrame(rows), diag


def add_extra_metrics(metrics: dict[str, Any], df: pd.DataFrame, p, scenario: ScenarioSpec, diag: dict[str, Any]) -> dict[str, Any]:
    if df.empty:
        return metrics
    scale = scale_vec(p)
    err = df[["T_control_error", "h_control_error", "S_control_error"]].to_numpy(dtype=float)
    norm = err / scale
    u_cols = [c for c in ["u_speed", "u_gap", "u_shape"] if c in df.columns]
    u = df[u_cols].to_numpy(dtype=float) if u_cols else np.zeros((len(df), 3), dtype=float)
    du = supported_command_deltas(u, df["segment_id"].to_numpy() if "segment_id" in df else None)
    peak_action = float(np.max(np.abs(u))) if len(u) else 0.0
    delta_peak = float(np.max(np.abs(du)) / (peak_action + 1e-9)) if len(du) else 0.0
    h_abs = np.abs(err[:, 1])
    s_abs = np.abs(err[:, 2])
    metrics = dict(metrics)
    metrics["h_RMS"] = float(np.sqrt(np.mean(np.square(err[:, 1]))))
    metrics["S_RMS"] = float(np.sqrt(np.mean(np.square(err[:, 2]))))
    metrics["P95_abs_h"] = float(np.percentile(h_abs, 95))
    metrics["P95_abs_S"] = float(np.percentile(s_abs, 95))
    metrics["delta_u_over_peak_action"] = delta_peak
    metrics["invalid_command_count"] = int(diag.get("invalid_command_count", 0))
    metrics["fallback_count"] = int(diag.get("fallback_count", 0))
    metrics["p95_compute_ms"] = float(diag.get("p95_compute_ms", 0.0))
    if scenario.reference_kind != "none":
        start = int(0.45 * len(df))
        threshold = np.array([0.12, 0.12, 0.12])
        post = np.max(np.abs(norm[start:]), axis=1) if start < len(norm) else np.array([])
        rec = np.nan
        for i in range(len(post)):
            if np.all(post[i:] < np.max(threshold)):
                rec = float(i)
                break
        metrics["recovery_samples"] = rec if np.isfinite(rec) else float(len(post))
    else:
        metrics["recovery_samples"] = float("nan")
    return metrics


def suite_metrics(suite, rp: ReplayPass, df: pd.DataFrame, diag: dict[str, Any], base_control: str, scenario: ScenarioSpec) -> dict[str, Any]:
    metrics = suite.control_metrics(df, rp.p, diag, base_control, "S5", int(diag["seed"]))
    metrics["source"] = rp.source
    metrics["pass_id"] = rp.pass_id
    metrics["condition_id"] = rp.condition_id
    metrics["trial_id"] = rp.trial_id
    metrics["file"] = rp.file
    metrics["control"] = diag["control"]
    metrics["base_control"] = base_control
    metrics["scenario"] = scenario.scenario
    metrics["scenario_group"] = scenario.group
    metrics["seed"] = int(diag["seed"])
    metrics["residual_gain"] = scenario.residual_gain
    metrics["measurement_noise_level"] = scenario.measurement_noise_level
    metrics["move_limit_scale"] = scenario.move_limit_scale
    metrics["reference_kind"] = scenario.reference_kind
    metrics["predictor_override"] = scenario.predictor_override or ""
    metrics["ablation"] = scenario.ablation or ""
    metrics["p95_compute_ms"] = diag.get("p95_compute_ms", 0.0)
    metrics["invalid_command_count"] = diag.get("invalid_command_count", 0)
    metrics["fallback_count"] = diag.get("fallback_count", 0)
    return add_extra_metrics(metrics, df, rp.p, scenario, diag)


def run_recorded_metrics(suite, rp: ReplayPass, scenario_name: str = "nominal") -> dict[str, Any]:
    p = rp.p
    y = rp.y_real
    steps = len(y)
    timebase = archived_timebase(rp)
    rows = []
    for k in range(steps):
        phase = suite.phase_name(min(k, steps - 2), steps, p)
        rows.append(
            {
                "k": k,
                "time_s": float(timebase.elapsed_s[k]),
                "effective_dt_s": float(timebase.effective_dt_s[k]),
                "phase": phase,
                "T_error": y[k, 0],
                "h_error": y[k, 1],
                "S_error": y[k, 2],
                "u_gap": 0.0,
                "u_shape": 0.0,
                "u_speed": 0.0,
                "du_speed": 0.0,
                "du_gap": 0.0,
                "du_shape": 0.0,
                "T_control_error": y[k, 0],
                "h_control_error": y[k, 1],
                "S_control_error": y[k, 2],
            }
        )
    df = pd.DataFrame(rows)
    diag = {
        "control": "RECORDED",
        "seed": 0,
        "filter_active": False,
        "mpc_active": False,
        "active_filter": "recorded",
        "sat_count": 0,
        "sat_ratio": 0.0,
        "constraint_count": 0,
        "constraint_ratio": 0.0,
        "fallback_count": 0,
        "invalid_command_count": 0,
        "mean_compute_ms": 0.0,
        "p95_compute_ms": 0.0,
        "max_compute_ms": 0.0,
        "filter_ms_mean": 0.0,
        "filter_ms_p95": 0.0,
        "solve_ms_mean": 0.0,
        "solve_ms_p95": 0.0,
        "solve_ms_max": 0.0,
    }
    metrics = suite.control_metrics(df, p, diag, "C0", "S5", 0)
    metrics["source"] = rp.source
    metrics["pass_id"] = rp.pass_id
    metrics["condition_id"] = rp.condition_id
    metrics["trial_id"] = rp.trial_id
    metrics["file"] = rp.file
    metrics["control"] = "RECORDED"
    metrics["base_control"] = "RECORDED"
    metrics["scenario"] = scenario_name
    metrics["scenario_group"] = "recorded"
    return add_extra_metrics(metrics, df, p, ScenarioSpec(scenario_name, "recorded"), diag)


def build_quantile_envelope(suite, data1_passes: list[ReplayPass], out_dir: Path) -> pd.DataFrame:
    rows = []
    for rp in data1_passes:
        n = len(rp.y_real)
        for k, y in enumerate(rp.y_real):
            phase = suite.phase_name(min(k, n - 2), n, rp.p)
            z = y / scale_vec(rp.p)
            rows.append({"phase": phase, "T_norm": z[0], "h_norm": z[1], "S_norm": z[2]})
    raw = pd.DataFrame(rows)
    env_rows = []
    for phase, g in raw.groupby("phase", sort=True):
        entry = {"phase": phase}
        for col in ["T_norm", "h_norm", "S_norm"]:
            entry[f"{col}_lo"] = float(g[col].quantile(0.025))
            entry[f"{col}_hi"] = float(g[col].quantile(0.975))
            entry[f"{col}_median"] = float(g[col].median())
        env_rows.append(entry)
    env = pd.DataFrame(env_rows)
    env.to_csv(out_dir / "data1_stage_quantile_envelope_p025_p975.csv", index=False, encoding="utf-8-sig")
    return env


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def tail_mean(values: list[float], quantile: float = 0.95) -> float:
    """Empirical upper-tail mean of the largest ceil((1-q)*N) finite samples.

    The supported range is 0 <= q < 1. The declared decimal q is interpreted
    exactly for the sample count, without fractional boundary weighting.
    Ties or a zero empirical quantile do not expand the selected sample set.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    k = upper_tail_count(arr.size, quantile)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.partition(arr, -k)[-k:]))


def envelope_metrics_for_curve(suite, rp: ReplayPass, df: pd.DataFrame, env: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "S_out": float("nan"),
            "h_out": float("nan"),
            "T_out": float("nan"),
            "outside_any": float("nan"),
            "longest_outside_samples": float("nan"),
            "S_excess_mean": float("nan"),
            "S_excess_p95": float("nan"),
            "S_excess_cvar95": float("nan"),
        }
    env_map = {r["phase"]: r for _, r in env.iterrows()}
    out_any = []
    out_t = []
    out_h = []
    out_s = []
    h_abs = []
    s_abs = []
    t_excess = []
    h_excess = []
    s_excess = []
    for _, row in df.iterrows():
        phase = row.get("phase", "steady")
        erow = env_map.get(phase)
        if erow is None:
            erow = env_map.get("steady")
        if erow is None:
            erow = next(iter(env_map.values()))
        vals = np.array(
            [
                row.get("T_control_error", row.get("T_error", 0.0)) / scale_vec(rp.p)[0],
                row.get("h_control_error", row.get("h_error", 0.0)) / scale_vec(rp.p)[1],
                row.get("S_control_error", row.get("S_error", 0.0)) / scale_vec(rp.p)[2],
            ],
            dtype=float,
        )
        t_o = vals[0] < erow["T_norm_lo"] or vals[0] > erow["T_norm_hi"]
        h_o = vals[1] < erow["h_norm_lo"] or vals[1] > erow["h_norm_hi"]
        s_o = vals[2] < erow["S_norm_lo"] or vals[2] > erow["S_norm_hi"]
        out_t.append(t_o)
        out_h.append(h_o)
        out_s.append(s_o)
        out_any.append(t_o or h_o or s_o)
        h_abs.append(abs(vals[1]))
        s_abs.append(abs(vals[2]))
        t_excess.append(max(float(erow["T_norm_lo"] - vals[0]), 0.0, float(vals[0] - erow["T_norm_hi"])))
        h_excess.append(max(float(erow["h_norm_lo"] - vals[1]), 0.0, float(vals[1] - erow["h_norm_hi"])))
        s_excess.append(max(float(erow["S_norm_lo"] - vals[2]), 0.0, float(vals[2] - erow["S_norm_hi"])))
    mask_any = np.array(out_any, dtype=bool)
    return {
        "T_out": float(np.mean(out_t)),
        "h_out": float(np.mean(out_h)),
        "S_out": float(np.mean(out_s)),
        "outside_any": float(np.mean(mask_any)),
        "longest_outside_samples": float(longest_true_run(mask_any)),
        "P95_abs_h_norm": float(np.percentile(h_abs, 95)),
        "P95_abs_S_norm": float(np.percentile(s_abs, 95)),
        "T_excess_mean": float(np.mean(t_excess)),
        "h_excess_mean": float(np.mean(h_excess)),
        "S_excess_mean": float(np.mean(s_excess)),
        "T_excess_p95": float(np.percentile(t_excess, 95)),
        "h_excess_p95": float(np.percentile(h_excess, 95)),
        "S_excess_p95": float(np.percentile(s_excess, 95)),
        "S_excess_cvar95": tail_mean(s_excess, 0.95),
    }


def run_scenario_set(
    # Auxiliary scenario path (see run_replay_control_protocol). Pass-only
    # seeds are used here as well.
    suite,
    data2_passes: list[ReplayPass],
    controls: list[tuple[str, str, PidParams | None]],
    scenarios: list[ScenarioSpec],
    repeats: int,
    envelope: pd.DataFrame,
    out_dir: Path,
    save_curves: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    curves_dir = ensure_dir(out_dir / "curves") if save_curves else None
    total = len(scenarios) * len(data2_passes) * len(controls) * repeats
    t0 = time.time()
    run_i = 0
    for scenario in scenarios:
        for rp in data2_passes:
            for base_control, label, pid_params in controls:
                for rep in range(repeats):
                    run_i += 1
                    # Pass-only seed: every control shares the same random
                    # realization for a given pass and scenario.
                    seed = RNG_SEED + rep + 1000 * (data2_passes.index(rp) + 1)
                    if run_i == 1 or run_i == total or run_i % 50 == 0:
                        print(
                            f"[replay] {run_i}/{total} scenario={scenario.scenario} "
                            f"pass={rp.pass_id} control={label} elapsed={elapsed_s(t0)}",
                            flush=True,
                        )
                    df, diag = run_replay_control_protocol(
                        suite,
                        rp,
                        base_control,
                        seed,
                        scenario,
                        locked_pid=pid_params,
                        label_override=label,
                    )
                    metrics = suite_metrics(suite, rp, df, diag, base_control, scenario)
                    metrics.update(envelope_metrics_for_curve(suite, rp, df, envelope))
                    rows.append(metrics)
                    if save_curves and curves_dir is not None and rep == 0:
                        safe_name = f"{scenario.scenario}__{rp.pass_id}__{label}.csv".replace("/", "_")
                        df.to_csv(curves_dir / safe_name, index=False, encoding="utf-8-sig")
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    return result


def aggregate_metrics(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metric_cols = [
        "composite_normalized_RMS",
        "control_total_variation",
        "control_delta_rms",
        "startup_motion",
        "restart_motion",
        "startup_restart_motion",
        "S_out",
        "h_out",
        "outside_any",
        "longest_outside_samples",
        "h_RMS",
        "S_RMS",
        "P95_abs_h",
        "P95_abs_S",
        "delta_u_over_peak_action",
        "p95_compute_ms",
        "invalid_command_count",
        "fallback_count",
        "recovery_samples",
    ]
    usable = [c for c in metric_cols if c in df.columns]
    agg = df.groupby(keys, dropna=False)[usable].agg(["mean", "std", "median"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col for col in agg.columns]
    return agg


def paired_effects(
    df: pd.DataFrame,
    scenario: str,
    treatment: str,
    baseline: str,
    metric: str,
    out_dir: Path,
    n_boot: int = 2000,
) -> dict[str, Any]:
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    subset = df[df["scenario"].eq(scenario)].copy()
    grouped = (
        subset.groupby(["pass_id", "seed", "control"], dropna=False)[metric]
        .mean()
        .reset_index()
        .pivot_table(index=["pass_id", "seed"], columns="control", values=metric, aggfunc="mean")
    )
    if treatment not in grouped.columns or baseline not in grouped.columns:
        return {"scenario": scenario, "treatment": treatment, "baseline": baseline, "metric": metric, "n": 0}
    paired = grouped[[baseline, treatment]].dropna()
    if paired.empty:
        return {"scenario": scenario, "treatment": treatment, "baseline": baseline, "metric": metric, "n": 0}
    diff = paired[treatment].to_numpy() - paired[baseline].to_numpy()
    rel_reduction = (paired[baseline].to_numpy() - paired[treatment].to_numpy()) / (np.abs(paired[baseline].to_numpy()) + 1e-12)
    rng = np.random.default_rng(RNG_SEED)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rel_reduction), size=len(rel_reduction))
        boots.append(float(np.median(rel_reduction[idx])))
    # Cliff's delta for treatment-vs-baseline values; negative favors lower treatment.
    a = paired[treatment].to_numpy()
    b = paired[baseline].to_numpy()
    greater = sum(float(x > y) for x in a for y in b)
    lesser = sum(float(x < y) for x in a for y in b)
    cliffs_delta = (greater - lesser) / (len(a) * len(b))
    p_value = float("nan")
    if wilcoxon is not None and len(diff) >= 2 and not np.allclose(diff, 0):
        try:
            p_value = float(wilcoxon(diff, zero_method="wilcox").pvalue)
        except Exception:
            p_value = float("nan")
    payload = {
        "scenario": scenario,
        "treatment": treatment,
        "baseline": baseline,
        "metric": metric,
        "n_pairs": int(len(paired)),
        "baseline_median": float(np.median(b)),
        "treatment_median": float(np.median(a)),
        "median_relative_reduction": float(np.median(rel_reduction)),
        "bootstrap_ci95_lo": float(np.percentile(boots, 2.5)),
        "bootstrap_ci95_hi": float(np.percentile(boots, 97.5)),
        "wilcoxon_p": p_value,
        "cliffs_delta": float(cliffs_delta),
    }
    return payload


def holm_correct(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = [(i, r.get("wilcoxon_p", float("nan"))) for i, r in enumerate(rows) if np.isfinite(r.get("wilcoxon_p", float("nan")))]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    adjusted = [float("nan")] * len(rows)
    running = 0.0
    for rank, (idx, pval) in enumerate(indexed, start=1):
        adj = min(1.0, (m - rank + 1) * pval)
        running = max(running, adj)
        adjusted[idx] = running
    for i, r in enumerate(rows):
        r["holm_p"] = adjusted[i]
    return rows


def make_figures(nominal: pd.DataFrame, stress: pd.DataFrame, ablation: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ensure_dir(out_dir / "figures")
    paths: list[Path] = []
    if not nominal.empty:
        agg = nominal.groupby("control", dropna=False).agg(
            RMS_c=("composite_normalized_RMS", "mean"),
            TV=("control_total_variation", "mean"),
            S_out=("S_out", "mean"),
            p95_ms=("p95_compute_ms", "mean"),
        )
        fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=160)
        sizes = 80 + 120 * np.sqrt(np.clip(agg["p95_ms"].fillna(0), 0, None) + 1)
        colors = np.where(agg["S_out"] <= 0.10, "#247B5E", "#B84A3A")
        ax.scatter(agg["TV"], agg["RMS_c"], s=sizes, c=colors, edgecolor="black", linewidth=0.7, alpha=0.9)
        for control, row in agg.iterrows():
            ax.annotate(control, (row["TV"], row["RMS_c"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Control total variation")
        ax.set_ylabel("Composite normalized RMS")
        ax.set_title("Pareto view: RMS vs TV (Data2 nominal replay)")
        ax.grid(True, alpha=0.25)
        path = fig_dir / "pareto_rms_vs_tv.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=160)
        ax.scatter(agg["TV"], agg["S_out"], s=sizes, c=colors, edgecolor="black", linewidth=0.7, alpha=0.9)
        for control, row in agg.iterrows():
            ax.annotate(control, (row["TV"], row["S_out"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.axhline(0.10, color="#777777", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Control total variation")
        ax.set_ylabel("Flatness outside-envelope ratio")
        ax.set_title("Pareto view: envelope risk vs TV (Data2 nominal replay)")
        ax.grid(True, alpha=0.25)
        path = fig_dir / "pareto_sout_vs_tv.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    if not stress.empty:
        piv = stress.pivot_table(
            index="scenario",
            columns="control",
            values="S_out",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(10.0, max(4.0, 0.35 * len(piv))), dpi=160)
        im = ax.imshow(piv.fillna(0).to_numpy(), cmap="YlOrRd", aspect="auto")
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=7)
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=35, ha="right", fontsize=8)
        ax.set_title("Stress suite: mean S_out")
        fig.colorbar(im, ax=ax, label="S_out")
        path = fig_dir / "stress_sout_heatmap.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    if not ablation.empty:
        agg = (
            ablation.groupby("control", dropna=False)
            .agg(RMS_c=("composite_normalized_RMS", "mean"), S_out=("S_out", "mean"), TV=("control_total_variation", "mean"))
            .sort_values("S_out")
        )
        fig, ax1 = plt.subplots(figsize=(9.2, 5.0), dpi=160)
        x = np.arange(len(agg))
        ax1.bar(x - 0.18, agg["RMS_c"], width=0.36, label="RMS_c", color="#3E6FA8")
        ax2 = ax1.twinx()
        ax2.bar(x + 0.18, agg["S_out"], width=0.36, label="S_out", color="#C66A45")
        ax1.set_xticks(x)
        ax1.set_xticklabels(agg.index, rotation=35, ha="right", fontsize=8)
        ax1.set_ylabel("Composite normalized RMS")
        ax2.set_ylabel("Flatness outside-envelope ratio")
        ax1.set_title("C7 ablation summary")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
        path = fig_dir / "c7_ablation_summary.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(
    out_dir: Path,
    manifest: dict[str, Any],
    locked_pid: PidParams,
    nominal: pd.DataFrame,
    stress: pd.DataFrame,
    ablation: pd.DataFrame,
    stats_rows: list[dict[str, Any]],
    figure_paths: list[Path],
) -> Path:
    nominal_agg = aggregate_metrics(nominal, ["control"]) if not nominal.empty else pd.DataFrame()
    stress_agg = aggregate_metrics(stress, ["scenario", "control"]) if not stress.empty else pd.DataFrame()
    ablation_agg = aggregate_metrics(ablation, ["control"]) if not ablation.empty else pd.DataFrame()
    nominal_agg.to_csv(out_dir / "nominal_summary_mean_std_median.csv", index=False, encoding="utf-8-sig")
    stress_agg.to_csv(out_dir / "stress_summary_mean_std_median.csv", index=False, encoding="utf-8-sig")
    ablation_agg.to_csv(out_dir / "ablation_summary_mean_std_median.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stats_rows).to_csv(out_dir / "paired_statistics.csv", index=False, encoding="utf-8-sig")

    def table_html(df: pd.DataFrame, cols: list[str] | None = None, n: int = 40) -> str:
        if df.empty:
            return "<p>No rows.</p>"
        use = df.copy()
        if cols:
            use = use[[c for c in cols if c in use.columns]]
        return use.head(n).to_html(index=False, escape=False, float_format=lambda x: f"{x:.4g}")

    figure_html = "\n".join(
        f'<figure><img src="{p.relative_to(out_dir).as_posix()}" alt="{p.name}" style="max-width:100%;"><figcaption>{p.name}</figcaption></figure>'
        for p in figure_paths
    )

    protocol_issue = ""
    if manifest["data"]["data1_count"] != EXPECTED_DATA1_PASSES or manifest["data"]["data2_count"] != EXPECTED_DATA2_PASSES:
        protocol_issue = (
            "<p><strong>Data-basis note.</strong> The protocol table lists "
            f"{EXPECTED_DATA1_PASSES} Data1 passes and {EXPECTED_DATA2_PASSES} Data2 passes, "
            "the supplied data directories are the locked basis for this run. "
            f"This run therefore uses all current files: {manifest['data']['data1_count']} Data1 CSV files and "
            f"{manifest['data']['data2_count']} Data2 CSV files.</p>"
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>C7 archived-update protocol report</title>
<style>
body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #1f2933; }}
h1, h2 {{ color: #172554; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #d6dde8; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #edf2fa; }}
.note {{ background: #fff7ed; border-left: 4px solid #c66a45; padding: 10px 14px; margin: 16px 0; }}
figure {{ margin: 22px 0; }}
figcaption {{ color: #52606d; font-size: 12px; }}
code {{ background: #f3f4f6; padding: 1px 4px; }}
</style>
</head>
<body>
<h1>C7 archived-update protocol report</h1>
<p>Generated at {manifest['generated_at']}. This run follows the locked Data1-to-Data2 protocol: PID tuning is selected on Data1 only, and Data2 is used only for independent replay evaluation.</p>
<div class="note">
{protocol_issue}
<p><strong>Locked LOPO PID:</strong> {locked_pid.key}. Selection criterion: Data1-LOPO mean composite normalized RMS only.</p>
</div>

<h2>Primary Nominal Results</h2>
{table_html(nominal_agg, n=30)}

<h2>Paired Statistics</h2>
{table_html(pd.DataFrame(stats_rows), n=80)}

<h2>Stress Suite</h2>
{table_html(stress_agg, n=80)}

<h2>C7 Ablations</h2>
{table_html(ablation_agg, n=40)}

<h2>Figures</h2>
{figure_html}

<h2>Audit Files</h2>
<p>Manifest: <code>protocol_manifest.json</code>. PID grid: <code>pid_full_grid.csv</code>. Raw metric files are stored under <code>nominal/</code>, <code>stress/</code>, and <code>ablation/</code>.</p>
</body>
</html>"""
    path = out_dir / "protocol_report.html"
    path.write_text(html, encoding="utf-8")
    return path


def build_scenarios(skip_stress: bool, skip_ablation: bool) -> tuple[list[ScenarioSpec], list[ScenarioSpec], list[ScenarioSpec]]:
    nominal = [ScenarioSpec("nominal_rg082", "nominal", residual_gain=0.82)]
    stress: list[ScenarioSpec] = []
    if not skip_stress:
        stress.extend(
            [
                ScenarioSpec("residual_gain_082", "residual_gain", residual_gain=0.82),
                ScenarioSpec("residual_gain_100", "residual_gain", residual_gain=1.00),
                ScenarioSpec("residual_gain_120", "residual_gain", residual_gain=1.20),
                ScenarioSpec("noise_mild_1pct", "noise", measurement_noise_level=0.01),
                ScenarioSpec("noise_medium_3pct", "noise", measurement_noise_level=0.03),
                ScenarioSpec("noise_high_5pct", "noise", measurement_noise_level=0.05),
                ScenarioSpec("move_limit_nominal_100", "move_limit", move_limit_scale=1.00),
                ScenarioSpec("move_limit_medium_075", "move_limit", move_limit_scale=0.75),
                ScenarioSpec("move_limit_tight_060", "move_limit", move_limit_scale=0.60),
                ScenarioSpec("reference_h_step", "reference_step", reference_kind="h_step"),
                ScenarioSpec("reference_S_step", "reference_step", reference_kind="S_step"),
                ScenarioSpec("predictor_mismatch_mild_dmc", "predictor_mismatch", predictor_override="dmc"),
                ScenarioSpec("predictor_mismatch_severe_persistence", "predictor_mismatch", predictor_override="persistence"),
                ScenarioSpec("predictor_mismatch_no_gate", "predictor_mismatch", ablation="C7-NoGate"),
            ]
        )
    ablations: list[ScenarioSpec] = []
    if not skip_ablation:
        for name in [
            "C7-Full",
            "C7-NoGate",
            "C7-NoStage",
            "C7-NoAlloc",
            "C7-NoFilter",
            "C7-NoPID",
            "C7-NoConstraint",
            "C7-TightConstraint",
        ]:
            ablations.append(ScenarioSpec(f"ablation_{name}", "ablation", ablation=name))
    return nominal, stress, ablations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--max-pid-candidates", type=int, default=None)
    parser.add_argument("--data2-pass-limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=DATA2_REPEATS_REQUIRED)
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--save-curves", action="store_true")
    args = parser.parse_args()

    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path(args.out_root) if args.out_root else ROOT / f"protocol_run_{timestamp}")
    ensure_dir(out_dir / "nominal")
    ensure_dir(out_dir / "stress")
    ensure_dir(out_dir / "ablation")
    ensure_dir(out_dir / "pid_data1_grid")
    ensure_dir(out_dir / "audit")

    suite, replay = load_suite_modules()
    data1_passes = load_passes(suite, replay, DATA1_DIR, "Data1")
    data2_passes = load_passes(suite, replay, DATA2_DIR, "Data2")
    if args.data2_pass_limit is not None:
        data2_passes = data2_passes[: args.data2_pass_limit]

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(out_dir),
        "project_dir": str(PROJECT_DIR),
        "protocol": protocol_paths(),
        "seed": RNG_SEED,
        "data": {
            "data1_dir": str(DATA1_DIR),
            "data2_dir": str(DATA2_DIR),
            "data1_count": len(data1_passes),
            "data2_count": len(data2_passes),
            "data1_condition_count": len({rp.condition_id for rp in data1_passes}),
            "data2_condition_count": len({rp.condition_id for rp in data2_passes}),
            "data1_files": [rp.file for rp in data1_passes],
            "data2_files": [rp.file for rp in data2_passes],
            "data1_conditions": sorted({rp.condition_id for rp in data1_passes}),
            "data2_conditions": sorted({rp.condition_id for rp in data2_passes}),
            "data2_condition_trials": {
                cid: [rp.file for rp in data2_passes if rp.condition_id == cid]
                for cid in sorted({rp.condition_id for rp in data2_passes})
            },
            "expected_data1_count": EXPECTED_DATA1_PASSES,
            "expected_data2_count": EXPECTED_DATA2_PASSES,
            "partition_counts_verified": True,
            "used_all_current_files": args.data2_pass_limit is None,
        },
        "scale_lock": scale_lock_manifest(),
        "requirements": {
            "timebase": "archived_update_rowwise_effective_dt",
            "archived_update_horizon": ARCHIVED_UPDATE_HORIZON,
            "legacy_move_limit_reference_dt_s": LEGACY_MOVE_LIMIT_REFERENCE_DT_S,
            "data2_repeats": args.repeats,
            "pid_selection": (
                "complete 1050-candidate grid through the canonical PID replay; "
                "channelwise normalized RMS values are averaged within pass, then "
                "the 13 Data1 pass scores receive equal weight"
            ),
            "main_replay": "measured residual replay, same residual sequence across controllers",
            "primary_metrics": ["composite_normalized_RMS", "S_out", "control_total_variation"],
            "data2_reporting": "raw file-weighted metrics are preserved; postprocess also reports condition-weighted metrics by thickness-condition id",
        },
    }
    write_json(out_dir / "protocol_manifest.json", manifest)

    print(f"[setup] output={out_dir}", flush=True)
    print(f"[setup] Data1 passes={len(data1_passes)} Data2 passes={len(data2_passes)} repeats={args.repeats}", flush=True)
    if len(data1_passes) != EXPECTED_DATA1_PASSES or len(data2_passes) != EXPECTED_DATA2_PASSES:
        pd.DataFrame(
            [
                {
                    "item": "pass_count",
                    "expected_data1": EXPECTED_DATA1_PASSES,
                    "actual_data1": len(data1_passes),
                    "expected_data2": EXPECTED_DATA2_PASSES,
                    "actual_data2": len(data2_passes),
                    "decision": "partition_count_mismatch_requires_explicit review",
                }
            ]
        ).to_csv(out_dir / "audit" / "protocol_count_deviation.csv", index=False, encoding="utf-8-sig")

    envelope = build_quantile_envelope(suite, data1_passes, out_dir / "audit")
    locked_pid = select_data1_pid_legacy_disabled(
        suite, data1_passes, out_dir / "pid_data1_grid",
        max_candidates=args.max_pid_candidates,
    )

    nominal_scenarios, stress_scenarios, ablation_scenarios = build_scenarios(args.skip_stress, args.skip_ablation)
    nominal_controls: list[tuple[str, str, PidParams | None]] = [
        ("C0", "PID-nominal", None),
        ("C0", "PID-Data1Grid", locked_pid),
        ("C2", "C2", None),
        ("ADRC", "ADRC", None),
        ("LSTM-MPC", "LSTM-MPC", None),
        ("DMC-KF", "DMC-KF", None),
        ("C7", "C7", None),
    ]
    stress_controls: list[tuple[str, str, PidParams | None]] = [
        ("C0", "PID-Data1Grid", locked_pid),
        ("C2", "C2", None),
        ("ADRC", "ADRC", None),
        ("C7", "C7", None),
    ]
    ablation_controls: list[tuple[str, str, PidParams | None]] = [
        ("C7", "C7-Full", None),
        ("C7", "C7-NoGate", None),
        ("C7", "C7-NoStage", None),
        ("C7", "C7-NoAlloc", None),
        ("C7", "C7-NoFilter", None),
        ("C7", "C7-NoPID", None),
        ("C7", "C7-NoConstraint", None),
        ("C7", "C7-TightConstraint", None),
        ("C0", "PID-Data1Grid", locked_pid),
    ]

    nominal = run_scenario_set(
        suite,
        data2_passes,
        nominal_controls,
        nominal_scenarios,
        args.repeats,
        envelope,
        out_dir / "nominal",
        save_curves=args.save_curves,
    )
    recorded_rows = [run_recorded_metrics(suite, rp, "nominal_rg082") for rp in data2_passes]
    nominal = pd.concat([pd.DataFrame(recorded_rows), nominal], ignore_index=True, sort=False)
    nominal.to_csv(out_dir / "nominal" / "metrics_with_recorded.csv", index=False, encoding="utf-8-sig")

    stress = pd.DataFrame()
    if stress_scenarios:
        stress = run_scenario_set(
            suite,
            data2_passes,
            stress_controls,
            stress_scenarios,
            args.repeats,
            envelope,
            out_dir / "stress",
            save_curves=False,
        )

    ablation = pd.DataFrame()
    if ablation_scenarios:
        # Each ablation scenario is evaluated with the same-labeled control.
        rows = []
        total = len(ablation_scenarios) * len(data2_passes) * args.repeats
        t0 = time.time()
        run_i = 0
        for scenario in ablation_scenarios:
            label = scenario.ablation or "C7-Full"
            control_tuple = next((x for x in ablation_controls if x[1] == label), ("C7", label, None))
            for rp in data2_passes:
                for rep in range(args.repeats):
                    run_i += 1
                    if run_i == 1 or run_i == total or run_i % 50 == 0:
                        print(f"[ablation] {run_i}/{total} {label} {rp.pass_id} elapsed={elapsed_s(t0)}", flush=True)
                    # Pass-only seed: every ablation label shares the same
                    # noise realization for a given pass, aligned with the
                    # publication replay.
                    seed = RNG_SEED + rep + 2000 * (data2_passes.index(rp) + 1)
                    df, diag = run_replay_control_protocol(
                        suite,
                        rp,
                        control_tuple[0],
                        seed,
                        scenario,
                        locked_pid=control_tuple[2],
                        label_override=label,
                    )
                    metrics = suite_metrics(suite, rp, df, diag, control_tuple[0], scenario)
                    metrics.update(envelope_metrics_for_curve(suite, rp, df, envelope))
                    rows.append(metrics)
        # Add PID-only to the ablation table under nominal conditions.
        pid_scenario = ScenarioSpec("ablation_PID-only", "ablation")
        for rp in data2_passes:
            for rep in range(args.repeats):
                seed = RNG_SEED + rep + 3000 * (data2_passes.index(rp) + 1)
                df, diag = run_replay_control_protocol(
                    suite, rp, "C0", seed, pid_scenario, locked_pid=locked_pid, label_override="PID-only"
                )
                metrics = suite_metrics(suite, rp, df, diag, "C0", pid_scenario)
                metrics.update(envelope_metrics_for_curve(suite, rp, df, envelope))
                rows.append(metrics)
        ablation = pd.DataFrame(rows)
        ablation.to_csv(out_dir / "ablation" / "metrics.csv", index=False, encoding="utf-8-sig")

    stats_rows = []
    for metric in ["composite_normalized_RMS", "S_out", "control_total_variation"]:
        for baseline in ["PID-Data1Grid", "C2", "ADRC"]:
            stats_rows.append(paired_effects(nominal, "nominal_rg082", "C7", baseline, metric, out_dir))
    stats_rows = holm_correct(stats_rows)
    write_json(out_dir / "paired_statistics.json", stats_rows)

    figures = make_figures(nominal, stress, ablation, out_dir)
    report = write_report(out_dir, manifest, locked_pid, nominal, stress, ablation, stats_rows, figures)
    print(f"[done] report={report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "The legacy monolithic protocol CLI is disabled. Use code/run_pipeline.py "
        "for the released workflow or code/run_pid_selection_audit.py for the "
        "canonical Data1 PID grid."
    )
