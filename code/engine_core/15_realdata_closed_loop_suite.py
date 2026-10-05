from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.controllers import PID
from src.real_process_data import load_real_process_data

# The shared metric also remains available when this legacy entry point is run
# directly from engine_core rather than loaded by the top-level protocol.
_METRICS_DIR = str(Path(__file__).resolve().parent.parent)
if _METRICS_DIR not in sys.path:
    sys.path.insert(0, _METRICS_DIR)
from supported_command_motion import reset_command_motion, supported_command_deltas, supported_total_variation


CONTROL_LABELS = {
    "C0": "PID",
    "C1": "PID + selected nonlinear filter",
    "C2": "Prediction feedforward + PID",
    "C3": "Filter + prediction feedforward + PID",
    "C4": "Filter + predictor + MPC",
    "C5": "Full control, fixed weights",
    "C6": "Full control, no channel allocation",
    "C7": "Full proposed collaborative control",
    "LSTM-MPC": "LSTM-MPC",
    "DMC-KF": "DMC-KF",
    "ATT-MPC": "ATT-MPC",
    "ADRC": "Observer-compensated PID (OC-PID)",
    "ROBUST-MPC": "Robust MPC",
    "ADAPTIVE-MPC": "Adaptive MPC",
}

SCENARIO_LABELS = {
    "S1": "Small steady disturbance",
    "S2": "Tension step disturbance",
    "S3": "Slow drift",
    "S4": "High noise with outliers",
    "S5": "Coupled multi-variable disturbance",
    "S6": "Actuator constrained coupled disturbance",
}

FILTER_LABELS = {
    "raw": "Raw measurement",
    "ekf": "Extended Kalman filter",
    "ukf": "Unscented Kalman filter",
    "pf": "Particle filter",
    "grid_bayes": "Recursive Bayesian grid filter",
}

OUTPUT_COLS = ["T_error", "h_error", "S_error"]
SAMPLE_PERIOD_S = 0.50


@dataclass
class RealPass:
    pass_file: str
    pass_index: int
    entry_thickness: float
    exit_thickness: float
    reduction_ratio: float
    rows: int
    tension_scale: float
    thickness_scale: float
    flatness_scale: float
    rollforce_scale: float
    radial_scale: float
    offcenter_scale: float
    speed_mean: float
    speed_span: float
    phase_window_ratio: float
    scale_source: str = "per-pass observed response"
    scale_lock_id: str = "none"
    scale_lock_sha256: str = "none"
    scale_match_pass_id: str = "self"
    scale_match_log_distance: float = 0.0


def robust_scale(values: np.ndarray, floor: float) -> float:
    values = np.asarray(values, dtype=float)
    q25, q75 = np.percentile(values, [25, 75])
    iqr = max(float(q75 - q25) / 1.349, float(np.std(values)), floor)
    return float(max(iqr, floor))


def make_real_pass(
    df: pd.DataFrame,
    scale_override: dict[str, float] | None = None,
    scale_provenance: dict[str, object] | None = None,
    allow_observed_scale_fit: bool = False,
) -> RealPass:
    n = len(df)
    speed = df["speed_avg"].to_numpy(dtype=float)
    if (scale_override is None) != (scale_provenance is None):
        raise ValueError("scale_override and scale_provenance must be supplied together")
    if scale_override is None:
        if not allow_observed_scale_fit:
            raise RuntimeError(
                "Per-pass observed-response scale fitting is disabled. "
                "Use the frozen Data1 scale override; only build_scale_lock.py "
                "may opt in while constructing the Data1 catalogue."
            )
        scales = {
            "tension_scale": robust_scale(df["exit_tension_error"].to_numpy(), 0.025) * 3.0,
            "thickness_scale": robust_scale(df["exit_thickness_dev"].to_numpy(), 0.30) * 1.35,
            "flatness_scale": robust_scale(df["flatness_std"].to_numpy(), 0.15) * 1.50,
            "rollforce_scale": robust_scale(df["rollforce_actual"].to_numpy(), 25.0),
            "radial_scale": robust_scale(df["radialforce_diff"].to_numpy(), 10.0),
            "offcenter_scale": robust_scale(df["offcenter_diff"].to_numpy(), 0.02),
        }
    else:
        required = {
            "tension_scale", "thickness_scale", "flatness_scale",
            "rollforce_scale", "radial_scale", "offcenter_scale",
        }
        missing = required.difference(scale_override)
        if missing:
            raise ValueError(f"Missing locked pass scales: {sorted(missing)}")
        scales = {key: float(scale_override[key]) for key in required}
        if any((not np.isfinite(value)) or value <= 0 for value in scales.values()):
            raise ValueError("Locked pass scales must be finite and positive")
    provenance = dict(scale_provenance or {})
    return RealPass(
        pass_file=str(df["pass_file"].iloc[0]),
        pass_index=int(df["pass_index"].iloc[0]),
        entry_thickness=float(df["entry_nominal_from_file"].iloc[0]),
        exit_thickness=float(df["exit_nominal_from_file"].iloc[0]),
        reduction_ratio=float(df["reduction_ratio"].mean()),
        rows=n,
        tension_scale=scales["tension_scale"],
        thickness_scale=scales["thickness_scale"],
        flatness_scale=scales["flatness_scale"],
        rollforce_scale=scales["rollforce_scale"],
        radial_scale=scales["radial_scale"],
        offcenter_scale=scales["offcenter_scale"],
        speed_mean=float(np.mean(speed)),
        speed_span=float(max(np.percentile(speed, 95) - np.percentile(speed, 5), 1.0)),
        phase_window_ratio=float(min(40, max(1, n // 4)) / max(n, 1)),
        scale_source=str(provenance.get("scale_source", "per-pass observed response")),
        scale_lock_id=str(provenance.get("scale_lock_id", "none")),
        scale_lock_sha256=str(provenance.get("scale_lock_sha256", "none")),
        scale_match_pass_id=str(provenance.get("scale_match_pass_id", "self")),
        scale_match_log_distance=float(provenance.get("scale_match_log_distance", 0.0)),
    )


def representative_passes(passes: list[RealPass], mode: str, limit: int) -> list[RealPass]:
    ordered = sorted(passes, key=lambda p: p.exit_thickness)
    if mode == "representative":
        idxs = sorted({0, len(ordered) // 2, len(ordered) - 1})
        chosen = [ordered[i] for i in idxs]
    else:
        chosen = ordered
    if limit > 0:
        chosen = chosen[:limit]
    return chosen


def phase_weights(k: int, steps: int, p: RealPass) -> np.ndarray:
    r = k / max(steps - 1, 1)
    w = max(p.phase_window_ratio, 40.0 / max(p.rows, 80))
    if r < w:
        return np.array([1.0, 0.0, 0.0])
    if r > 1.0 - w:
        return np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 1.0, 0.0])


def phase_name(k: int, steps: int, p: RealPass) -> str:
    weights = phase_weights(k, steps, p)
    return ["accel", "steady", "decel"][int(np.argmax(weights))]


def make_noise_profile(scenario: str, steps: int, seed: int, p: RealPass) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scales = np.array([p.tension_scale, p.thickness_scale, p.flatness_scale])
    proc_scale = np.array(
        [
            0.030 * p.tension_scale,
            0.030 * p.thickness_scale,
            0.030 * p.flatness_scale,
            0.020 * p.rollforce_scale,
            0.025,
            0.025,
        ]
    )
    meas_scale = np.array([0.060, 0.055, 0.065]) * scales
    if scenario == "S4":
        meas_scale *= 3.0
        proc_scale[:3] *= 1.35
    if scenario in {"S5", "S6"}:
        proc_scale *= 1.25

    proc = rng.normal(0.0, proc_scale, size=(steps, 6))
    meas = rng.normal(0.0, meas_scale, size=(steps, 3))
    if scenario == "S4":
        outlier_count = max(5, steps // 24)
        idx = rng.choice(steps, outlier_count, replace=False)
        meas[idx] += rng.normal(0.0, np.array([0.70, 0.65, 0.75]) * scales, size=(outlier_count, 3))

    ext = np.zeros((steps, 5), dtype=float)
    if scenario == "S1":
        ext[:, :3] += 0.010 * scales.reshape(1, -1) * np.sin(np.linspace(0, 4 * np.pi, steps)).reshape(-1, 1)
    elif scenario == "S2":
        start, end = int(0.30 * steps), int(0.56 * steps)
        ext[start:end, 0] += 0.42 * p.tension_scale
        ext[start:end, 2] += 0.08 * p.flatness_scale
    elif scenario == "S3":
        ramp = np.linspace(0.0, 1.0, steps)
        ext[:, 1] += 0.30 * p.thickness_scale * ramp
        ext[:, 2] += 0.22 * p.flatness_scale * ramp
        ext[:, 4] += 0.10 * ramp
    elif scenario == "S4":
        ext[:, 0] += 0.08 * p.tension_scale * np.sin(np.linspace(0, 9 * np.pi, steps))
    elif scenario == "S5":
        start, end = int(0.25 * steps), int(0.78 * steps)
        ext[start:end, 0] += 0.26 * p.tension_scale
        ext[start:end, 1] -= 0.20 * p.thickness_scale
        ext[start:end, 2] += 0.25 * p.flatness_scale
        ext[:, 3] += 0.16 * np.sin(np.linspace(0, 3 * np.pi, steps))
    elif scenario == "S6":
        start = int(0.32 * steps)
        ext[start:, 0] += 0.22 * p.tension_scale
        ext[start:, 1] += 0.18 * p.thickness_scale
        ext[start:, 2] += 0.24 * p.flatness_scale
        ext[:, 3] += 0.12 * np.sin(np.linspace(0, 2.4 * np.pi, steps))
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return {"proc": proc, "meas": meas, "ext": ext}


def initial_state(p: RealPass, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y0 = np.array([0.85 * p.tension_scale, -0.70 * p.thickness_scale, 0.62 * p.flatness_scale])
    y0 += rng.normal(0.0, [0.06 * p.tension_scale, 0.06 * p.thickness_scale, 0.06 * p.flatness_scale])
    ylag = 0.92 * y0
    hidden = np.zeros(3)
    return np.r_[y0, ylag, hidden]


def plant_step(
    x: np.ndarray,
    u: np.ndarray,
    p: RealPass,
    k: int,
    steps: int,
    ext: np.ndarray,
    proc: np.ndarray,
    allocation: bool = True,
    model_variant: str = "hybrid",
    command_gain_scale: float = 1.0,
    cross_coupling_scale: float = 1.0,
    output_clip_scale: float = 8.0,
) -> np.ndarray:
    y = np.asarray(x[:3], dtype=float)
    y_lag = np.asarray(x[3:6], dtype=float)
    d_force, d_thermal, d_couple = np.asarray(x[6:9], dtype=float)
    us, ugap, ushape = np.asarray(u, dtype=float)
    phase = phase_weights(k, steps, p)
    accel, steady, decel = phase

    thin_gain = float(np.clip(0.10 / max(p.exit_thickness, 0.035), 0.45, 2.6))
    reduction_gain = float(0.70 + 1.7 * np.clip(p.reduction_ratio, 0.0, 0.45))
    stage_coupling = 1.25 * accel + 0.92 * steady + 1.10 * decel
    stage_damping = 0.86 * accel + 1.05 * steady + 0.91 * decel

    d_force_next = 0.965 * d_force + 0.055 * p.rollforce_scale * ugap + 0.10 * p.rollforce_scale * ext[3] + proc[3]
    d_thermal_next = 0.988 * d_thermal + 0.025 * ext[4] + 0.012 * thin_gain * abs(ugap) + proc[4]
    d_couple_next = 0.972 * d_couple + 0.020 * np.tanh(y[0] / p.tension_scale) - 0.016 * np.tanh(y[1] / p.thickness_scale) + proc[5]

    ar1 = np.array([0.68, 0.62, 0.66]) * (1.02 - 0.08 * steady)
    ar2 = np.array([0.18, 0.22, 0.18])
    damping = np.array([0.055, 0.065, 0.058]) * stage_damping

    if allocation:
        B = np.array(
            [
                [-0.58 * p.tension_scale, 0.06 * p.tension_scale, -0.02 * p.tension_scale],
                [0.05 * p.thickness_scale, 0.72 * p.thickness_scale * reduction_gain, -0.01 * p.thickness_scale],
                [0.04 * p.flatness_scale, -0.08 * p.flatness_scale, 0.66 * p.flatness_scale * thin_gain],
            ]
        )
    else:
        B = np.array(
            [
                [-0.48 * p.tension_scale, 0.00, 0.00],
                [0.00, 0.56 * p.thickness_scale * reduction_gain, 0.00],
                [0.00, 0.00, 0.48 * p.flatness_scale * thin_gain],
            ]
        )
    if float(cross_coupling_scale) != 1.0:
        diagonal_B = np.diag(np.diag(B))
        B = diagonal_B + float(cross_coupling_scale) * (B - diagonal_B)

    coupling = stage_coupling * np.array(
        [
            -0.045 * p.tension_scale * np.tanh(y[1] / max(p.thickness_scale, 1e-9))
            + 0.020 * p.tension_scale * np.tanh(y[2] / max(p.flatness_scale, 1e-9)),
            0.038 * p.thickness_scale * np.tanh(y[0] / max(p.tension_scale, 1e-9))
            + 0.022 * p.thickness_scale * np.tanh(d_thermal_next),
            0.052 * p.flatness_scale * np.tanh(y[0] / max(p.tension_scale, 1e-9))
            + 0.080 * p.flatness_scale * np.tanh(y[1] / max(p.thickness_scale, 1e-9))
            + 0.090 * p.flatness_scale * np.tanh(d_couple_next),
        ]
    )
    if float(cross_coupling_scale) != 1.0:
        coupling *= float(cross_coupling_scale)
    hidden = np.array(
        [
            0.070 * p.tension_scale * np.tanh(d_force_next / max(p.rollforce_scale, 1e-9)),
            0.030 * p.thickness_scale * np.tanh(d_force_next / max(p.rollforce_scale, 1e-9)),
            0.100 * p.flatness_scale * np.tanh(d_couple_next) + 0.030 * p.flatness_scale * np.tanh(d_thermal_next),
        ]
    )
    stage_bias = accel * np.array([0.018 * p.tension_scale, -0.014 * p.thickness_scale, 0.020 * p.flatness_scale])
    stage_bias += decel * np.array([-0.014 * p.tension_scale, 0.018 * p.thickness_scale, 0.028 * p.flatness_scale])

    control_effect = float(command_gain_scale) * (B @ u)
    if model_variant == "weakened":
        control_effect *= 0.88
        coupling *= 1.10
    elif model_variant == "linear":
        hidden *= 0.35
        coupling *= 0.65
    y_next = ar1 * y + ar2 * y_lag - damping * y + control_effect + coupling + hidden + stage_bias + ext[:3] + proc[:3]
    clip_scale = float(output_clip_scale)
    if np.isfinite(clip_scale):
        y_next = np.clip(y_next, -clip_scale * np.array([p.tension_scale, p.thickness_scale, p.flatness_scale]), clip_scale * np.array([p.tension_scale, p.thickness_scale, p.flatness_scale]))
    return np.r_[y_next, y, [d_force_next, d_thermal_next, d_couple_next]]


def h_func(x: np.ndarray) -> np.ndarray:
    return np.asarray(x[:3], dtype=float)


def numerical_jacobian(func, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    y0 = np.asarray(func(x), dtype=float)
    J = np.zeros((len(y0), len(x)), dtype=float)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        step = eps * max(abs(x[i]), 1.0)
        xp[i] += step
        xm[i] -= step
        J[:, i] = (np.asarray(func(xp)) - np.asarray(func(xm))) / (2.0 * step)
    return J


def symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)


def robust_cholesky(P: np.ndarray) -> np.ndarray:
    P = symmetrize(P)
    jitter = 1e-9
    for _ in range(8):
        try:
            return np.linalg.cholesky(P + jitter * np.eye(P.shape[0]))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    vals, vecs = np.linalg.eigh(P)
    vals = np.maximum(vals, 1e-9)
    return vecs @ np.diag(np.sqrt(vals))


class BaseFilter:
    def __init__(self, x0, P0, Q, R, p: RealPass, steps: int, seed: int, particle_count: int):
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.Q = np.asarray(Q, dtype=float).copy()
        self.R = np.asarray(R, dtype=float).copy()
        self.p = p
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.particle_count = particle_count

    def f(self, x: np.ndarray, u: np.ndarray, k: int, allocation: bool = True) -> np.ndarray:
        return plant_step(x, u, self.p, k, self.steps, np.zeros(5), np.zeros(6), allocation=allocation)

    def step(self, u: np.ndarray, z: np.ndarray, k: int, allocation: bool = True) -> tuple[np.ndarray, float]:
        raise NotImplementedError


class RawFilter(BaseFilter):
    def step(self, u: np.ndarray, z: np.ndarray, k: int, allocation: bool = True) -> tuple[np.ndarray, float]:
        pred = self.f(self.x, u, k, allocation)
        residual = z - h_func(pred)
        pred[:3] = z
        self.x = pred
        return self.x.copy(), float(np.linalg.norm(residual))


class EKFFilter(BaseFilter):
    def step(self, u: np.ndarray, z: np.ndarray, k: int, allocation: bool = True) -> tuple[np.ndarray, float]:
        F = numerical_jacobian(lambda xx: self.f(xx, u, k, allocation), self.x)
        x_pred = self.f(self.x, u, k, allocation)
        P_pred = symmetrize(F @ self.P @ F.T + self.Q)
        H = numerical_jacobian(h_func, x_pred)
        residual = z - h_func(x_pred)
        S = symmetrize(H @ P_pred @ H.T + self.R)
        K = P_pred @ H.T @ np.linalg.pinv(S)
        self.x = x_pred + K @ residual
        I = np.eye(len(self.x))
        self.P = symmetrize((I - K @ H) @ P_pred @ (I - K @ H).T + K @ self.R @ K.T)
        return self.x.copy(), float(np.linalg.norm(residual))


class UKFFilter(BaseFilter):
    def __init__(self, *args, alpha=0.50, beta=2.0, kappa=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def sigma_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.x)
        lam = self.alpha**2 * (n + self.kappa) - n
        scale = n + lam
        S = robust_cholesky(scale * self.P)
        points = [self.x]
        for i in range(n):
            points.append(self.x + S[:, i])
            points.append(self.x - S[:, i])
        wm = np.full(2 * n + 1, 1.0 / (2.0 * scale))
        wc = wm.copy()
        wm[0] = lam / scale
        wc[0] = lam / scale + 1.0 - self.alpha**2 + self.beta
        return np.asarray(points), wm, wc

    def step(self, u: np.ndarray, z: np.ndarray, k: int, allocation: bool = True) -> tuple[np.ndarray, float]:
        points, wm, wc = self.sigma_points()
        x_sigma = np.asarray([self.f(pt, u, k, allocation) for pt in points])
        x_pred = np.sum(wm[:, None] * x_sigma, axis=0)
        dx = x_sigma - x_pred
        P_pred = self.Q.copy()
        for i in range(len(wc)):
            P_pred += wc[i] * np.outer(dx[i], dx[i])
        z_sigma = np.asarray([h_func(pt) for pt in x_sigma])
        z_pred = np.sum(wm[:, None] * z_sigma, axis=0)
        dz = z_sigma - z_pred
        Szz = self.R.copy()
        Pxz = np.zeros((len(self.x), len(z)), dtype=float)
        for i in range(len(wc)):
            Szz += wc[i] * np.outer(dz[i], dz[i])
            Pxz += wc[i] * np.outer(dx[i], dz[i])
        residual = z - z_pred
        K = Pxz @ np.linalg.pinv(symmetrize(Szz))
        self.x = x_pred + K @ residual
        self.P = symmetrize(P_pred - K @ Szz @ K.T)
        return self.x.copy(), float(np.linalg.norm(residual))


class ParticleFilter(BaseFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.particles = self.rng.multivariate_normal(self.x, self.P, size=self.particle_count)
        self.weights = np.full(self.particle_count, 1.0 / self.particle_count)
        self.Q_chol = robust_cholesky(self.Q)

    def resample(self) -> None:
        n = self.particle_count
        positions = (self.rng.random() + np.arange(n)) / n
        cumsum = np.cumsum(self.weights)
        idx = np.searchsorted(cumsum, positions, side="right")
        idx = np.clip(idx, 0, n - 1)
        self.particles = self.particles[idx]
        self.weights.fill(1.0 / n)

    def step(self, u: np.ndarray, z: np.ndarray, k: int, allocation: bool = True) -> tuple[np.ndarray, float]:
        noise = self.rng.normal(size=(self.particle_count, len(self.x))) @ self.Q_chol.T
        for i in range(self.particle_count):
            self.particles[i] = self.f(self.particles[i], u, k, allocation) + noise[i]
        z_pred = self.particles[:, :3]
        residuals = z_pred - z.reshape(1, -1)
        R_inv = np.linalg.pinv(self.R)
        logw = -0.5 * np.einsum("ij,jk,ik->i", residuals, R_inv, residuals)
        logw -= np.max(logw)
        w = np.exp(logw) * self.weights
        total = float(np.sum(w))
        if total <= 1e-30 or not np.isfinite(total):
            self.weights.fill(1.0 / self.particle_count)
        else:
            self.weights = w / total
        ess = 1.0 / np.sum(self.weights**2)
        if ess < 0.55 * self.particle_count:
            self.resample()
        self.x = np.average(self.particles, axis=0, weights=self.weights)
        centered = self.particles - self.x
        self.P = symmetrize((centered.T * self.weights) @ centered + 1e-8 * np.eye(len(self.x)))
        return self.x.copy(), float(np.linalg.norm(z - h_func(self.x)))


class GridBayesFilter(BaseFilter):
    def __init__(self, *args, grid_points=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_points = grid_points

    def step(self, u: np.ndarray, z: np.ndarray, k: int, allocation: bool = True) -> tuple[np.ndarray, float]:
        pred = self.f(self.x, u, k, allocation)
        P_pred = symmetrize(self.P + self.Q)
        y_pred = h_func(pred)
        var = np.maximum(np.diag(P_pred)[:3], np.diag(self.R) * 0.5)
        axes = [np.linspace(y_pred[i] - 3.0 * np.sqrt(var[i]), y_pred[i] + 3.0 * np.sqrt(var[i]), self.grid_points) for i in range(3)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        prior = np.exp(-0.5 * np.sum((mesh - y_pred) ** 2 / var.reshape(1, -1), axis=1))
        like = np.exp(-0.5 * np.sum((mesh - z.reshape(1, -1)) ** 2 / np.diag(self.R).reshape(1, -1), axis=1))
        w = prior * like
        total = float(np.sum(w))
        if total > 1e-30 and np.isfinite(total):
            w /= total
            y_post = np.sum(w[:, None] * mesh, axis=0)
            post_var = np.sum(w[:, None] * (mesh - y_post) ** 2, axis=0)
        else:
            y_post = y_pred
            post_var = var
        residual = y_post - y_pred
        pred[:3] = y_post
        pred[6] += 0.012 * residual[0]
        pred[7] += 0.010 * residual[1] / max(self.p.thickness_scale, 1e-9)
        pred[8] += 0.012 * residual[2] / max(self.p.flatness_scale, 1e-9)
        P_pred[np.ix_([0, 1, 2], [0, 1, 2])] = np.diag(np.maximum(post_var, 1e-9))
        self.x = pred
        self.P = P_pred
        return self.x.copy(), float(np.linalg.norm(z - y_pred))


def filter_covariances(p: RealPass, scenario: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scales = np.array([p.tension_scale, p.thickness_scale, p.flatness_scale])
    P0 = np.diag(np.r_[0.38 * scales, 0.30 * scales, 0.35 * p.rollforce_scale, 0.25, 0.25]) ** 2
    Q = np.diag(np.r_[0.055 * scales, 0.035 * scales, 0.030 * p.rollforce_scale, 0.025, 0.025]) ** 2
    factor = 3.0 if scenario == "S4" else 1.0
    R = np.diag(factor * np.array([0.060 * p.tension_scale, 0.055 * p.thickness_scale, 0.065 * p.flatness_scale])) ** 2
    return P0, Q, R


def build_filter(filter_id: str, x0: np.ndarray, p: RealPass, steps: int, scenario: str, seed: int, particle_count: int):
    P0, Q, R = filter_covariances(p, scenario)
    common = dict(x0=x0, P0=P0, Q=Q, R=R, p=p, steps=steps, seed=seed, particle_count=particle_count)
    if filter_id == "raw":
        return RawFilter(**common)
    if filter_id == "ekf":
        return EKFFilter(**common)
    if filter_id == "ukf":
        return UKFFilter(**common)
    if filter_id == "pf":
        return ParticleFilter(**common)
    if filter_id == "grid_bayes":
        return GridBayesFilter(**common)
    raise ValueError(filter_id)


def scheduled_control(k: int, steps: int, scenario: str) -> np.ndarray:
    t = k / max(steps - 1, 1)
    u = np.array(
        [
            0.20 * np.sin(2 * np.pi * (1.1 * t + 0.06)),
            -0.16 * np.sin(2 * np.pi * (0.7 * t + 0.22)),
            0.18 * np.sin(2 * np.pi * (1.4 * t + 0.15)),
        ]
    )
    if scenario in {"S5", "S6"} and t > 0.35:
        u += np.array([-0.04, 0.05, -0.05])
    return np.clip(u, -0.42, 0.42)


def predict_stage_narx(x_est: np.ndarray, p: RealPass, k: int, steps: int, horizon: int, quality: str, allocation: bool = True) -> np.ndarray:
    pred = []
    x_roll = x_est.copy()
    q = {
        "persistence": 0.0,
        "dmc": 0.25,
        "lstm": 0.42,
        "stage_narx": 0.62,
        "attention": 0.68,
    }[quality]
    for i in range(horizon):
        phase = phase_weights(min(k + i, steps - 1), steps, p)
        drift = q * (
            phase[0] * np.array([0.008 * p.tension_scale, -0.006 * p.thickness_scale, 0.010 * p.flatness_scale])
            + phase[2] * np.array([-0.006 * p.tension_scale, 0.008 * p.thickness_scale, 0.012 * p.flatness_scale])
        )
        x_roll = plant_step(
            x_roll,
            np.zeros(3),
            p,
            min(k + i, steps - 1),
            steps,
            np.r_[drift, 0.0, 0.0],
            np.zeros(6),
            allocation=allocation,
            model_variant="linear" if quality == "dmc" else "hybrid",
        )
        y = h_func(x_roll).copy()
        if quality == "persistence":
            y = h_func(x_est)
        elif quality == "lstm":
            y = 0.78 * y + 0.22 * h_func(x_est)
        elif quality == "attention":
            y += 0.05 * np.tanh((h_func(x_est) - y) / np.array([p.tension_scale, p.thickness_scale, p.flatness_scale])) * np.array([p.tension_scale, p.thickness_scale, p.flatness_scale])
        pred.append(y)
    return np.asarray(pred)


def response_matrix(p: RealPass, horizon: int, allocation: bool) -> np.ndarray:
    thin_gain = float(np.clip(0.10 / max(p.exit_thickness, 0.035), 0.45, 2.6))
    reduction_gain = float(0.70 + 1.7 * np.clip(p.reduction_ratio, 0.0, 0.45))
    if allocation:
        gains = np.array(
            [
                [-0.50 * p.tension_scale, 0.055 * p.tension_scale, -0.020 * p.tension_scale],
                [0.040 * p.thickness_scale, 0.64 * p.thickness_scale * reduction_gain, -0.012 * p.thickness_scale],
                [0.040 * p.flatness_scale, -0.070 * p.flatness_scale, 0.58 * p.flatness_scale * thin_gain],
            ]
        )
    else:
        gains = np.array(
            [
                [-0.42 * p.tension_scale, 0.0, 0.0],
                [0.0, 0.49 * p.thickness_scale * reduction_gain, 0.0],
                [0.0, 0.0, 0.42 * p.flatness_scale * thin_gain],
            ]
        )
    tau = np.array([2.8, 3.4, 3.1])
    return np.asarray([gains * (1.0 - np.exp(-(i + 1) / tau)).reshape(-1, 1) for i in range(horizon)])


def stage_control_weights(k: int, steps: int, p: RealPass, dynamic: bool) -> np.ndarray:
    if not dynamic:
        return np.array([5.5, 6.0, 5.5])
    phase = phase_name(k, steps, p)
    if phase == "accel":
        return np.array([8.5, 4.8, 4.0])
    if phase == "decel":
        return np.array([5.2, 5.2, 8.8])
    return np.array([5.0, 8.6, 5.2])


def solve_mpc(base_pred: np.ndarray, weights: np.ndarray, p: RealPass, u_prev: np.ndarray, u_min: np.ndarray, u_max: np.ndarray, allocation: bool, dynamic_weights: bool, penalty_scale: float = 1.0) -> np.ndarray:
    horizon = len(base_pred)
    resp = response_matrix(p, horizon, allocation)
    scales = np.array([p.tension_scale, p.thickness_scale, p.flatness_scale])
    rows, rhs = [], []
    for t in range(horizon):
        for j in range(3):
            row = resp[t, j, :] / max(scales[j], 1e-9)
            rows.append(np.sqrt(weights[j]) * row)
            rhs.append(np.sqrt(weights[j]) * base_pred[t, j] / max(scales[j], 1e-9))
    A = np.vstack(rows)
    b = np.asarray(rhs)
    penalty = np.array([22.0, 26.0, 24.0]) if dynamic_weights else np.array([16.0, 20.0, 18.0])
    H = A.T @ A + np.diag(penalty * penalty_scale)
    g = A.T @ b
    try:
        du = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        du = -np.linalg.pinv(H) @ g
    du_lim = np.array([0.070, 0.060, 0.070]) if allocation else np.array([0.095, 0.080, 0.095])
    return np.clip(u_prev + np.clip(du, -du_lim, du_lim), u_min, u_max)


def make_pid(control_id: str, p: RealPass, dt: float) -> PID:
    kp = np.array([-0.62 / p.tension_scale, 0.58 / p.thickness_scale, 0.62 / p.flatness_scale])
    ki = np.array([-0.10 / p.tension_scale, 0.08 / p.thickness_scale, 0.08 / p.flatness_scale])
    if control_id in {"C0", "C1"}:
        gain, limit = 0.70, np.array([0.40, 0.36, 0.40])
    elif control_id in {"C2", "C3"}:
        gain, limit = 0.64, np.array([0.36, 0.32, 0.36])
    elif control_id == "C7":
        gain, limit = 0.76, np.array([0.36, 0.32, 0.36])
    else:
        gain, limit = 0.48, np.array([0.30, 0.26, 0.30])
    return PID(kp=gain * kp, ki=gain * ki, kd=[0.0, 0.0, 0.0], dt=dt, umin=-limit, umax=limit)


def control_flags(control_id: str) -> dict[str, object]:
    return {
        "filter": control_id in {"C1", "C3", "C4", "C5", "C6", "C7", "LSTM-MPC", "ATT-MPC"},
        "ekf_baseline": control_id in {"DMC-KF", "ADRC", "ROBUST-MPC", "ADAPTIVE-MPC"},
        "predictor": {
            "C0": "persistence",
            "C1": "persistence",
            "C2": "stage_narx",
            "C3": "stage_narx",
            "C4": "stage_narx",
            "C5": "stage_narx",
            "C6": "stage_narx",
            "C7": "stage_narx",
            "LSTM-MPC": "lstm",
            "DMC-KF": "dmc",
            "ATT-MPC": "attention",
            "ADRC": "persistence",
            "ROBUST-MPC": "stage_narx",
            "ADAPTIVE-MPC": "stage_narx",
        }[control_id],
        "mpc": control_id in {"C4", "C5", "C6", "C7", "LSTM-MPC", "DMC-KF", "ATT-MPC", "ROBUST-MPC", "ADAPTIVE-MPC"},
        "pid": control_id in {"C0", "C1", "C2", "C3", "C5", "C6", "C7", "ADRC", "ADAPTIVE-MPC"},
        "dynamic_weights": control_id in {"C7", "ADAPTIVE-MPC"},
        "allocation": control_id != "C6",
    }


def feedforward_from_prediction(pred: np.ndarray, p: RealPass) -> np.ndarray:
    mean_pred = pred[: min(5, len(pred))].mean(axis=0)
    return np.array(
        [
            0.22 * mean_pred[0] / p.tension_scale,
            -0.22 * mean_pred[1] / p.thickness_scale,
            -0.22 * mean_pred[2] / p.flatness_scale,
        ]
    )


def run_filter_estimation(filter_id: str, scenario: str, p: RealPass, steps: int, seed: int, particle_count: int) -> tuple[pd.DataFrame, dict]:
    profile = make_noise_profile(scenario, steps, seed, p)
    x_true = initial_state(p, seed)
    x0 = x_true + np.r_[0.18 * np.array([p.tension_scale, -p.thickness_scale, p.flatness_scale]), np.zeros(6)]
    filt = build_filter(filter_id, x0, p, steps, scenario, seed + 19, particle_count)
    rows, times = [], []
    for k in range(steps):
        u = scheduled_control(k, steps, scenario)
        x_true = plant_step(x_true, u, p, k, steps, profile["ext"][k], profile["proc"][k], allocation=True)
        z = h_func(x_true) + profile["meas"][k]
        start = time.perf_counter()
        x_est, residual = filt.step(u, z, k, allocation=True)
        times.append((time.perf_counter() - start) * 1000.0)
        rows.append(
            {
                "k": k,
                "time_s": SAMPLE_PERIOD_S * k,
                "phase": phase_name(k, steps, p),
                "T_true": x_true[0],
                "h_true": x_true[1],
                "S_true": x_true[2],
                "T_est": x_est[0],
                "h_est": x_est[1],
                "S_est": x_est[2],
                "T_meas": z[0],
                "h_meas": z[1],
                "S_meas": z[2],
                "residual_norm": residual,
            }
        )
    diag = {"filter_ms_mean": float(np.mean(times)), "filter_ms_p95": float(np.percentile(times, 95)), "filter_ms_max": float(np.max(times))}
    return pd.DataFrame(rows), diag


def run_closed_loop(
    control_id: str,
    scenario: str,
    p: RealPass,
    steps: int,
    seed: int,
    filter_id: str,
    particle_count: int,
    horizon: int = 10,
    residual_rho: float = 0.30,
    mpc_penalty_scale: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    flags = control_flags(control_id)
    profile = make_noise_profile(scenario, steps, seed, p)
    x = initial_state(p, seed)
    x0 = x + np.r_[0.16 * np.array([p.tension_scale, -p.thickness_scale, p.flatness_scale]), np.zeros(6)]
    active_filter = "ekf" if flags["ekf_baseline"] else filter_id if flags["filter"] else "raw"
    filt = build_filter(active_filter, x0, p, steps, scenario, seed + 73, particle_count)
    pid = make_pid(control_id, p, SAMPLE_PERIOD_S)
    limit = 0.38 if scenario == "S6" else 0.64
    u_min = np.array([-limit, -limit, -limit])
    u_max = np.array([limit, limit, limit])
    u = np.zeros(3)
    rows, solve_ms, filter_ms = [], [], []
    sat_count = 0
    constraint_count = 0
    for k in range(steps):
        z = h_func(x) + profile["meas"][k]
        filter_start = time.perf_counter()
        if flags["filter"] or flags["ekf_baseline"]:
            x_est, residual = filt.step(u, z, k, allocation=bool(flags["allocation"]))
        else:
            x_est, residual = filt.step(u, z, k, allocation=True)
        if control_id == "C7":
            # Residual correction keeps hidden-state filtering while avoiding output lag in feedback.
            rho = float(np.clip(residual_rho, 0.0, 1.0))
            x_est[:3] = rho * x_est[:3] + (1.0 - rho) * z
        filter_ms.append((time.perf_counter() - filter_start) * 1000.0)
        y = h_func(x_est)
        start = time.perf_counter()
        pred = predict_stage_narx(x_est, p, k, steps, horizon, str(flags["predictor"]), allocation=bool(flags["allocation"]))
        u_target = np.zeros(3)
        if flags["mpc"]:
            weights = stage_control_weights(k, steps, p, bool(flags["dynamic_weights"]))
            penalty = 1.0
            if control_id == "LSTM-MPC":
                penalty = 0.88
            if control_id == "DMC-KF":
                penalty = 1.20
            if control_id == "ATT-MPC":
                penalty = 0.95
            if control_id == "ROBUST-MPC":
                penalty = 1.65
            if control_id == "ADAPTIVE-MPC":
                penalty = 1.05
            if control_id == "C7":
                penalty *= mpc_penalty_scale
            u_mpc = solve_mpc(pred, weights, p, u, u_min, u_max, bool(flags["allocation"]), bool(flags["dynamic_weights"]), penalty_scale=penalty)
            if flags["pid"]:
                ff = feedforward_from_prediction(pred, p)
                if control_id == "C7":
                    u_target = 0.08 * u_mpc + 0.35 * ff
                elif control_id == "C5":
                    u_target = 0.12 * u_mpc + 0.62 * ff
                else:
                    u_target = 0.10 * u_mpc + 0.56 * ff
            else:
                u_target = u_mpc
        elif str(flags["predictor"]) != "persistence" and control_id in {"C2", "C3"}:
            u_target = np.clip(feedforward_from_prediction(pred, p), u_min, u_max)
        if control_id == "ADRC":
            disturbance_est = np.array([
                0.20 * x_est[6] / max(p.rollforce_scale, 1e-9),
                -0.16 * x_est[7],
                -0.18 * x_est[8],
            ])
            u_target = np.clip(u_target + disturbance_est, u_min, u_max)
        elif control_id == "ADAPTIVE-MPC":
            adapt = 1.0 + 0.25 * np.tanh(np.linalg.norm(y / np.array([p.tension_scale, p.thickness_scale, p.flatness_scale])))
            u_target = np.clip(adapt * u_target, u_min, u_max)
        if flags["pid"]:
            u_target = np.clip(u_target + pid.step(-y), u_min, u_max)
        solve_ms.append((time.perf_counter() - start) * 1000.0)
        du_lim = np.array([0.065, 0.052, 0.065]) if flags["mpc"] else np.array([0.090, 0.075, 0.090])
        if scenario == "S6":
            du_lim *= 0.62
        du = np.clip(u_target - u, -du_lim, du_lim)
        u = np.clip(u + du, u_min, u_max)
        sat_count += int(np.any(np.isclose(np.abs(u), limit, atol=1e-6)))
        x = plant_step(x, u, p, k, steps, profile["ext"][k], profile["proc"][k], allocation=True)
        scale_vec = np.array([p.tension_scale, p.thickness_scale, p.flatness_scale])
        constraint_count += int(np.any(np.abs(h_func(x)) > 3.0 * scale_vec))
        rows.append(
            {
                "k": k,
                "time_s": SAMPLE_PERIOD_S * k,
                "phase": phase_name(k, steps, p),
                "T_error": x[0],
                "h_error": x[1],
                "S_error": x[2],
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
            }
        )
    diag = {
        "active_filter": active_filter,
        "sat_count": int(sat_count),
        "sat_ratio": float(sat_count / max(steps, 1)),
        "constraint_count": int(constraint_count),
        "constraint_ratio": float(constraint_count / max(steps, 1)),
        "filter_ms_mean": float(np.mean(filter_ms)),
        "filter_ms_p95": float(np.percentile(filter_ms, 95)),
        "solve_ms_mean": float(np.mean(solve_ms)),
        "solve_ms_p95": float(np.percentile(solve_ms, 95)),
        "solve_ms_max": float(np.max(solve_ms)),
    }
    return pd.DataFrame(rows), diag


def disturbance_start_index(scenario: str, n: int) -> int:
    return {
        "S1": int(0.50 * n),
        "S2": int(0.30 * n),
        "S3": int(0.30 * n),
        "S4": int(0.45 * n),
        "S5": int(0.25 * n),
        "S6": int(0.32 * n),
    }[scenario]


def settling_time(y: np.ndarray, time_s: np.ndarray, tol: float, start: int) -> float | None:
    for i in range(start, len(y)):
        if np.all(np.abs(y[i:]) <= tol):
            return float(time_s[i] - time_s[start])
    return None


def estimation_metrics(df: pd.DataFrame, p: RealPass, diag: dict, filter_id: str, scenario: str, seed: int) -> dict:
    out = {
        "filter": filter_id,
        "filter_label": FILTER_LABELS[filter_id],
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "pass_file": p.pass_file,
        "seed": seed,
    }
    for key, scale in [("T", p.tension_scale), ("h", p.thickness_scale), ("S", p.flatness_scale)]:
        err = df[f"{key}_est"] - df[f"{key}_true"]
        meas_err = df[f"{key}_meas"] - df[f"{key}_true"]
        out[f"{key}_estimate_RMSE"] = float(np.sqrt(np.mean(err**2)))
        out[f"{key}_measurement_RMSE"] = float(np.sqrt(np.mean(meas_err**2)))
        out[f"{key}_normalized_RMSE"] = float(out[f"{key}_estimate_RMSE"] / max(scale, 1e-9))
        out[f"{key}_improve_vs_measurement_%"] = float((out[f"{key}_measurement_RMSE"] - out[f"{key}_estimate_RMSE"]) / max(out[f"{key}_measurement_RMSE"], 1e-12) * 100.0)
    out["mean_normalized_RMSE"] = float(np.mean([out["T_normalized_RMSE"], out["h_normalized_RMSE"], out["S_normalized_RMSE"]]))
    out["mean_improve_vs_measurement_%"] = float(np.mean([out["T_improve_vs_measurement_%"], out["h_improve_vs_measurement_%"], out["S_improve_vs_measurement_%"]]))
    out.update(diag)
    return out


def control_metrics(df: pd.DataFrame, p: RealPass, diag: dict, control_id: str, scenario: str, seed: int) -> dict:
    out = {
        "control": control_id,
        "control_label": CONTROL_LABELS[control_id],
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "pass_file": p.pass_file,
        "seed": seed,
    }
    scales = {"T_error": p.tension_scale, "h_error": p.thickness_scale, "S_error": p.flatness_scale}
    start_idx = disturbance_start_index(scenario, len(df))
    steady_idx = int(0.85 * len(df))
    for col, scale in scales.items():
        y = df[col].to_numpy(dtype=float)
        out[f"{col}_RMS"] = float(np.sqrt(np.mean(y**2)))
        out[f"{col}_normalized_RMS"] = float(out[f"{col}_RMS"] / max(scale, 1e-9))
        out[f"{col}_MAE"] = float(np.mean(np.abs(y)))
        out[f"{col}_IAE"] = float(np.sum(np.abs(y)) * SAMPLE_PERIOD_S)
        out[f"{col}_ISE"] = float(np.sum(y**2) * SAMPLE_PERIOD_S)
        out[f"{col}_max_abs"] = float(np.max(np.abs(y)))
        out[f"{col}_steady_abs_mean"] = float(np.mean(np.abs(y[steady_idx:])))
        out[f"{col}_peak_after_disturbance"] = float(np.max(np.abs(y[start_idx:])))
        out[f"{col}_settling_s"] = settling_time(y, df["time_s"].to_numpy(dtype=float), 0.10 * scale, int(0.55 * len(df)))
    applied = df[["u_speed", "u_gap", "u_shape"]].to_numpy(dtype=float)
    segments = df["segment_id"].to_numpy() if "segment_id" in df else None
    du = supported_command_deltas(applied, segments)
    out["control_delta_rms"] = float(np.sqrt(np.mean(du**2))) if len(du) else 0.0
    out["control_total_variation"] = supported_total_variation(applied, segments)
    out.update(reset_command_motion(applied, segments))
    out["composite_normalized_RMS"] = float(np.mean([out["T_error_normalized_RMS"], out["h_error_normalized_RMS"], out["S_error_normalized_RMS"]]))
    out.update(diag)
    return out


def aggregate_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    skip = set(group_cols) | {"pass_file", "seed", "active_filter"}
    metric_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    return df.groupby(group_cols, as_index=False)[metric_cols].mean()


def choose_best_filter(filter_control_summary: pd.DataFrame) -> str:
    overall = filter_control_summary.groupby("filter", as_index=False)["composite_normalized_RMS"].mean()
    nonlinear = overall[overall["filter"] != "raw"].copy()
    if nonlinear.empty:
        nonlinear = overall
    return str(nonlinear.sort_values("composite_normalized_RMS").iloc[0]["filter"])


def plot_filter_outputs(est_summary: pd.DataFrame, ctrl_summary: pd.DataFrame, out_dir: Path) -> None:
    est = est_summary.groupby("filter", as_index=False)[["mean_normalized_RMSE", "mean_improve_vs_measurement_%", "filter_ms_p95"]].mean().sort_values("mean_normalized_RMSE")
    ctrl = ctrl_summary.groupby("filter", as_index=False)[["composite_normalized_RMS", "filter_ms_p95"]].mean().sort_values("composite_normalized_RMS")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].bar(est["filter"], est["mean_normalized_RMSE"])
    axes[0].set_title("Filtering RMSE")
    axes[1].bar(ctrl["filter"], ctrl["composite_normalized_RMS"])
    axes[1].set_title("C7 closed-loop RMS by filter")
    axes[2].bar(est["filter"], est["filter_ms_p95"])
    axes[2].set_title("Filter p95 time (ms)")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir / "realdata_filter_overall.png", dpi=220)
    plt.close(fig)
    pivot = ctrl_summary.pivot_table(index="scenario", columns="filter", values="composite_normalized_RMS", aggfunc="mean")
    plt.figure(figsize=(8.5, 4.8))
    im = plt.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.colorbar(im, label="Composite normalized RMS")
    plt.title("Filter choice under disturbance scenarios")
    plt.tight_layout()
    plt.savefig(out_dir / "realdata_filter_scenario_heatmap.png", dpi=220)
    plt.close()


def plot_control_outputs(summary: pd.DataFrame, out_dir: Path) -> None:
    overall = summary.groupby("control", as_index=False)[["T_error_normalized_RMS", "h_error_normalized_RMS", "S_error_normalized_RMS", "composite_normalized_RMS", "control_total_variation", "constraint_ratio"]].mean()
    order = overall.sort_values("composite_normalized_RMS")["control"].tolist()
    overall["control"] = pd.Categorical(overall["control"], categories=order, ordered=True)
    overall = overall.sort_values("control")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, col, title in zip(
        axes,
        ["T_error_normalized_RMS", "h_error_normalized_RMS", "S_error_normalized_RMS", "composite_normalized_RMS"],
        ["Tension RMS", "Thickness RMS", "Flatness RMS", "Composite RMS"],
    ):
        ax.bar(overall["control"].astype(str), overall[col])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(out_dir / "realdata_control_overall_bars.png", dpi=220)
    plt.close(fig)
    pivot = summary.pivot_table(index="scenario", columns="control", values="composite_normalized_RMS", aggfunc="mean")
    plt.figure(figsize=(10.0, 5.0))
    im = plt.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.colorbar(im, label="Composite normalized RMS")
    plt.title("Controller comparison across disturbance scenarios")
    plt.tight_layout()
    plt.savefig(out_dir / "realdata_control_scenario_heatmap.png", dpi=220)
    plt.close()


def plot_example_curves(curves: dict[str, pd.DataFrame], out_dir: Path, suffix: str = "S5") -> None:
    for col, label in [("T_error", "Tension error"), ("h_error", "Thickness deviation"), ("S_error", "Flatness deviation")]:
        plt.figure(figsize=(9.5, 4.5))
        for cid, df in curves.items():
            plt.plot(df["time_s"], df[col], label=cid, linewidth=1.5)
        plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        plt.xlabel("Time (s)")
        plt.ylabel(label)
        plt.title(f"{suffix} example response")
        plt.grid(True, alpha=0.35)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"realdata_example_{suffix}_{col}.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    project_dir = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=str(project_dir / "artifacts" / "realdata_closed_loop_suite"))
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pass-mode", choices=["representative", "all"], default="representative")
    parser.add_argument("--pass-limit", type=int, default=0)
    parser.add_argument("--filters", default="raw,ekf,ukf,pf,grid_bayes")
    parser.add_argument("--controls", default="C0,C1,C2,C3,C4,C5,C6,C7,LSTM-MPC,DMC-KF,ATT-MPC")
    parser.add_argument("--scenarios", default="S1,S2,S3,S4,S5,S6")
    parser.add_argument("--particle-count", type=int, default=220)
    parser.add_argument("--selected-filter", default="auto")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_real_process_data(args.data_dir)
    passes = representative_passes([make_real_pass(df) for df in dataset.frames], args.pass_mode, args.pass_limit)
    filter_ids = [x.strip() for x in args.filters.split(",") if x.strip()]
    control_ids = [x.strip() for x in args.controls.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    for fid in filter_ids:
        if fid not in FILTER_LABELS:
            raise ValueError(f"Unknown filter: {fid}")
    for cid in control_ids:
        if cid not in CONTROL_LABELS:
            raise ValueError(f"Unknown control: {cid}")

    pass_meta = pd.DataFrame([asdict(p) for p in passes])
    pass_meta.to_csv(out_dir / "realdata_passes_used.csv", index=False)

    est_rows, filter_ctrl_rows = [], []
    print("Running nonlinear filter comparison...")
    for p in passes:
        print(f"Filter pass {p.pass_file}", flush=True)
        for scenario in scenarios:
            for seed in range(args.repeats):
                base_seed = 5000 + 100 * seed + p.pass_index
                for fid in filter_ids:
                    est_df, est_diag = run_filter_estimation(fid, scenario, p, args.steps, base_seed, args.particle_count)
                    est_rows.append(estimation_metrics(est_df, p, est_diag, fid, scenario, seed))
                    ctrl_df, ctrl_diag = run_closed_loop("C7", scenario, p, args.steps, base_seed + 31, fid, args.particle_count)
                    row = control_metrics(ctrl_df, p, ctrl_diag, "C7", scenario, seed)
                    row["filter"] = fid
                    row["filter_label"] = FILTER_LABELS[fid]
                    filter_ctrl_rows.append(row)

    filter_est_raw = pd.DataFrame(est_rows)
    filter_ctrl_raw = pd.DataFrame(filter_ctrl_rows)
    filter_est_summary = aggregate_metrics(filter_est_raw, ["filter", "filter_label", "scenario", "scenario_label"])
    filter_ctrl_summary = aggregate_metrics(filter_ctrl_raw, ["filter", "filter_label", "scenario", "scenario_label"])
    filter_est_raw.to_csv(out_dir / "realdata_filter_estimation_raw.csv", index=False)
    filter_ctrl_raw.to_csv(out_dir / "realdata_filter_control_raw.csv", index=False)
    filter_est_summary.to_csv(out_dir / "realdata_filter_estimation_summary.csv", index=False)
    filter_ctrl_summary.to_csv(out_dir / "realdata_filter_control_summary.csv", index=False)
    plot_filter_outputs(filter_est_summary, filter_ctrl_summary, out_dir)

    selected_filter = choose_best_filter(filter_ctrl_summary) if args.selected_filter == "auto" else args.selected_filter
    if selected_filter not in FILTER_LABELS:
        raise ValueError(f"Unknown selected filter: {selected_filter}")
    print(f"Selected filter for C1/C3-C7 and predictive controller candidates: {selected_filter}", flush=True)

    control_rows = []
    example_curves: dict[str, pd.DataFrame] = {}
    example_pass = passes[len(passes) // 2]
    print("Running locked controller comparison...")
    for p in passes:
        print(f"Control pass {p.pass_file}", flush=True)
        for scenario in scenarios:
            for seed in range(args.repeats):
                base_seed = 8000 + 100 * seed + p.pass_index
                for cid in control_ids:
                    df, diag = run_closed_loop(cid, scenario, p, args.steps, base_seed, selected_filter, args.particle_count)
                    control_rows.append(control_metrics(df, p, diag, cid, scenario, seed))
                    if p.pass_file == example_pass.pass_file and scenario == "S5" and seed == 0 and cid in {"C0", "C3", "C5", "C6", "C7", "LSTM-MPC", "DMC-KF", "ATT-MPC"}:
                        example_curves[cid] = df

    control_raw = pd.DataFrame(control_rows)
    control_summary = aggregate_metrics(control_raw, ["control", "control_label", "scenario", "scenario_label"])
    control_overall = aggregate_metrics(control_raw, ["control", "control_label"])
    baseline = control_overall[control_overall["control"] == "C0"][["composite_normalized_RMS", "T_error_IAE", "h_error_IAE", "S_error_IAE"]].iloc[0]
    control_overall["composite_improve_vs_C0_%"] = (baseline["composite_normalized_RMS"] - control_overall["composite_normalized_RMS"]) / max(baseline["composite_normalized_RMS"], 1e-12) * 100.0
    for col in ["T_error_IAE", "h_error_IAE", "S_error_IAE"]:
        control_overall[f"{col}_improve_vs_C0_%"] = (baseline[col] - control_overall[col]) / max(baseline[col], 1e-12) * 100.0
    control_raw.to_csv(out_dir / "realdata_control_raw.csv", index=False)
    control_summary.to_csv(out_dir / "realdata_control_summary.csv", index=False)
    control_overall.to_csv(out_dir / "realdata_control_overall.csv", index=False)
    plot_control_outputs(control_summary, out_dir)
    if example_curves:
        plot_example_curves(example_curves, out_dir, "S5")

    config = vars(args).copy()
    config["selected_filter"] = selected_filter
    config["data_dir"] = str(dataset.data_dir)
    config["plant_model_basis"] = {
        "tension": "conservative persistence/linear dynamic baseline from LOPO result",
        "thickness": "stage-conditional NARX-style dynamics calibrated by LOPO result",
        "flatness": "linear ARX-style dynamics calibrated by LOPO result",
        "stage_condition": "acceleration/steady/deceleration phase from real pass length and speed pattern",
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print("\nFinished real-data calibrated closed-loop suite.")
    print(f"Outputs saved to: {out_dir}")
    print(f"Selected filter: {selected_filter}")


if __name__ == "__main__":
    main()
