
import numpy as np


class PID:
    def __init__(self, kp, ki, kd, dt, umin, umax, anti_windup=True, anti_windup_mode="vectorwise"):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.dt = dt
        self.umin = np.asarray(umin, dtype=float)
        self.umax = np.asarray(umax, dtype=float)
        self.anti_windup = anti_windup
        self.anti_windup_mode = str(anti_windup_mode)
        self.integral = np.zeros_like(self.kp)
        self.prev_error = np.zeros_like(self.kp)
        self.last_saturation_mask = np.zeros_like(self.kp, dtype=bool)
        self.vector_freeze_count = 0
        self.channel_freeze_count = np.zeros_like(self.kp, dtype=int)

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0

    def step(self, error, dt=None):
        error = np.asarray(error, dtype=float)
        step_dt = float(self.dt if dt is None else dt)
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError(f"PID requires a positive finite dt, received {step_dt!r}")
        derivative = (error - self.prev_error) / max(step_dt, 1e-9)
        trial_integral = self.integral + error * step_dt
        u_unsat = self.kp * error + self.ki * trial_integral + self.kd * derivative
        u = np.clip(u_unsat, self.umin, self.umax)
        saturated = ~np.isclose(u, u_unsat)
        self.last_saturation_mask = saturated
        if np.any(saturated):
            self.vector_freeze_count += 1
            self.channel_freeze_count += saturated.astype(int)
        if not self.anti_windup:
            self.integral = trial_integral
        elif self.anti_windup_mode == "vectorwise":
            if not np.any(saturated):
                self.integral = trial_integral
        elif self.anti_windup_mode == "per_channel":
            self.integral = np.where(saturated, self.integral, trial_integral)
        else:
            raise ValueError(f"Unknown anti_windup_mode={self.anti_windup_mode!r}")
        self.prev_error = error
        return u
