from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_SPECS = {
    "exit_thickness_dev": {
        "label": "Exit thickness deviation",
        "raw_cols": ["Exit Thickness Deviation", "Exit Thickness Actual"],
        "exclude_patterns": ["Exit Thickness Actual", "Exit Thickness Deviation"],
    },
    "exit_tension_error": {
        "label": "Exit tension error",
        "raw_cols": ["Exit Tension Actual"],
        "exclude_patterns": ["Exit Tension Actual"],
    },
    "flatness_mean": {
        "label": "Flatness mean",
        "raw_cols": ["Actual Flatness Value"],
        "exclude_patterns": ["Actual Flatness Value", "flatness_mean", "flatness_std"],
    },
    "flatness_std": {
        "label": "Flatness standard deviation",
        "raw_cols": ["Actual Flatness Value"],
        "exclude_patterns": ["Actual Flatness Value", "flatness_mean", "flatness_std"],
    },
}


BASE_INPUT_CANDIDATES = [
    "rollforce_actual",
    "entry_tension_actual",
    "entry_tension_error",
    "entry_tension_setpoint",
    "exit_tension_setpoint",
    "entry_thickness_actual",
    "entry_thickness_dev",
    "entry_thickness_setpoint",
    "exit_thickness_setpoint",
    "reduction_ratio",
    "entry_speed",
    "exit_speed",
    "speed_avg",
    "speed_diff",
    "speed_dev_from_steady",
    "speed_change",
    "speed_accel",
    "speed_decel",
    "speed_abs_change",
    "sample_pos_norm",
    "phase_accel",
    "phase_steady",
    "phase_decel",
    "strip_width",
    "strip_length_norm",
    "offcenter_l",
    "offcenter_r",
    "offcenter_mean",
    "offcenter_diff",
    "radialforce_mean",
    "radialforce_std",
    "radialforce_left_mean",
    "radialforce_right_mean",
    "radialforce_diff",
]


@dataclass
class RealProcessDataset:
    frames: list[pd.DataFrame]
    pass_summary: pd.DataFrame
    data_dir: Path


def find_real_data_dir(data_dir: str | None = None) -> Path:
    if data_dir:
        path = Path(data_dir)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    raise ValueError("data_dir must be supplied explicitly")


def parse_pass_name(path: Path) -> tuple[float, float]:
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", path.stem)]
    if len(nums) >= 2:
        return nums[0], nums[1]
    if len(nums) == 1:
        return nums[0], nums[0]
    return np.nan, np.nan


def _as_float(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _safe_mean(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return frame[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _safe_std(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return frame[cols].apply(pd.to_numeric, errors="coerce").std(axis=1).fillna(0.0)


def load_real_process_data(data_dir: str | None = None) -> RealProcessDataset:
    root = find_real_data_dir(data_dir)
    files = sorted(root.glob("*.csv"), key=lambda p: parse_pass_name(p)[1])
    frames: list[pd.DataFrame] = []
    summaries = []

    for pass_index, path in enumerate(files):
        raw = pd.read_csv(path)
        entry_nominal, exit_nominal = parse_pass_name(path)
        flat_cols = [c for c in raw.columns if "Actual Flatness Value" in c]
        radial_cols = [c for c in raw.columns if "Actual Radialforce" in c]
        radial_left = radial_cols[: len(radial_cols) // 2]
        radial_right = radial_cols[len(radial_cols) // 2 :]

        df = pd.DataFrame(index=raw.index)
        df["pass_file"] = path.name
        df["pass_index"] = pass_index
        df["raw_timestamp"] = _as_float(raw, "Timestamp", np.nan)
        raw_dt = df["raw_timestamp"].diff()
        positive_dt = raw_dt[(raw_dt > 0.0) & np.isfinite(raw_dt)]
        fallback_dt = float(positive_dt.median()) if not positive_dt.empty else np.nan
        df["sample_dt_s"] = raw_dt.where(raw_dt > 0.0, np.nan).fillna(fallback_dt)
        df["entry_nominal_from_file"] = entry_nominal
        df["exit_nominal_from_file"] = exit_nominal
        # Preserve the exported acquisition clock for timebase audits.  The
        # legacy replay code remains sample-domain unless a caller explicitly
        # uses these columns, so this is backward compatible with frozen runs.
        timestamp = _as_float(raw, "Timestamp", np.nan)
        df["timestamp_raw"] = timestamp
        raw_dt = timestamp.diff()
        positive_dt = raw_dt[(raw_dt > 0.0) & np.isfinite(raw_dt)]
        median_dt = float(positive_dt.median()) if not positive_dt.empty else 0.5
        df["dt_raw_s"] = raw_dt.fillna(median_dt)
        df["dt_positive_s"] = raw_dt.where(raw_dt > 0.0, np.nan)
        df["entry_thickness_setpoint"] = _as_float(raw, "Entry Thickness Setpoint", entry_nominal)
        df["exit_thickness_setpoint"] = _as_float(raw, "Exit Thickness Setpoint", exit_nominal)
        df["entry_tension_setpoint"] = _as_float(raw, "Entry Tension Setpoint")
        df["exit_tension_setpoint"] = _as_float(raw, "Exit Tension Setpoint")
        df["strip_width"] = _as_float(raw, "Strip Width Setpoint")
        strip_length = _as_float(raw, "Strip Length Actual")
        df["strip_length_actual"] = strip_length
        df["strip_length_norm"] = strip_length / max(float(strip_length.max()), 1e-9)

        df["rollforce_actual"] = _as_float(raw, "Rollforce Actual")
        df["entry_tension_actual"] = _as_float(raw, "Entry Tension Actual")
        df["exit_tension_actual"] = _as_float(raw, "Exit Tension Actual")
        df["entry_tension_error"] = df["entry_tension_actual"] - df["entry_tension_setpoint"]
        df["exit_tension_error"] = df["exit_tension_actual"] - df["exit_tension_setpoint"]

        df["entry_thickness_actual"] = _as_float(raw, "Entry Thickness Actual")
        df["exit_thickness_actual"] = _as_float(raw, "Exit Thickness Actual")
        df["entry_thickness_dev"] = _as_float(raw, "Entry Thickness Deviation")
        df["exit_thickness_dev"] = _as_float(raw, "Exit Thickness Deviation")
        denom = df["entry_thickness_setpoint"].replace(0.0, np.nan)
        df["reduction_ratio"] = ((df["entry_thickness_setpoint"] - df["exit_thickness_setpoint"]) / denom).fillna(0.0)

        df["entry_speed"] = _as_float(raw, "Entry Speed")
        df["exit_speed"] = _as_float(raw, "Exit Speed")
        df["speed_avg"] = (df["entry_speed"] + df["exit_speed"]) / 2.0
        df["speed_diff"] = df["exit_speed"] - df["entry_speed"]
        n_rows = len(df)
        pos = np.arange(n_rows, dtype=float)
        phase_window = min(40, max(1, n_rows // 4))
        df["sample_pos_norm"] = pos / max(n_rows - 1, 1)
        df["phase_accel"] = (pos < phase_window).astype(float)
        df["phase_decel"] = (pos >= n_rows - phase_window).astype(float)
        df["phase_steady"] = 1.0 - df["phase_accel"] - df["phase_decel"]
        steady_slice = df["speed_avg"].iloc[phase_window : max(phase_window + 1, n_rows - phase_window)]
        if steady_slice.empty:
            steady_speed = float(df["speed_avg"].median())
        else:
            steady_speed = float(steady_slice.median())
        df["speed_dev_from_steady"] = df["speed_avg"] - steady_speed
        speed_change = df["speed_avg"].diff().fillna(0.0)
        df["speed_change"] = speed_change
        df["speed_accel"] = speed_change.clip(lower=0.0)
        df["speed_decel"] = (-speed_change).clip(lower=0.0)
        df["speed_abs_change"] = speed_change.abs()

        df["offcenter_l"] = _as_float(raw, "EDS Offcenter VKI L.H.")
        df["offcenter_r"] = _as_float(raw, "EDS Offcenter VKI R.H.")
        df["offcenter_mean"] = (df["offcenter_l"] + df["offcenter_r"]) / 2.0
        df["offcenter_diff"] = df["offcenter_r"] - df["offcenter_l"]

        df["flatness_mean"] = _safe_mean(raw, flat_cols)
        df["flatness_std"] = _safe_std(raw, flat_cols)
        df["radialforce_mean"] = _safe_mean(raw, radial_cols)
        df["radialforce_std"] = _safe_std(raw, radial_cols)
        df["radialforce_left_mean"] = _safe_mean(raw, radial_left)
        df["radialforce_right_mean"] = _safe_mean(raw, radial_right)
        df["radialforce_diff"] = df["radialforce_right_mean"] - df["radialforce_left_mean"]

        numeric_cols = [c for c in df.columns if c != "pass_file"]
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        # Forward fill only: a missing row is never completed from future rows.
        df[numeric_cols] = df[numeric_cols].ffill().fillna(0.0)
        frames.append(df)

        summaries.append(
            {
                "pass_file": path.name,
                "entry_nominal_from_file": entry_nominal,
                "exit_nominal_from_file": exit_nominal,
                "rows": len(df),
                "entry_thickness_setpoint_mean": float(df["entry_thickness_setpoint"].mean()),
                "exit_thickness_setpoint_mean": float(df["exit_thickness_setpoint"].mean()),
                "exit_tension_setpoint_mean": float(df["exit_tension_setpoint"].mean()),
                "rollforce_mean": float(df["rollforce_actual"].mean()),
                "entry_speed_mean": float(df["entry_speed"].mean()),
                "exit_speed_mean": float(df["exit_speed"].mean()),
                "exit_thickness_dev_std": float(df["exit_thickness_dev"].std()),
                "exit_tension_error_std": float(df["exit_tension_error"].std()),
                "flatness_mean_std": float(df["flatness_mean"].std()),
            }
        )
    return RealProcessDataset(frames=frames, pass_summary=pd.DataFrame(summaries), data_dir=root)


def candidate_inputs_for_target(target: str, include_radial_channels: bool = False) -> list[str]:
    inputs = list(BASE_INPUT_CANDIDATES)
    if target == "exit_tension_error":
        inputs += ["entry_thickness_dev", "exit_thickness_setpoint"]
    elif target == "exit_thickness_dev":
        inputs += ["entry_tension_actual", "exit_tension_setpoint"]
    elif target in {"flatness_mean", "flatness_std"}:
        inputs += ["exit_thickness_dev", "entry_tension_error", "exit_tension_error"]

    # Avoid direct leakage from the current target measurement as an exogenous input.
    exclusions = set()
    if target == "exit_tension_error":
        exclusions.update(["exit_tension_actual", "exit_tension_error"])
    elif target == "exit_thickness_dev":
        exclusions.update(["exit_thickness_actual", "exit_thickness_dev"])
    elif target in {"flatness_mean", "flatness_std"}:
        exclusions.update(["flatness_mean", "flatness_std"])
    out = []
    for col in inputs:
        if col not in exclusions and col not in out:
            out.append(col)
    return out


def split_frame_by_time(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(df)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : min(n_train + n_val, n)].copy()
    test = df.iloc[min(n_train + n_val, n) :].copy()
    return train, val, test
