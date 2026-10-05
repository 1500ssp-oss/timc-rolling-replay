from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TARGET_COLS = ["exit_tension_error", "exit_thickness_dev", "flatness_error"]
TARGET_LABELS = {
    "exit_tension_error": "Tension error",
    "exit_thickness_dev": "Thickness deviation",
    "flatness_error": "Flatness error",
}

TENSION_BASE = [
    "exit_tension_error",
    "entry_tension_error",
    "exit_tension_setpoint",
    "entry_tension_setpoint",
    "speed_avg",
    "speed_diff",
]
THICKNESS_BASE = [
    "exit_thickness_dev",
    "entry_thickness_dev",
    "entry_thickness_setpoint",
    "exit_thickness_setpoint",
    "reduction_ratio",
    "rollforce_dev",
]
FLATNESS_BASE = [
    "flatness_error",
    "flatness_std",
    "radialforce_mean",
    "radialforce_std",
    "offcenter_mean",
    "offcenter_diff",
]
CONTEXT_COLS = ["pass_value", "strip_width", "strip_length_norm"]


@dataclass
class AblationDataset:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    Y_test_raw: np.ndarray
    Y_test_anchor_raw: np.ndarray
    pass_test: np.ndarray
    feature_cols: list[str]
    group_indices: dict[str, list[int]]
    target_cols: list[str]
    scaler_X: StandardScaler
    scaler_Y: StandardScaler
    split: dict[str, list[str]]
    pass_summary: pd.DataFrame


def find_data1_dir(data_dir: str | None = None) -> Path:
    if data_dir:
        path = Path(data_dir)
        if path.exists():
            return path
        raise FileNotFoundError(path)

    raise ValueError("data_dir must be supplied explicitly")


def _as_float_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _causal_kalman_1d(values: np.ndarray, q_scale: float = 0.010, r_scale: float = 0.080) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    finite = values[np.isfinite(values)]
    fallback = float(np.nanmean(finite)) if len(finite) else 0.0
    values = np.where(np.isfinite(values), values, fallback)
    var = float(np.var(values))
    q = max(var * q_scale, 1e-9)
    r = max(var * r_scale, 1e-9)
    x = values[0]
    p = var + 1e-6
    out = np.empty_like(values, dtype=float)
    for i, z in enumerate(values):
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1.0 - k) * p
        out[i] = x
    return out


def load_multi_pass_frames(data_dir: str | None = None) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    root = find_data1_dir(data_dir)
    files = sorted(root.glob("*.csv"), key=lambda p: float(p.stem))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {root}")

    frames: list[pd.DataFrame] = []
    summaries = []
    for pass_index, path in enumerate(files):
        raw = pd.read_csv(path)
        df = pd.DataFrame(index=raw.index)
        pass_value = float(path.stem)

        flat_cols = [c for c in raw.columns if "Actual Flatness Value" in c]
        radial_cols = [c for c in raw.columns if "Actual Radialforce" in c]
        flat_values = raw[flat_cols].apply(pd.to_numeric, errors="coerce") if flat_cols else pd.DataFrame(index=raw.index)
        radial_values = raw[radial_cols].apply(pd.to_numeric, errors="coerce") if radial_cols else pd.DataFrame(index=raw.index)

        df["pass_file"] = path.name
        df["pass_index"] = pass_index
        df["pass_value"] = pass_value
        df["entry_tension_setpoint"] = _as_float_series(raw, "Entry Tension Setpoint")
        df["exit_tension_setpoint"] = _as_float_series(raw, "Exit Tension Setpoint")
        df["entry_thickness_setpoint"] = _as_float_series(raw, "Entry Thickness Setpoint", pass_value)
        df["exit_thickness_setpoint"] = _as_float_series(raw, "Exit Thickness Setpoint", pass_value)
        denom = df["entry_thickness_setpoint"].replace(0, np.nan)
        df["reduction_ratio"] = ((df["entry_thickness_setpoint"] - df["exit_thickness_setpoint"]) / denom).fillna(0.0)
        df["strip_width"] = _as_float_series(raw, "Strip Width Setpoint")
        strip_length = _as_float_series(raw, "Strip Length Actual")
        df["strip_length_norm"] = strip_length / max(float(strip_length.max()), 1e-9)

        entry_tension = _as_float_series(raw, "Entry Tension Actual")
        exit_tension = _as_float_series(raw, "Exit Tension Actual")
        df["entry_tension_error"] = entry_tension - df["entry_tension_setpoint"]
        df["exit_tension_error"] = exit_tension - df["exit_tension_setpoint"]
        df["entry_thickness_dev"] = _as_float_series(raw, "Entry Thickness Deviation")
        df["exit_thickness_dev"] = _as_float_series(raw, "Exit Thickness Deviation")
        df["speed_avg"] = (_as_float_series(raw, "Entry Speed") + _as_float_series(raw, "Exit Speed")) / 2.0
        df["speed_diff"] = _as_float_series(raw, "Exit Speed") - _as_float_series(raw, "Entry Speed")

        rollforce = _as_float_series(raw, "Rollforce Actual")
        df["rollforce_actual"] = rollforce
        df["rollforce_dev"] = rollforce - float(rollforce.mean())
        df["flatness_error"] = flat_values.mean(axis=1) if flat_cols else 0.0
        df["flatness_std"] = flat_values.std(axis=1).fillna(0.0) if flat_cols else 0.0
        df["radialforce_mean"] = radial_values.mean(axis=1) if radial_cols else 0.0
        df["radialforce_std"] = radial_values.std(axis=1).fillna(0.0) if radial_cols else 0.0
        off_l = _as_float_series(raw, "EDS Offcenter VKI L.H.")
        off_r = _as_float_series(raw, "EDS Offcenter VKI R.H.")
        df["offcenter_mean"] = (off_l + off_r) / 2.0
        df["offcenter_diff"] = off_r - off_l

        smooth_cols = [
            "exit_tension_error",
            "entry_tension_error",
            "exit_thickness_dev",
            "entry_thickness_dev",
            "flatness_error",
            "flatness_std",
            "rollforce_dev",
            "speed_avg",
            "speed_diff",
            "radialforce_mean",
            "radialforce_std",
            "offcenter_mean",
            "offcenter_diff",
        ]
        for col in smooth_cols:
            df[f"{col}_ekf"] = _causal_kalman_1d(df[col].to_numpy())

        numeric_cols = [c for c in df.columns if c != "pass_file"]
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        # Forward fill only: a missing row is never completed from future rows.
        df[numeric_cols] = df[numeric_cols].ffill().fillna(0.0)
        frames.append(df)

        summaries.append(
            {
                "pass_file": path.name,
                "pass_value": pass_value,
                "rows": len(df),
                "entry_thickness_setpoint": float(df["entry_thickness_setpoint"].mean()),
                "exit_thickness_setpoint": float(df["exit_thickness_setpoint"].mean()),
                "exit_tension_setpoint": float(df["exit_tension_setpoint"].mean()),
                "tension_error_std": float(df["exit_tension_error"].std()),
                "thickness_dev_std": float(df["exit_thickness_dev"].std()),
                "flatness_error_std": float(df["flatness_error"].std()),
                "speed_mean": float(df["speed_avg"].mean()),
            }
        )

    return frames, pd.DataFrame(summaries)


def _resolve_feature_cols(use_ekf: bool, feature_mode: str) -> tuple[list[str], dict[str, list[str]]]:
    def maybe_ekf(cols: list[str]) -> list[str]:
        if not use_ekf:
            return cols
        smoothed = {
            "exit_tension_error",
            "entry_tension_error",
            "exit_thickness_dev",
            "entry_thickness_dev",
            "flatness_error",
            "flatness_std",
            "rollforce_dev",
            "speed_avg",
            "speed_diff",
            "radialforce_mean",
            "radialforce_std",
            "offcenter_mean",
            "offcenter_diff",
        }
        return [f"{c}_ekf" if c in smoothed else c for c in cols]

    tension = maybe_ekf(TENSION_BASE) + CONTEXT_COLS
    thickness = maybe_ekf(THICKNESS_BASE) + CONTEXT_COLS
    flatness = maybe_ekf(FLATNESS_BASE) + CONTEXT_COLS
    context = CONTEXT_COLS

    if feature_mode == "tension_only":
        groups = {"tension": tension, "thickness": context, "flatness": context}
    elif feature_mode == "tension_thickness":
        groups = {"tension": tension, "thickness": thickness, "flatness": context}
    elif feature_mode == "full":
        groups = {"tension": tension, "thickness": thickness, "flatness": flatness}
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    feature_cols: list[str] = []
    for cols in groups.values():
        for col in cols:
            if col not in feature_cols:
                feature_cols.append(col)
    return feature_cols, groups


def _default_split(pass_files: list[str], holdout_pass: str | None = None) -> dict[str, list[str]]:
    n = len(pass_files)
    if n < 5:
        raise ValueError("At least five passes are recommended for pass-wise ablation.")
    if holdout_pass:
        if holdout_pass not in pass_files:
            raise ValueError(f"holdout_pass={holdout_pass!r} not found in available passes: {pass_files}")
        rest = [p for p in pass_files if p != holdout_pass]
        n_val = max(1, round(len(rest) * 0.15))
        return {
            "train": rest[: len(rest) - n_val],
            "val": rest[len(rest) - n_val :],
            "test": [holdout_pass],
        }
    n_test = max(2, round(n * 0.15))
    n_val = max(2, round(n * 0.15))
    return {
        "train": pass_files[: n - n_val - n_test],
        "val": pass_files[n - n_val - n_test : n - n_test],
        "test": pass_files[n - n_test :],
    }


def _make_sequences_for_frames(
    frames: list[pd.DataFrame],
    pass_names: set[str],
    feature_cols: list[str],
    seq_len: int,
    pred_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xs, Ys, anchors, passes = [], [], [], []
    for df in frames:
        pass_name = str(df["pass_file"].iloc[0])
        if pass_name not in pass_names:
            continue
        X = df[feature_cols].to_numpy(dtype=float)
        Y = df[TARGET_COLS].to_numpy(dtype=float)
        n = len(df) - seq_len - pred_steps + 1
        for i in range(max(n, 0)):
            Xs.append(X[i : i + seq_len])
            Ys.append(Y[i + seq_len : i + seq_len + pred_steps])
            anchors.append(Y[i + seq_len - 1])
            passes.append(pass_name)
    if not Xs:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=float),
            np.empty((0, pred_steps, len(TARGET_COLS)), dtype=float),
            np.empty((0, len(TARGET_COLS)), dtype=float),
            np.empty((0,), dtype=object),
        )
    return np.asarray(Xs, dtype=float), np.asarray(Ys, dtype=float), np.asarray(anchors, dtype=float), np.asarray(passes, dtype=object)


def _make_sequences_within_pass(
    frames: list[pd.DataFrame],
    feature_cols: list[str],
    seq_len: int,
    pred_steps: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[str, list[str]], list[pd.DataFrame]]:
    buckets = {
        "train": ([], [], [], []),
        "val": ([], [], [], []),
        "test": ([], [], [], []),
    }
    split = {"train": [], "val": [], "test": []}
    train_row_parts = []
    for df in frames:
        pass_name = str(df["pass_file"].iloc[0])
        X = df[feature_cols].to_numpy(dtype=float)
        Y = df[TARGET_COLS].to_numpy(dtype=float)
        nseq = len(df) - seq_len - pred_steps + 1
        if nseq <= 0:
            continue
        n_train = max(1, int(nseq * train_ratio))
        n_val = max(1, int(nseq * val_ratio))
        train_row_end = min(len(df), n_train + seq_len + pred_steps)
        train_row_parts.append(df.iloc[:train_row_end])
        ranges = {
            "train": range(0, n_train),
            "val": range(n_train, min(n_train + n_val, nseq)),
            "test": range(min(n_train + n_val, nseq), nseq),
        }
        for split_name, indices in ranges.items():
            split[split_name].append(pass_name)
            Xs, Ys, anchors, passes = buckets[split_name]
            for i in indices:
                Xs.append(X[i : i + seq_len])
                Ys.append(Y[i + seq_len : i + seq_len + pred_steps])
                anchors.append(Y[i + seq_len - 1])
                passes.append(pass_name)

    out = {}
    for name, (Xs, Ys, anchors, passes) in buckets.items():
        if Xs:
            out[name] = (
                np.asarray(Xs, dtype=float),
                np.asarray(Ys, dtype=float),
                np.asarray(anchors, dtype=float),
                np.asarray(passes, dtype=object),
            )
        else:
            out[name] = (
                np.empty((0, seq_len, len(feature_cols)), dtype=float),
                np.empty((0, pred_steps, len(TARGET_COLS)), dtype=float),
                np.empty((0, len(TARGET_COLS)), dtype=float),
                np.empty((0,), dtype=object),
            )
    return out, split, train_row_parts


def prepare_ablation_dataset(
    data_dir: str | None = None,
    seq_len: int = 20,
    pred_steps: int = 10,
    use_ekf: bool = True,
    feature_mode: str = "full",
    split_mode: str = "within_pass",
    holdout_pass: str | None = None,
) -> AblationDataset:
    frames, summary = load_multi_pass_frames(data_dir)
    pass_files = [str(df["pass_file"].iloc[0]) for df in frames]
    feature_cols, group_cols = _resolve_feature_cols(use_ekf=use_ekf, feature_mode=feature_mode)

    if split_mode == "within_pass":
        seq_parts, split, train_row_parts = _make_sequences_within_pass(frames, feature_cols, seq_len, pred_steps)
        train_rows = pd.concat(train_row_parts, axis=0)
    elif split_mode == "pass_holdout":
        split = _default_split(pass_files, holdout_pass=holdout_pass)
        train_rows = pd.concat([df[df["pass_file"].isin(split["train"])] for df in frames], axis=0)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    scaler_X.fit(train_rows[feature_cols].to_numpy(dtype=float))
    scaler_Y.fit(train_rows[TARGET_COLS].to_numpy(dtype=float))

    if split_mode == "within_pass":
        X_train, Y_train_raw, _, _ = seq_parts["train"]
        X_val, Y_val_raw, _, _ = seq_parts["val"]
        X_test, Y_test_raw, Y_test_anchor_raw, pass_test = seq_parts["test"]
    else:
        X_train, Y_train_raw, _, _ = _make_sequences_for_frames(frames, set(split["train"]), feature_cols, seq_len, pred_steps)
        X_val, Y_val_raw, _, _ = _make_sequences_for_frames(frames, set(split["val"]), feature_cols, seq_len, pred_steps)
        X_test, Y_test_raw, Y_test_anchor_raw, pass_test = _make_sequences_for_frames(frames, set(split["test"]), feature_cols, seq_len, pred_steps)

    def tx(X: np.ndarray) -> np.ndarray:
        shape = X.shape
        return scaler_X.transform(X.reshape(-1, shape[-1])).reshape(shape)

    def ty(Y: np.ndarray) -> np.ndarray:
        shape = Y.shape
        return scaler_Y.transform(Y.reshape(-1, shape[-1])).reshape(shape)

    group_indices = {
        name: [feature_cols.index(col) for col in cols if col in feature_cols]
        for name, cols in group_cols.items()
    }
    return AblationDataset(
        X_train=tx(X_train),
        Y_train=ty(Y_train_raw),
        X_val=tx(X_val),
        Y_val=ty(Y_val_raw),
        X_test=tx(X_test),
        Y_test=ty(Y_test_raw),
        Y_test_raw=Y_test_raw,
        Y_test_anchor_raw=Y_test_anchor_raw,
        pass_test=pass_test,
        feature_cols=feature_cols,
        group_indices=group_indices,
        target_cols=TARGET_COLS,
        scaler_X=scaler_X,
        scaler_Y=scaler_Y,
        split=split,
        pass_summary=summary,
    )
