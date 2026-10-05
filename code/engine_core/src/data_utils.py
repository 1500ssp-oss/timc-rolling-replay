
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def find_col(pattern, cols):
    key = pattern.lower().replace(" ", "")
    for col in cols:
        if key in col.lower().replace(" ", ""):
            return col
    return None


def load_rolling_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = df.columns

    feature_names = [
        "Rollforce Actual",
        "Entry Tension Actual",
        "Entry Thickness Actual",
        "Entry Speed",
        "Exit Speed",
        "Entry Thickness Deviation",
        "Exit Thickness Deviation",
    ]

    used_features = []
    missing = []
    for name in feature_names:
        col = find_col(name, cols)
        if col is None:
            missing.append(name)
        else:
            used_features.append(col)
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {missing}")

    tension_col = find_col("Exit Tension Actual", cols)
    thickness_col = find_col("Exit Thickness Actual", cols)
    flatness_cols = [col for col in cols if "Actual Flatness Value" in col]

    if tension_col is None:
        raise ValueError("Exit Tension Actual column was not found")
    if thickness_col is None:
        raise ValueError("Exit Thickness Actual column was not found")
    if not flatness_cols:
        raise ValueError("No Actual Flatness Value* columns were found")

    X = df[used_features].to_numpy(dtype=float)
    y_tension = df[tension_col].to_numpy(dtype=float).reshape(-1, 1)
    y_thickness = df[thickness_col].to_numpy(dtype=float).reshape(-1, 1)
    y_flatness = df[flatness_cols].mean(axis=1).to_numpy(dtype=float).reshape(-1, 1)
    Y = np.hstack([y_tension, y_thickness, y_flatness])

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1) | np.isinf(X).any(axis=1) | np.isinf(Y).any(axis=1))
    X = X[mask]
    Y = Y[mask]

    var_x = X.var(axis=0)
    keep = var_x > 1e-12
    removed = [used_features[i] for i, ok in enumerate(keep) if not ok]
    X = X[:, keep]
    used_features = [c for c, ok in zip(used_features, keep) if ok]

    return X, Y, used_features, [tension_col, thickness_col, "FlatnessMean"], removed


def create_sequences(X, Y, seq_len: int, pred_steps: int):
    Xs, Ys = [], []
    n = len(X) - seq_len - pred_steps + 1
    if n <= 0:
        raise ValueError("Insufficient rows; reduce seq_len or pred_steps")
    for i in range(n):
        Xs.append(X[i : i + seq_len])
        Ys.append(Y[i + seq_len : i + seq_len + pred_steps])
    return np.asarray(Xs), np.asarray(Ys)


def time_order_split(X_seq, Y_seq, train_ratio=0.70, val_ratio=0.15):
    n = len(X_seq)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, Y_train = X_seq[:n_train], Y_seq[:n_train]
    X_val, Y_val = X_seq[n_train:n_train+n_val], Y_seq[n_train:n_train+n_val]
    X_test, Y_test = X_seq[n_train+n_val:], Y_seq[n_train+n_val:]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def fit_transform_time_series(X, Y, train_end_index):
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    scaler_X.fit(X[:train_end_index])
    scaler_Y.fit(Y[:train_end_index])
    return scaler_X.transform(X), scaler_Y.transform(Y), scaler_X, scaler_Y


def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
