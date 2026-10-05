from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOCKED_GAMMA = {
    "PID": 0.60,
    "OC-PID": 0.95,
    "C2": 0.80,
    "C7-Core": 0.80,
}


def policy_name(label: str) -> str:
    if label.startswith("PID"):
        return "PID"
    if label.startswith("ADRC") or label.startswith("OC-PID"):
        return "OC-PID"
    if label.startswith("C2"):
        return "C2"
    if label.startswith("C7"):
        return "C7-Core"
    raise ValueError(f"Unknown policy label: {label}")


def condition_weighted(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    file_level = data.groupby(
        ["policy", "rate_multiplier", "effective_gamma", "condition_id", "file"],
        as_index=False,
    )[metrics].mean()
    condition_level = file_level.groupby(
        ["policy", "rate_multiplier", "effective_gamma", "condition_id"],
        as_index=False,
    )[metrics].mean()
    return condition_level.groupby(
        ["policy", "rate_multiplier", "effective_gamma"], as_index=False
    )[metrics].mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()

    source_columns = [
        "control",
        "scenario",
        "outer_du_scale",
        "condition_id",
        "file",
        "TV_L_per_100m",
        "S_excess_mean",
        "amplitude_projection_ratio",
        "sat_ratio_logged",
    ]
    data = pd.read_csv(args.metrics, usecols=source_columns)
    data["policy"] = data["control"].map(policy_name)
    data["rate_multiplier"] = data["scenario"].str.extract(
        r"shell_rate_([0-9.]+)", expand=False
    ).astype(float)
    data["effective_gamma"] = data["outer_du_scale"].astype(float)

    expected_gamma = data.apply(
        lambda row: LOCKED_GAMMA[row["policy"]] * row["rate_multiplier"], axis=1
    )
    gamma_ok = np.allclose(data["effective_gamma"], expected_gamma, atol=1e-12, rtol=0)
    if not gamma_ok:
        raise RuntimeError("Rate configurations are not relative to the locked policy gamma")

    metrics = [
        "TV_L_per_100m",
        "S_excess_mean",
        "amplitude_projection_ratio",
        "sat_ratio_logged",
    ]
    result = condition_weighted(data, metrics).rename(
        columns={
            "policy": "Policy",
            "rate_multiplier": "Rate multiplier",
            "effective_gamma": "Effective gamma",
            "TV_L_per_100m": "TV/100 m",
            "S_excess_mean": "Mean two-sided band distance",
            "amplitude_projection_ratio": "Amplitude active",
            "sat_ratio_logged": "Slew active",
        }
    )
    order = {"PID": 0, "OC-PID": 1, "C2": 2, "C7-Core": 3}
    result["_order"] = result["Policy"].map(order)
    result = result.sort_values(["_order", "Rate multiplier"]).drop(columns="_order")

    canonical = pd.read_csv(args.canonical).copy()
    canonical["Policy"] = canonical["policy"].replace({"C7": "C7-Core"})
    anchor = result[np.isclose(result["Rate multiplier"], 1.0)].set_index("Policy")
    canonical = canonical.set_index("Policy")
    anchor_checks = {
        "TV/100 m": "TV_L_per_100m",
        "Mean two-sided band distance": "S_excess_mean",
        "Amplitude active": "amplitude_projection_ratio",
    }
    differences: dict[str, dict[str, float]] = {}
    for policy in LOCKED_GAMMA:
        differences[policy] = {}
        for table_metric, canonical_metric in anchor_checks.items():
            differences[policy][table_metric] = float(
                anchor.loc[policy, table_metric] - canonical.loc[policy, canonical_metric]
            )
    max_abs_difference = max(
        abs(value) for policy in differences.values() for value in policy.values()
    )
    if max_abs_difference > 1e-12:
        raise RuntimeError(
            f"Rate multiplier 1.00 does not reproduce canonical results: {max_abs_difference}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    audit = {
        "definition": "effective_gamma = locked_policy_gamma * rate_multiplier",
        "locked_gamma": LOCKED_GAMMA,
        "rows": int(len(result)),
        "source_replays": int(len(data)),
        "conditions": int(data["condition_id"].nunique()),
        "files": int(data["file"].nunique()),
        "multiplier_1_max_abs_difference": max_abs_difference,
        "multiplier_1_differences": differences,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
