from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "RMS_c": ("composite_normalized_RMS", "penalty"),
    "TV": ("control_total_variation", "reduction"),
    "S_excess_mean": ("S_excess_mean", "reduction"),
    "S_excess_p95": ("S_excess_p95", "reduction"),
}


def effect(c7: np.ndarray, baseline: np.ndarray, direction: str) -> np.ndarray:
    if direction == "penalty":
        return 100.0 * (c7 - baseline) / baseline
    return 100.0 * (baseline - c7) / baseline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    out_dir = Path(args.out_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.metrics_csv)
    rng = np.random.default_rng(args.seed)
    rows = []
    for scenario, scenario_data in data.groupby("scenario", sort=True):
        c7_name = f"C7-{scenario}"
        condition = scenario_data.groupby(["condition_id", "control"], as_index=False)[list(v[0] for v in METRICS.values())].mean()
        pivot_by_metric = {column: condition.pivot(index="condition_id", columns="control", values=column) for column, _ in METRICS.values()}
        controls = set(condition["control"])
        for family in ["PID", "ADRC"]:
            baseline = f"{family}-{scenario}"
            if c7_name not in controls or baseline not in controls:
                continue
            for metric, (column, direction) in METRICS.items():
                pivot = pivot_by_metric[column][[c7_name, baseline]].dropna()
                c7 = pivot[c7_name].to_numpy(dtype=float)
                base = pivot[baseline].to_numpy(dtype=float)
                draws = rng.integers(0, len(c7), size=(args.bootstrap, len(c7)))
                boot = effect(c7[draws].mean(axis=1), base[draws].mean(axis=1), direction)
                point = float(effect(np.array([c7.mean()]), np.array([base.mean()]), direction)[0])
                rows.append(
                    {
                        "scenario": scenario,
                        "baseline": family,
                        "metric": metric,
                        "direction": direction,
                        "point_effect_pct": point,
                        "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "n_conditions": int(len(c7)),
                        "invalid_c7": int(scenario_data[scenario_data["control"].eq(c7_name)]["invalid_command_count"].sum()),
                        "invalid_baseline": int(scenario_data[scenario_data["control"].eq(baseline)]["invalid_command_count"].sum()),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "timestamp_matched_stress_clustered_effects.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
