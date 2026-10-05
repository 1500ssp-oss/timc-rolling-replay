from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import controller_replay as runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_root", type=Path)
    args = parser.parse_args()
    out = args.out_root.resolve()
    metrics = pd.read_csv(out / "stage_allocation_ablation_metrics.csv")
    if "pass_id" in metrics and "condition_id" in metrics:
        metrics.loc[metrics["pass_id"].eq("P06"), "condition_id"] = (
            "0.628->0.458 acceleration"
        )
        metrics.to_csv(
            out / "stage_allocation_ablation_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
    step_path = out / "stage_allocation_ablation_step_logs.csv.gz"
    if step_path.exists():
        steps = pd.read_csv(step_path)
    else:
        steps = pd.DataFrame(
            columns=["block", "scenario", "control", "pass_id", "repeat_idx", "k"]
        )
    metric_cols = [
        column
        for column in [
            "composite_normalized_RMS", "TV_L_per_100m", "S_out",
            "S_excess_mean", "S_excess_cvar95", "amplitude_projection_ratio",
            "amplitude_active_speed_ratio", "amplitude_active_gap_ratio",
            "amplitude_active_shape_ratio", "raw_to_applied_L2_mean",
            "output_clip_ratio", "pid_aw_vector_freeze_count",
            "pid_aw_speed_count", "pid_aw_gap_count", "pid_aw_shape_count",
        ]
        if column in metrics.columns
    ]
    runner.condition_weighted_summary(
        metrics, ["block", "scenario", "control", "comparison_base"], metric_cols
    ).to_csv(
        out / "stage_allocation_ablation_condition_weighted_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    runner.paired_effect_summary(metrics, steps).to_csv(
        out / "stage_allocation_ablation_paired_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
