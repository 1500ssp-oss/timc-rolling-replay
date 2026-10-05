"""Global-seed stability audit for the canonical replay.

All policies share the same seeded segment-start state perturbation for a
given pass; canonical measurement noise is zero.  This script shifts the
global seed by two offsets and reports the largest condition-weighted change
of the headline metrics, bounding sensitivity to that seeded initialization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import batch_archive as archive  # noqa: E402
import controller_replay as runner  # noqa: E402

METRIC_COLS = [
    "composite_normalized_RMS",
    "S_out",
    "S_excess_mean",
    "S_excess_cvar95",
    "TV_L_per_100m",
    "amplitude_projection_ratio",
]


def condition_weighted(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    if "pass_id" in metrics and "condition_id" in metrics:
        metrics.loc[metrics["pass_id"].eq("P06"), "condition_id"] = "0.628->0.458 acceleration"
    file_level = metrics.groupby(["control", "condition_id", "file"], as_index=False)[METRIC_COLS].mean()
    condition_level = file_level.groupby(["control", "condition_id"], as_index=False)[METRIC_COLS].mean()
    return condition_level.groupby("control", as_index=False)[METRIC_COLS].mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--offsets", default="1000000,2000000")
    args = parser.parse_args()
    offsets = [int(item) for item in args.offsets.split(",") if item.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runner.protocol.PROJECT_DIR = archive.FULL_PROJECT
    suite, replay = runner.protocol.load_suite_modules()
    archive.BATCH_ROOT = Path(args.batch_root)
    all_passes, _ = archive.load_batch_passes(runner.protocol, suite, replay)
    data2 = sorted(
        [rp for rp in all_passes if rp.pass_id in archive.CANONICAL_DATA2_IDS],
        key=lambda rp: archive.CANONICAL_SEED_INDEX[rp.pass_id],
    )
    manifest = json.loads(
        (archive.CANONICAL_RUN / "03_nominal" / "stage_allocation_ablation_manifest.json").read_text(encoding="utf-8")
    )
    locked_pid = runner.protocol.PidParams(**manifest["locked_pid"])
    envelope = pd.read_csv(
        archive.CANONICAL_RUN / "03_nominal" / "audit" / "data1_stage_quantile_envelope_p025_p975.csv"
    )
    configs = [
        runner.ExposureConfig(**item)
        for item in json.loads(
            (ROOT.parent / "config" / "selected_controller_configs.json").read_text(encoding="utf-8")
        )
    ]

    canonical_metrics = pd.read_csv(ROOT.parent / "outputs" / "canonical" / "03_nominal" / "stage_allocation_ablation_metrics.csv")
    canonical_cw = condition_weighted(canonical_metrics).set_index("control")

    rows = []
    for offset in offsets:
        metric_rows = []
        for config in configs:
            for rp in data2:
                seed = runner.RNG_SEED + 2000 * (archive.CANONICAL_SEED_INDEX[rp.pass_id] + 1) + offset
                curve, diag = runner.run_exposure_control(suite, rp, config, seed, locked_pid, 0)
                curve["pass_id"] = rp.pass_id
                curve["control"] = config.label
                metric_rows.append(runner.suite_metrics_with_config(suite, rp, curve, diag, config, envelope))
        cw = condition_weighted(pd.DataFrame(metric_rows)).set_index("control")
        for metric in METRIC_COLS:
            delta = (cw[metric] - canonical_cw[metric]).abs()
            rows.append({
                "seed_offset": offset,
                "metric": metric,
                "max_abs_delta_pct_of_canonical": float(100.0 * delta.max() / max(canonical_cw[metric].max(), 1e-12)),
                "max_abs_delta_abs": float(delta.max()),
                "worst_policy": str(delta.idxmax()),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "seed_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
