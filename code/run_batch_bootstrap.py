"""Batch-clustered bootstrap sensitivity for the paired C7-Core effects.

The eight operating conditions nest inside four production batches
(II, III, V, VI).  The canonical interval resamples conditions first; this
script additionally resamples the four batches first, then conditions and
files inside the sampled batches, and reports the resulting 95% intervals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

import batch_archive as archive  # noqa: E402
import implementation_sensitivity as impl  # noqa: E402

CANON = ROOT / "outputs" / "canonical" / "03_nominal"
METRICS = ["composite_normalized_RMS", "TV_L_per_100m", "S_excess_mean", "S_excess_cvar95"]
N_BOOT = 10_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(CANON / "stage_allocation_ablation_metrics.csv")
    metrics["policy"] = metrics["control"].map(impl.policy_name)
    if "pass_id" in metrics and "condition_id" in metrics:
        metrics = metrics.copy()
        metrics.loc[metrics["pass_id"].eq("P06"), "condition_id"] = "0.628->0.458 acceleration"
    inventory = pd.read_csv(ROOT / "outputs" / "expanded" / "production_batch_map.csv")
    metrics = metrics.merge(inventory[["pass_id", "batch"]], on="pass_id", how="left")
    metrics = metrics[metrics.pass_id.isin(archive.CANONICAL_DATA2_IDS)]

    rng = np.random.default_rng(20260712)
    rows = []
    for metric in METRICS:
        for baseline in ["PID", "OC-PID", "C2"]:
            file_level = metrics.groupby(["batch", "condition_id", "file", "policy"], as_index=False)[metric].mean()
            wide = file_level.pivot(index=["batch", "condition_id", "file"], columns="policy", values=metric).dropna(subset=[baseline, "C7-Core"])
            wide["effect"] = 100.0 * (wide["C7-Core"] - wide[baseline]) / wide[baseline]
            by_batch = {
                str(b): {str(c): g["effect"].to_numpy(float) for c, g in group.groupby("condition_id")}
                for b, group in wide.reset_index().groupby("batch")
            }
            batches = np.array(sorted(by_batch), dtype=object)
            point = float(wide["effect"].groupby(level="condition_id").mean().mean())
            draws = np.empty(N_BOOT, dtype=float)
            for b in range(N_BOOT):
                sampled_batches = rng.choice(batches, size=len(batches), replace=True)
                values = []
                for batch in sampled_batches:
                    conditions = np.array(sorted(by_batch[str(batch)]), dtype=object)
                    for condition in rng.choice(conditions, size=len(conditions), replace=True):
                        files = by_batch[str(batch)][str(condition)]
                        values.append(float(rng.choice(files, size=len(files), replace=True).mean()))
                draws[b] = float(np.mean(values))
            rows.append({
                "metric": metric,
                "comparison": f"C7 vs {baseline}",
                "point_pct": point,
                "condition_bootstrap_ci_low_pct": None,
                "condition_bootstrap_ci_high_pct": None,
                "batch_clustered_ci_low_pct": float(np.percentile(draws, 2.5)),
                "batch_clustered_ci_high_pct": float(np.percentile(draws, 97.5)),
                "batch_clustered_se_pct": float(draws.std(ddof=1)),
                "n_batches": int(len(batches)),
                "bootstrap_draws": N_BOOT,
            })
    frame = pd.DataFrame(rows)
    # Fill the condition-level bootstrap CI from the paired bootstrap file for
    # side-by-side comparison.
    paired = ROOT / "results" / "paired_effect_bootstrap.csv"
    if paired.exists():
        paired_df = pd.read_csv(paired)[["metric", "comparison", "ci_low_pct", "ci_high_pct"]]
        paired_df = paired_df.rename(columns={"ci_low_pct": "condition_bootstrap_ci_low_pct", "ci_high_pct": "condition_bootstrap_ci_high_pct"})
        frame = frame.drop(columns=["condition_bootstrap_ci_low_pct", "condition_bootstrap_ci_high_pct"]).merge(
            paired_df, on=["metric", "comparison"], how="left")
    frame.to_csv(out / "batch_clustered_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
