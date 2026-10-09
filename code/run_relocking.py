"""Data1 source-sequence omission audit.

For each omission fold S1/S2/S3 this script (a) re-selects the PID outer slew
scale on the retained Data1 source passes from the pre-specified compact grid
{0.60, 0.80} while keeping the locked inner PID (Kp 1.6, Ki 1.4, Kd 0.75,
move limit 1.4), and (b) re-evaluates the four locked policies on the ten
the Data2 files. The predictor bank remains the frozen Data1 GRU lock for
every fold, isolating the source-omission effect on the controller lock rather
than re-fitting predictors per fold.

Outputs match the published result schemas:
    source_sequence_relocking_nominal_summary.csv
    source_sequence_relocking_selected_configs.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import batch_archive as archive  # noqa: E402
import controller_replay as runner  # noqa: E402

OMISSIONS = {
    "S1": {f"P{i:02d}" for i in range(1, 5)},
    "S2": {f"P{i:02d}" for i in range(12, 15)},
    "S3": {f"P{i:02d}" for i in range(21, 27)},
}
OUTER_GRID = [0.60, 0.80]
PREDICTOR_MEAN_MASE = 0.8312331297433392  # frozen Data1 GRU lock

BASE_TO_POLICY = {"C0": "PID", "ADRC": "OC-PID", "C2": "C2", "C7": "C7"}


def short_name(config: runner.ExposureConfig) -> str:
    return BASE_TO_POLICY[config.base_control]


def load_passes(batch_root: Path):
    runner.protocol.PROJECT_DIR = archive.FULL_PROJECT
    suite, replay = runner.protocol.load_suite_modules()
    extension = archive
    extension.BATCH_ROOT = Path(batch_root)
    all_passes, _ = extension.load_batch_passes(runner.protocol, suite, replay)
    return suite, all_passes


def run_policy(suite, rp, config, locked_pid, envelope, seed):
    curve, diag = runner.run_exposure_control(suite, rp, config, seed, locked_pid, 0)
    curve["pass_id"] = rp.pass_id
    curve["control"] = config.label
    return runner.suite_metrics_with_config(suite, rp, curve, diag, config, envelope)


def condition_weighted(metrics: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    if "pass_id" in metrics and "condition_id" in metrics:
        metrics = metrics.copy()
        metrics.loc[metrics["pass_id"].eq("P06"), "condition_id"] = "0.628->0.458 acceleration"
    file_level = metrics.groupby(["control", "fold", "condition_id", "file"], as_index=False)[metric_cols].mean()
    condition_level = file_level.groupby(["control", "fold", "condition_id"], as_index=False)[metric_cols].mean()
    return condition_level.groupby(["control", "fold"], as_index=False)[metric_cols].mean()


def motion_summary_values(summary: pd.Series) -> dict[str, float]:
    # TV/100m is already in the core row; append the other motion fields in the
    # historical publication order after the relative-effect columns.
    return {field: float(summary[field]) for field in archive.MOTION_METRICS if field != "TV_L_per_100m"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite, all_passes = load_passes(args.batch_root)

    manifest = json.loads(
        (archive.CANONICAL_RUN / "03_nominal" / "stage_allocation_ablation_manifest.json").read_text(encoding="utf-8")
    )
    locked_pid = runner.protocol.PidParams(**manifest["locked_pid"])
    envelope = pd.read_csv(
        archive.CANONICAL_RUN / "03_nominal" / "audit" / "data1_stage_quantile_envelope_p025_p975.csv"
    )
    base_configs = {
        item["base_control"]: runner.ExposureConfig(**item)
        for item in json.loads((ROOT.parent / "config" / "selected_controller_configs.json").read_text(encoding="utf-8"))
    }

    data1 = [rp for rp in all_passes if rp.pass_id in archive.DATA1_IDS]
    data2 = [rp for rp in all_passes if rp.pass_id in archive.CANONICAL_DATA2_IDS]
    data2.sort(key=lambda rp: archive.CANONICAL_SEED_INDEX[rp.pass_id])

    metric_cols = list(dict.fromkeys(
        ["composite_normalized_RMS", "S_out", "S_excess_mean", "S_excess_cvar95", "TV_L_per_100m"]
        + archive.MOTION_METRICS
    ))
    selected_rows: list[dict] = []
    summary_rows: list[dict] = []

    for fold in sorted(OMISSIONS):
        omitted = OMISSIONS[fold]
        retained = [rp for rp in data1 if rp.pass_id not in omitted]

        # (a) PID outer-slew re-selection on retained Data1 sources.
        best_scale, best_score = None, float("inf")
        for outer in OUTER_GRID:
            config = runner.ExposureConfig(**{
                **base_configs["C0"].__dict__,
                "label": f"PID-g1.20-du{outer:.2f}",
                "outer_du_scale": outer,
                "seed_group": f"PID-g1.20-du{outer:.2f}",
                "scenario_id": f"relock_{fold}",
            })
            values = []
            for rp in retained:
                seed = runner.RNG_SEED + 2000 * (archive.CANONICAL_SEED_INDEX.get(rp.pass_id, 0) + 1)
                metrics = run_policy(suite, rp, config, locked_pid, envelope, seed)
                values.append(metrics["composite_normalized_RMS"])
            score = float(np.mean(values))
            if score < best_score:
                best_score, best_scale = score, outer

        # (b) Data2 evaluation with the selected PID and the locked others.
        pid_config = runner.ExposureConfig(**{
            **base_configs["C0"].__dict__,
            "label": f"PID-g1.20-du{best_scale:.2f}",
            "outer_du_scale": best_scale,
            "seed_group": f"PID-g1.20-du{best_scale:.2f}",
            "scenario_id": f"relock_{fold}",
        })
        configs = [pid_config, base_configs["ADRC"], base_configs["C2"], base_configs["C7"]]
        for config in configs:
            for rp in data2:
                seed = runner.RNG_SEED + 2000 * (archive.CANONICAL_SEED_INDEX[rp.pass_id] + 1)
                metrics = run_policy(suite, rp, config, locked_pid, envelope, seed)
                metrics["fold"] = fold
                summary_rows.append(metrics)

            selected_rows.append({
                "omitted_source_sequence": fold,
                "policy": short_name(config),
                "selected_config": config.label,
                "selected_predictor": "GRU (frozen Data1 lock)",
                "predictor_mean_mase": PREDICTOR_MEAN_MASE,
                "outer_du_scale": config.outer_du_scale,
                "pid_gain_scale": config.pid_gain_scale,
                "mpc_share": config.mpc_share,
                "ff_share": config.ff_share,
                "clamp_scale": config.clamp_scale,
                "locked_pid_kp_scale": locked_pid.kp_scale,
                "locked_pid_ki_scale": locked_pid.ki_scale,
                "locked_pid_kd_scale": locked_pid.kd_scale,
                "locked_pid_move_limit_scale": locked_pid.move_limit_scale,
                "scope": "nominal-only outer source-sequence omission audit (frozen predictor)",
            })

    metrics = pd.DataFrame(summary_rows)
    metrics["policy"] = metrics["base_control"].map(BASE_TO_POLICY)
    # Within a fold each policy appears exactly once (only the selected PID
    # candidate is kept), so the short policy name can replace the run label.
    metrics["control"] = metrics["policy"]
    cw = condition_weighted(metrics, metric_cols)
    cw["policy"] = cw["control"]
    retained_by_fold = {
        fold: ";".join(sorted(archive.DATA1_IDS - OMISSIONS[fold])) for fold in sorted(OMISSIONS)
    }
    selected_frame = pd.DataFrame(selected_rows).drop_duplicates(subset=["omitted_source_sequence", "policy"])
    rows = []
    for fold in sorted(OMISSIONS):
        by_policy = cw.set_index(["policy", "fold"]).xs(fold, level="fold")
        sel = selected_frame[selected_frame.omitted_source_sequence.eq(fold)].set_index("policy")
        for policy in ["C7", "C2", "OC-PID", "PID"]:
            base = by_policy.loc[policy]
            config = sel.loc[policy]
            row = {
                "policy": policy,
                "composite_normalized_RMS": float(base["composite_normalized_RMS"]),
                "S_excess_mean": float(base["S_excess_mean"]),
                "S_excess_cvar95": float(base["S_excess_cvar95"]),
                "S_out": float(base["S_out"]),
                "TV_L_per_100m": float(base["TV_L_per_100m"]),
                "omitted_source_sequence": fold,
                "retained_source_passes": retained_by_fold[fold],
                "n_source_passes": len(archive.DATA1_IDS - OMISSIONS[fold]),
                "selected_predictor": config["selected_predictor"],
                "predictor_mean_mase": config["predictor_mean_mase"],
                "locked_pid_kp_scale": config["locked_pid_kp_scale"],
                "locked_pid_ki_scale": config["locked_pid_ki_scale"],
                "locked_pid_kd_scale": config["locked_pid_kd_scale"],
                "locked_pid_move_limit_scale": config["locked_pid_move_limit_scale"],
                "scope": config["scope"],
                "time_protocol_version": "segment_gap_v2",
                "selected_config": config["selected_config"],
                "outer_du_scale": config["outer_du_scale"],
                "pid_gain_scale": config["pid_gain_scale"],
                "mpc_share": config["mpc_share"],
                "ff_share": config["ff_share"],
            }
            for baseline in ["PID", "OC-PID"]:
                if baseline not in by_policy.index:
                    continue
                b = by_policy.loc[baseline]
                row[f"relative_S_excess_mean_vs_{baseline}_pct"] = (
                    100.0 * (base["S_excess_mean"] - b["S_excess_mean"]) / b["S_excess_mean"]
                    if abs(b["S_excess_mean"]) > 1e-12 else np.nan
                )
                row[f"relative_S_excess_cvar95_vs_{baseline}_pct"] = (
                    100.0 * (base["S_excess_cvar95"] - b["S_excess_cvar95"]) / b["S_excess_cvar95"]
                    if abs(b["S_excess_cvar95"]) > 1e-12 else np.nan
                )
                row[f"relative_composite_normalized_RMS_vs_{baseline}_pct"] = (
                    100.0 * (base["composite_normalized_RMS"] - b["composite_normalized_RMS"]) / b["composite_normalized_RMS"]
                    if abs(b["composite_normalized_RMS"]) > 1e-12 else np.nan
                )
                row[f"relative_TV_L_per_100m_vs_{baseline}_pct"] = (
                    100.0 * (base["TV_L_per_100m"] - b["TV_L_per_100m"]) / b["TV_L_per_100m"]
                    if abs(b["TV_L_per_100m"]) > 1e-12 else np.nan
                )
            row.update(motion_summary_values(base))
            rows.append(row)
    out_frame = pd.DataFrame(rows)
    out_frame.to_csv(out / "source_sequence_relocking_nominal_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(out / "source_sequence_relocking_selected_configs.csv", index=False, encoding="utf-8-sig")
    print(f"[relocking] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
