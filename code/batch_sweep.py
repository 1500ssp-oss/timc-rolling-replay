from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import pandas as pd


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--extension-script", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--dataset", choices=["Data1", "Data2"], default="Data2")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--pass-ids", default="")
    parser.add_argument("--no-step-log", action="store_true")
    parser.add_argument("--locked-pid-json", type=Path, default=None)
    parser.add_argument("--predictor-bank-path", type=Path, default=None)
    parser.add_argument("--include-repeats", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    runner = load_module("current_batch_runner", root / "controller_replay.py")
    extension = load_module("current_batch_extension", args.extension_script.resolve())
    extension.ENGINE_ROOT = root
    extension.FULL_PROJECT = root / "engine_core"
    extension.BATCH_ROOT = args.batch_root.resolve()
    runner.protocol.PROJECT_DIR = extension.FULL_PROJECT
    suite, replay = runner.protocol.load_suite_modules()
    all_passes, inventory = extension.load_batch_passes(runner.protocol, suite, replay)

    if args.dataset == "Data2":
        allowed = set(extension.CANONICAL_DATA2_IDS)
        if args.include_repeats:
            allowed |= set(extension.REPEAT_EXTENSION_IDS)
        seed_index = extension.CANONICAL_SEED_INDEX
    else:
        allowed = extension.DATA1_IDS
        ordered = [
            "P14", "P13", "P12", "P04", "P03", "P02", "P01",
            "P26", "P25", "P24", "P23", "P22", "P21",
        ]
        seed_index = {pass_id: index for index, pass_id in enumerate(ordered)}
    passes = [item for item in all_passes if item.pass_id in allowed]
    if args.pass_ids:
        requested = {item.strip() for item in args.pass_ids.split(",") if item.strip()}
        passes = [item for item in passes if item.pass_id in requested]
    passes.sort(key=lambda item: seed_index[item.pass_id])

    configs = [
        runner.ExposureConfig(**item)
        for item in json.loads(args.config_json.read_text(encoding="utf-8"))
    ]
    if args.predictor_bank_path is not None:
        predictor_path = str(args.predictor_bank_path.resolve())
        configs = [
            replace(config, predictor_bank_path=predictor_path)
            if config.predictor_bank_path
            else config
            for config in configs
        ]
    manifest = json.loads(
        (args.run_root / "03_nominal" / "stage_allocation_ablation_manifest.json").read_text(encoding="utf-8")
    )
    locked_pid = runner.protocol.PidParams(**manifest["locked_pid"])
    if args.locked_pid_json is not None:
        payload = json.loads(args.locked_pid_json.read_text(encoding="utf-8"))
        names = {item.name for item in fields(runner.protocol.PidParams)}
        locked_pid = runner.protocol.PidParams(**{key: payload[key] for key in names})
    envelope = pd.read_csv(
        args.run_root / "03_nominal" / "audit" / "data1_stage_quantile_envelope_p025_p975.csv"
    )

    metric_rows = []
    step_frames = []
    total = len(configs) * len(passes) * args.repeats
    count = 0
    for config in configs:
        for replay_pass in passes:
            for repeat in range(args.repeats):
                count += 1
                # Seed depends only on pass and repeat: every policy shares the
                # same random realization for a given pass, keeping the paired
                # comparison free of policy-specific noise draws.
                seed = (
                    runner.RNG_SEED
                    + repeat
                    + 2000 * (seed_index[replay_pass.pass_id] + 1)
                )
                curve, diag = runner.run_exposure_control(
                    suite, replay_pass, config, seed, locked_pid, repeat
                )
                metric_rows.append(
                    runner.suite_metrics_with_config(
                        suite, replay_pass, curve, diag, config, envelope
                    )
                )
                if not args.no_step_log:
                    step_frames.append(curve)
                if count == 1 or count == total or count % 100 == 0:
                    print(
                        f"[batch-sweep] {count}/{total} {config.label} {replay_pass.pass_id}",
                        flush=True,
                    )

    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    # P06 is assigned its acceleration-specific analysis condition centrally
    # in batch_archive.load_batch_passes, so metrics and row logs cannot drift.
    metrics.to_csv(out / "stage_allocation_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    if step_frames:
        steps = pd.concat(step_frames, ignore_index=True, sort=False)
        steps.to_csv(
            out / "stage_allocation_ablation_step_logs.csv.gz",
            index=False,
            encoding="utf-8-sig",
            compression="gzip",
        )
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
    summary = runner.condition_weighted_summary(
        metrics, ["block", "scenario", "control", "comparison_base"], metric_cols
    )
    summary.to_csv(
        out / "stage_allocation_ablation_condition_weighted_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    runner.paired_effect_summary(metrics, steps).to_csv(
        out / "stage_allocation_ablation_paired_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (out / "current_batch_sweep_manifest.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "passes": [item.pass_id for item in passes],
                "configs": len(configs),
                "repeats": args.repeats,
                "timebase": "segment_gap_v2",
                "batch_root": str(args.batch_root.resolve()),
                "inventory_rows": len(inventory),
                "locked_pid": {
                    item.name: getattr(locked_pid, item.name)
                    for item in fields(runner.protocol.PidParams)
                },
                "predictor_bank_override": str(args.predictor_bank_path.resolve())
                if args.predictor_bank_path is not None
                else None,
                "include_repeats": bool(args.include_repeats),
                **runner.protocol.scale_lock_manifest(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
