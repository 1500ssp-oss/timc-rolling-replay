"""Master rerun pipeline for the publication analysis package.

Runs every replay-dependent experiment in order, then assembles the
publication tables, the rate-sensitivity table (with its 1e-12 canonical
anchor), figures and the SHA256 manifest. Raw production data are read only
through the batch root supplied on the command line; nothing is written
outside the package.

Usage:
    py -3.14 code/run_pipeline.py --batch-root "<BATCH_ROOT>" [--skip-stress]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from raw_archive import resolve_archive

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
PY = sys.executable


def run(name: str, args: list[str], cwd: Path = ROOT, timeout: int | None = None) -> None:
    t0 = time.time()
    print(f"\n===== STEP {name} =====", flush=True)
    print(" ".join(str(a) for a in args), flush=True)
    completed = subprocess.run(
        [PY, *args], cwd=str(cwd), capture_output=False, timeout=timeout
    )
    if completed.returncode != 0:
        raise RuntimeError(f"STEP {name} failed with rc={completed.returncode}")
    print(f"----- STEP {name} done in {time.time()-t0:,.1f}s", flush=True)


def exists(path: Path) -> bool:
    return path.exists()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_archive(batch_root: Path) -> None:
    batch_map = ROOT / "data_map" / "production_batch_map.csv"
    archive_map, resolved = resolve_archive(batch_root, batch_map)
    if len(resolved) != len(archive_map):
        raise RuntimeError("Not every archive-map digest resolved to one raw CSV")
    print("[input-lock] verified 26 raw pass hashes against the released batch map", flush=True)


def pipeline_input_fingerprint() -> str:
    paths: list[Path] = sorted((ROOT / "code").rglob("*.py"))
    paths.extend(sorted((ROOT / "tests").rglob("*.py")))
    paths.append(ROOT / "verify_results.py")
    for path in sorted((ROOT / "config").rglob("*")):
        if not path.is_file() or "implementation_sensitivity" in path.parts:
            continue
        paths.append(path)
    for base in [ROOT / "models", ROOT / "data_map"]:
        paths.extend(path for path in sorted(base.rglob("*")) if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-root",
        required=True,
        type=Path,
        help="folder containing the seven production-batch directories",
    )
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument(
        "--audit-published-pid-grid", action="store_true",
        help="reuse the complete published PID grid and recheck its selected candidate on raw Data1; default recomputes all 1050 candidates",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fresh", action="store_true",
        help="remove only this package's outputs directory before rerunning",
    )
    mode.add_argument(
        "--resume", action="store_true",
        help="explicitly resume a partial run after scale-lock verification",
    )
    args = parser.parse_args()
    if not args.batch_root.is_dir():
        parser.error(f"batch root is not a directory: {args.batch_root}")
    batch_root = str(args.batch_root.resolve())
    verify_input_archive(Path(batch_root))
    outputs = (ROOT / "outputs").resolve()
    if outputs.exists() and any(outputs.iterdir()) and not (args.fresh or args.resume):
        parser.error("outputs is non-empty; use --fresh for a clean rerun or --resume explicitly")
    if args.fresh:
        if outputs.parent != ROOT.resolve() or outputs.name != "outputs":
            raise RuntimeError(f"Refusing unsafe fresh target: {outputs}")
        if outputs.exists():
            shutil.rmtree(outputs)
        print(f"[fresh] cleared replay outputs under {outputs}", flush=True)

    common = [
        "--run-root", "config/reference_run",
        "--batch-root", batch_root,
        "--extension-script", "code/batch_archive.py",
        "--dataset", "Data2",
        "--predictor-bank-path", "models/final_thickness_model.pt",
    ]

    # 0a. Verify the frozen Data1-only scale catalogue against the supplied
    # raw Data1 files before any Data2 file is loaded.
    run("Data1 scale lock", [
        str(CODE / "build_scale_lock.py"),
        "--batch-root", batch_root,
        "--check",
    ])
    scale_hash = json.loads(
        (ROOT / "config" / "data1_scale_lock.json").read_text(encoding="utf-8")
    )["payload_sha256"]
    outputs.mkdir(parents=True, exist_ok=True)
    preexisting_top = {path.name for path in outputs.iterdir()}
    scale_marker = outputs / ".scale_lock_sha256"
    if args.resume and scale_marker.exists():
        recorded_hash = scale_marker.read_text(encoding="utf-8").strip()
        if recorded_hash != scale_hash:
            raise RuntimeError(
                "Existing outputs were generated under a different scale lock; rerun with --fresh"
            )
    elif args.resume and preexisting_top:
        if preexisting_top != {"pid_selection_audit"}:
            raise RuntimeError(
                "Existing outputs have no scale-lock marker; rerun with --fresh"
            )
        audit_path = outputs / "pid_selection_audit" / "pid_selection_audit.json"
        if not audit_path.exists() or json.loads(
            audit_path.read_text(encoding="utf-8")
        ).get("scale_lock_payload_sha256") != scale_hash:
            raise RuntimeError(
                "PID audit does not match the active scale lock; rerun with --fresh"
            )

    reference_manifest = json.loads(
        (ROOT / "config" / "reference_run" / "03_nominal" /
         "stage_allocation_ablation_manifest.json").read_text(encoding="utf-8")
    )
    reference_pid = reference_manifest["locked_pid"]
    standalone_pid = json.loads(
        (ROOT / "config" / "locked_pid.json").read_text(encoding="utf-8")
    )
    if reference_pid != standalone_pid:
        raise RuntimeError(
            "PID lock drift: config/locked_pid.json differs from the reference manifest"
        )
    pid_hash = stable_json_sha256(reference_pid)
    pid_marker = outputs / ".pid_lock_sha256"
    if args.resume and pid_marker.exists():
        recorded_hash = pid_marker.read_text(encoding="utf-8").strip()
        if recorded_hash != pid_hash:
            raise RuntimeError(
                "Existing outputs were generated under a different PID lock; rerun with --fresh"
            )
    elif args.resume and preexisting_top:
        if preexisting_top != {"pid_selection_audit"}:
            raise RuntimeError(
                "Existing outputs have no PID-lock marker; rerun with --fresh"
            )
        audit_path = outputs / "pid_selection_audit" / "pid_selection_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("selection_protocol_id")
            != "canonical_pid_replay_segment_gap_v2"
            or audit.get("selected_all_replay_contract_checks_pass") is not True
            or audit.get("selected") != reference_pid
        ):
            raise RuntimeError(
                "PID audit is not a valid canonical-replay audit for the active PID lock; "
                "rerun the audit or use --fresh"
            )
    pipeline_hash = pipeline_input_fingerprint()
    pipeline_marker = outputs / ".pipeline_input_sha256"
    if args.resume and pipeline_marker.exists():
        recorded_hash = pipeline_marker.read_text(encoding="utf-8").strip()
        if recorded_hash != pipeline_hash:
            raise RuntimeError(
                "Replay code/config/model inputs changed; rerun with --fresh"
            )
    elif args.resume and preexisting_top and preexisting_top != {"pid_selection_audit"}:
        raise RuntimeError(
            "Existing outputs have no pipeline-input fingerprint; rerun with --fresh"
        )
    scale_marker.write_text(str(scale_hash) + "\n", encoding="utf-8")
    pid_marker.write_text(pid_hash + "\n", encoding="utf-8")
    pipeline_marker.write_text(pipeline_hash + "\n", encoding="utf-8")

    # 0b. Regenerate every sensitivity configuration from the frozen lock so
    # the studies can never drift from the canonical policy definitions.
    run("sensitivity configs", [str(CODE / "build_sensitivity_configs.py")])
    run("predictor timeline audit", [
        str(ROOT / "tests" / "test_predictor_timeline.py"),
        "--audit-json", str(ROOT / "results" / "predictor_timeline_audit.json"),
    ])
    # This audits the retained historical aggregates; it does not retrain
    # the frozen checkpoint or reconstruct missing historical fitting logs.
    run("predictor training provenance audit", [
        str(CODE / "predictor_training_audit.py"),
    ])

    pid_audit = ROOT / "outputs" / "pid_selection_audit"
    if not exists(pid_audit / "pid_selection_audit.json"):
        pid_grid_args = []
        if args.audit_published_pid_grid:
            pid_audit.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "results" / "pid_full_grid.csv", pid_audit / "pid_full_grid.csv")
            pid_grid_args = ["--audit-existing-grid"]
        run("Data1 PID selection audit", [
            str(CODE / "run_pid_selection_audit.py"),
            "--batch-root", batch_root,
            "--out-dir", str(pid_audit),
            *pid_grid_args,
        ], timeout=14400)
    else:
        audit = json.loads(
            (pid_audit / "pid_selection_audit.json").read_text(encoding="utf-8")
        )
        if (
            audit.get("selection_protocol_id")
            != "canonical_pid_replay_segment_gap_v2"
            or audit.get("selected_all_replay_contract_checks_pass") is not True
            or audit.get("selected") != reference_pid
        ):
            raise RuntimeError(
                "Existing PID selection output does not validate the active lock; "
                "rerun with --fresh"
            )

    # 0c. Canonical main replay (the primary result that every later step
    # consumes).  Step logs are kept for the episode/command audits.
    canon_dir = ROOT / "outputs" / "canonical" / "03_nominal"
    if not exists(canon_dir / "stage_allocation_ablation_metrics.csv"):
        run("canonical replay", [
            str(CODE / "batch_sweep.py"),
            "--out-root", str(canon_dir),
            *common,
            "--config-json", "config/selected_controller_configs.json",
        ])

    # 0d. Make the canonical run dir complete for downstream audits.
    canon_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "config" / "reference_run" / "03_nominal" / "stage_allocation_ablation_manifest.json",
        canon_dir / "stage_allocation_ablation_manifest.json",
    )
    audit_dir = canon_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for src in (ROOT / "config" / "reference_run" / "03_nominal" / "audit").glob("*"):
        if src.is_file():
            shutil.copy2(src, audit_dir / src.name)

    # 1-7. Implementation-sensitivity sweeps.
    sweeps = [
        ("01_emulator_compact", "config/implementation_sensitivity/emulator_compact_configs.json"),
        ("02_amplitude_envelope", "config/implementation_sensitivity/amplitude_envelope_configs.json"),
        ("03_anti_windup", "config/implementation_sensitivity/anti_windup_configs.json"),
        ("04_output_clip", "config/implementation_sensitivity/output_clip_configs.json"),
        ("05_no_output_clip", "config/implementation_sensitivity/output_no_clip_configs.json"),
        ("06_rate", "config/implementation_sensitivity/rate_sensitivity_configs.json"),
        ("07_architecture_ablation", "config/implementation_sensitivity/architecture_ablation_configs.json"),
    ]
    for sub, config_json in sweeps:
        out = ROOT / "outputs" / "sensitivity" / sub
        if not exists(out / "stage_allocation_ablation_metrics.csv"):
            run(f"sweep {sub}", [
                str(CODE / "batch_sweep.py"),
                "--out-root", str(out),
                *common,
                "--config-json", config_json,
                "--no-step-log",
            ])

    # 7. Data1 sweep (step logs for the command audit).
    data1_out = ROOT / "outputs" / "data1_sweep"
    if not exists(data1_out / "stage_allocation_ablation_metrics.csv"):
        run("data1 sweep", [
            str(CODE / "batch_sweep.py"),
            "--out-root", str(data1_out),
            "--run-root", "config/reference_run",
            "--batch-root", batch_root,
            "--extension-script", "code/batch_archive.py",
            "--config-json", "config/selected_controller_configs.json",
            "--dataset", "Data1",
        ])

    # 8. Expanded archive, repeat sequence, leave-one-batch-out, fixed-dt.
    expanded_out = ROOT / "outputs" / "expanded"
    if not exists(expanded_out / "expanded_26pass_batch_weighted_policy_summary.csv"):
        run("expanded archive + fixed-dt", [
            str(CODE / "batch_archive.py"),
            "--out-root", str(expanded_out),
            "--batch-root", batch_root,
        ])

    # 9. Source-sequence relocking.
    relock_out = ROOT / "outputs" / "relocking"
    if not exists(relock_out / "source_sequence_relocking_nominal_summary.csv"):
        run("source-sequence relocking", [
            str(CODE / "run_relocking.py"),
            "--batch-root", batch_root,
            "--out-dir", str(relock_out),
        ])

    # 10. Stress matrix.
    stress_root = ROOT / "outputs" / "stress_root"
    stress_metrics = stress_root / "run" / "06_stress" / "stress_metrics_current.csv"
    if not args.skip_stress and not exists(stress_metrics):
        manifest_dir = stress_root / "run" / "06_stress" / "configs"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "config" / "stress" / "scenario_manifest.json", manifest_dir / "scenario_manifest.json")
        shutil.copytree(ROOT / "config" / "reference_run" / "03_nominal", stress_root / "run" / "03_nominal", dirs_exist_ok=True)
        run("stress sweep", [
            str(CODE / "stress_sweep.py"),
            "--root", str(stress_root),
            "--batch-root", batch_root,
            "--extension-script", "code/batch_archive.py",
            "--workers", "4",
            "--repeats", "3",
        ], timeout=4 * 3600)
        run("stress analysis", [
            str(CODE / "stress_analysis.py"),
            str(stress_metrics),
            "--out-root", str(stress_root),
        ])

    # 11. Local OPC UA loopback.
    opcua_out = ROOT / "outputs" / "opcua"
    if not exists(opcua_out / "results.json"):
        run("opcua loopback", [
            str(CODE / "opcua_loopback.py"),
            "--out-dir", str(opcua_out),
        ], timeout=3600)

    # 12. Command audit.
    command_out = ROOT / "outputs" / "command_audit"
    if not exists(command_out / "data2_command_audit_summary.csv"):
        run("command audit", [
            str(CODE / "command_audit.py"),
            "--data1-logs", str(data1_out / "stage_allocation_ablation_step_logs.csv.gz"),
            "--data2-logs", str(canon_dir / "stage_allocation_ablation_step_logs.csv.gz"),
            "--out-dir", str(command_out),
        ])

    # 13. Corridor sensitivity.
    corridor_out = ROOT / "outputs" / "corridor"
    if not exists(corridor_out / "reference_corridor_coverage_sensitivity.csv"):
        run("corridor sensitivity", [
            str(CODE / "corridor_sensitivity.py"),
            "--run", str(ROOT / "outputs" / "canonical"),
            "--out-dir", str(corridor_out),
            "--batch-root", batch_root,
            "--extension-script", "code/batch_archive.py",
        ])

    # 13b. Characterize the archived response law on Data1 with zero command
    # and zero innovation at rolling origins.  This is explicitly a forward
    # discrepancy diagnostic, not physical or causal plant validation.
    farch_out = ROOT / "outputs" / "farch_forward_diagnostic"
    if not exists(farch_out / "farch_forward_audit.json"):
        run("F_arch forward discrepancy diagnostic", [
            str(CODE / "farch_forward_diagnostic.py"),
            "--batch-root", batch_root,
            "--out-dir", str(farch_out),
        ], timeout=3600)

    # 14. Fault diagnostics.
    faults_out = ROOT / "outputs" / "faults"
    if not exists(faults_out / "fault_diagnostic_summary.csv"):
        run("fault diagnostics", [
            str(CODE / "run_faults.py"),
            "--opcua-dir", str(opcua_out),
            "--batch-root", batch_root,
            "--out-dir", str(faults_out),
        ])

    # 15. Appendix audit (maturity, MDE, episodes, hard-fail).
    appendix_dir = canon_dir / "10_appendix_audit"
    if not exists(appendix_dir / "20_episode_metrics_by_file.csv"):
        run("appendix audit", [
            str(CODE / "appendix_audit.py"),
            "--run", str(ROOT / "outputs" / "canonical"),
            "--batch-root", batch_root,
            "--extension-script", "code/batch_archive.py",
        ])

    # 15b. Directly compare the canonical replay flatness row distribution
    # with recorded Data2 under the Data1-locked maturity thresholds.  This is
    # a descriptive replay diagnostic, not a causal or physical validation.
    distribution_out = ROOT / "outputs" / "replay_distribution"
    if not exists(distribution_out / "replay_flatness_distribution_audit.json"):
        run("replay flatness distribution diagnostic", [
            str(CODE / "replay_distribution_diagnostic.py"),
            "--run", str(ROOT / "outputs" / "canonical"),
            "--batch-root", batch_root,
            "--extension-script", "code/batch_archive.py",
            "--out-dir", str(distribution_out),
        ])

    # 16. Seed stability.
    seed_out = ROOT / "outputs" / "seed_sensitivity"
    if not exists(seed_out / "seed_sensitivity_summary.csv"):
        run("seed sensitivity", [
            str(CODE / "run_seed_sensitivity.py"),
            "--batch-root", batch_root,
            "--out-dir", str(seed_out),
        ])

    # 17. Sensitivity analysis (bootstrap, ROPE, MDE, condition effects).
    run("implementation sensitivity analysis", [
        str(CODE / "implementation_sensitivity.py"),
        "--run", str(ROOT / "outputs" / "canonical"),
        "--base", str(ROOT / "outputs" / "sensitivity"),
    ])

    # 18. Assemble publication tables, then the rate table, then the manifest.
    run("publish tables", [
        str(CODE / "publish_tables.py"),
        "--batch-root", batch_root,
    ])
    # Rebuild the lower-/upper-side recorded-band decomposition directly from
    # canonical step logs. This also audits exact equality to the retained
    # legacy S_out/S_excess fields and aligns k->k+1 guardrail flags with the
    # reached row before producing the descriptive guardrail ledger.
    run("recorded-band decomposition", [
        str(CODE / "recorded_band_decomposition.py"),
    ])
    run("rate table", [
        str(CODE / "build_rate_sensitivity_table.py"),
        "--metrics", str(ROOT / "outputs" / "sensitivity" / "06_rate" / "stage_allocation_ablation_metrics.csv"),
        "--canonical", str(ROOT / "results" / "canonical_nominal_condition_weighted.csv"),
        "--output", str(ROOT / "results" / "Table_S12_rate_sensitivity.csv"),
        "--audit", str(ROOT / "results" / "Table_S12_rate_sensitivity_anchor_audit.json"),
    ])
    run("publish figures", [str(CODE / "publish_figures.py")])

    # 19. Batch-clustered bootstrap (conditions nest inside four batches).
    run("batch-clustered bootstrap", [
        str(CODE / "run_batch_bootstrap.py"),
        "--out-dir", str(ROOT / "outputs" / "batch_bootstrap"),
    ])
    batch_boot = ROOT / "outputs" / "batch_bootstrap" / "batch_clustered_bootstrap.csv"
    if batch_boot.exists():
        shutil.copy2(batch_boot, ROOT / "results" / "batch_clustered_bootstrap.csv")

    run("independent publication statistics audit", [
        str(CODE / "public_statistics_audit.py"),
        "--audit-json", str(ROOT / "results" / "public_statistics_audit.json"),
    ])
    run("independent supported-command motion audit", [
        str(CODE / "supported_motion_audit.py"),
        "--canonical-step-log", str(canon_dir / "stage_allocation_ablation_step_logs.csv.gz"),
        "--audit-json", str(ROOT / "results" / "supported_motion_audit.json"),
    ])
    run("manifest", [str(CODE / "publish_tables.py"), "--manifest-only"])
    run("verify", [str(ROOT / "verify_results.py")])
    print("\n===== PIPELINE COMPLETE =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
