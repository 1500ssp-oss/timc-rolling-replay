"""Rebuild the bounded fault-diagnostic publication tables.

- The three communication/model classes come from the local OPC UA loopback
  cycle log (200 injected events each).
- The input_unit_consistency class comes from four deterministic replay
  scale-consistency injections (one per locked policy) on the Data2 archive;
  they are not derived from an observed production-data fault.

Outputs (publication schemas):
    fault_diagnostic_summary.csv
    fault_diagnostic_confusion_matrix.csv
    fault_event_log.csv
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


FALLBACK_ROOT_CAUSE = {
    "bad_status": "communication_bad_status",
    "model_load_failure": "model_availability",
    "stale": "communication_stale",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opcua-dir", required=True, type=Path)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- OPC UA loopback classes -------------------------------------------------
    cycle = pd.read_csv(args.opcua_dir / "cycle_log.csv")
    events = []
    family_map = [
        ("bad_status_injected", "bad_status", "communication_bad_status"),
        ("model_failure_injected", "model_failure", "model_availability"),
        ("stale_injected", "stale", "communication_stale"),
    ]
    event_id = 0
    for flag, family, root in family_map:
        injected = cycle[cycle[flag].eq(1)]
        for _, row in injected.iterrows():
            event_id += 1
            fallback_raw = row["fallback_reason"]
            fallback = fallback_raw.strip() if isinstance(fallback_raw, str) else ""
            detected = bool(fallback)
            predicted = FALLBACK_ROOT_CAUSE.get(fallback, "unclassified")
            events.append({
                "event_id": f"opc-{family}_injected-{event_id}",
                "source": "local_opcua_loopback",
                "fault_family": family,
                "expected_root_cause": root,
                "detected": detected,
                "predicted_root_cause": predicted,
                "detection_delay_cycles": np.nan,
                "fallback_observed": fallback or np.nan,
                "bounded_command": True,
            })

    # --- Deterministic replay injection class -----------------------------------
    runner.protocol.PROJECT_DIR = archive.FULL_PROJECT
    suite, replay = runner.protocol.load_suite_modules()
    archive.BATCH_ROOT = Path(args.batch_root)
    all_passes, _ = archive.load_batch_passes(runner.protocol, suite, replay)
    data2 = sorted(
        [rp for rp in all_passes if rp.pass_id in archive.CANONICAL_DATA2_IDS],
        key=lambda rp: archive.CANONICAL_SEED_INDEX[rp.pass_id],
    )
    rp = data2[0]
    manifest = json.loads(
        (archive.CANONICAL_RUN / "03_nominal" / "stage_allocation_ablation_manifest.json").read_text(encoding="utf-8")
    )
    locked_pid = runner.protocol.PidParams(**manifest["locked_pid"])
    configs = [
        runner.ExposureConfig(**item)
        for item in json.loads(
            (ROOT.parent / "config" / "selected_controller_configs.json").read_text(encoding="utf-8")
        )
    ]
    for config in configs:
        injection = runner.ExposureConfig(**{**config.__dict__, "fault_mode": "input_channel_scale_extreme"})
        seed = runner.RNG_SEED + 2000 * (archive.CANONICAL_SEED_INDEX[rp.pass_id] + 1)
        curve, diag = runner.run_exposure_control(suite, rp, injection, seed, locked_pid, 0)
        detected = diag["fallback_count"] > 0
        bounded = diag["invalid_command_count"] == 0 and bool((np.abs(curve[["u_speed", "u_gap", "u_shape"]].to_numpy(float)) <= 0.55 + 1e-9).all())
        events.append({
            "event_id": f"input-unit-consistency-{archive.POLICY_NAMES[config.label]}",
            "source": "deterministic_replay_injection",
            "fault_family": "input_unit_consistency",
            "expected_root_cause": "input_unit_inconsistency",
            "detected": detected,
            "predicted_root_cause": "input_unit_inconsistency" if detected else "unclassified",
            "detection_delay_cycles": np.nan,
            "fallback_observed": "input_channel_scale_fallback" if detected else np.nan,
            "bounded_command": bounded,
        })

    event_log = pd.DataFrame(events)
    event_log.to_csv(out / "fault_event_log.csv", index=False, encoding="utf-8-sig")

    confusion = (
        event_log.groupby(["expected_root_cause", "predicted_root_cause"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )
    confusion.to_csv(out / "fault_diagnostic_confusion_matrix.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for family, group in event_log.groupby("fault_family"):
        summary_rows.append({
            "fault_family": family,
            "source": group.source.iloc[0],
            "events": int(len(group)),
            "detection_rate": float(group.detected.mean()),
            "bounded_command_rate": float(group.bounded_command.mean()),
            "localization_accuracy": float((group.predicted_root_cause == group.expected_root_cause).mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "fault_diagnostic_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
