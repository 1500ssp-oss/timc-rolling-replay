"""Small consistency check for the canonical Data1 PID selection scorer.

This intentionally evaluates only two candidates on one Data1 pass.  It does
not launch the 1050-candidate grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_pid_selection_audit as audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runner, suite, data1, config, envelope, seed_index, manifest = audit._load_data(
        str(args.batch_root.resolve())
    )
    replay_pass = data1[0]
    seed = audit.canonical_data1_seed(runner, replay_pass.pass_id, seed_index)
    frozen = runner.protocol.PidParams(**manifest["locked_pid"])
    contrast = runner.protocol.PidParams(0.4, 0.4, 0.0, 0.6)
    rows = []
    for params in (frozen, contrast):
        checked = audit.score_pid_pass(
            runner,
            suite,
            replay_pass,
            config,
            envelope,
            params,
            seed,
            verify_replay_contract=True,
        )
        # Independent invocation written exactly as batch_sweep.py invokes the
        # runner, used here to guard the audit wrapper itself.
        curve, diag = runner.run_exposure_control(
            suite, replay_pass, config, seed, params, 0
        )
        batch_metrics = runner.suite_metrics_with_config(
            suite, replay_pass, curve, diag, config, envelope
        )
        direct = float(batch_metrics[audit.COMPOSITE_FIELD])
        delta = abs(direct - checked["composite_normalized_RMS"])
        if delta > 1e-12:
            raise AssertionError(
                f"{params.key}: audit scorer differs from batch-sweep invocation by {delta}"
            )
        rows.append(
            {
                "candidate_key": params.key,
                "pass_id": replay_pass.pass_id,
                "seed": seed,
                "audit_RMS_c": checked["composite_normalized_RMS"],
                "direct_batch_sweep_RMS_c": direct,
                "absolute_delta": delta,
                "channel_mean_delta": checked["composite_reconstruction_abs_delta"],
                "segment_count": checked["formal_segment_count"],
                "timebase_source_counts": checked["formal_timebase_source_counts"],
                "all_contract_checks_pass": all(checked["checks"].values()),
            }
        )

    result = {
        "selection_protocol_id": audit.PID_SELECTION_PROTOCOL_ID,
        "test_scope": "two candidates on one Data1 pass; no full-grid execution",
        "canonical_config_label": config.label,
        "canonical_pid_gain_scale": float(config.pid_gain_scale),
        "outer_shell": audit._outer_shell_audit(config),
        "candidate_checks": rows,
        "passed": bool(
            all(row["absolute_delta"] <= 1e-12 for row in rows)
            and all(row["channel_mean_delta"] <= 1e-12 for row in rows)
            and all(row["all_contract_checks_pass"] for row in rows)
        ),
    }
    if not result["passed"]:
        raise AssertionError("PID selection protocol consistency check failed")
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
