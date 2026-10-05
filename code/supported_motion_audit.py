"""Independent checks of supported-command motion, without production helpers.

The default audit uses public aggregate rows. On an authorized machine the
optional retained canonical log directly verifies the segmented numerators.
Neither mode reconstructs plant validity or historical predictor training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHANNELS = ["speed", "gap", "shape"]


def normalize_policy(value):
    if str(value) == "OC-PID":
        return "OC-PID"
    if str(value).startswith("ADRC"):
        return "OC-PID"
    if str(value).startswith("C7"):
        return "C7-Core"
    return str(value).split("-")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-step-log", type=Path)
    parser.add_argument("--audit-json", type=Path)
    args = parser.parse_args()
    table = pd.read_csv(ROOT / "results/canonical_nominal_pass_metrics.csv")
    table["policy"] = table.policy.map(normalize_policy)
    fields = ["control_total_variation", "postclamp_total_variation", "control_delta_rms", "startup_motion", "restart_motion", "startup_restart_motion", "TV_L_per_100m", "TV_t_per_s", "distance_m", "duration_s"]
    if table[fields].isna().any().any() or not np.isfinite(table[fields].to_numpy(float)).all():
        raise ValueError("Canonical motion aggregates must be finite")
    if (table[fields].to_numpy(float) < -1e-12).any() or table.distance_m.le(0).any() or table.duration_s.le(0).any():
        raise ValueError("Invalid motion values or unsupported denominator")
    errors = [float((table.control_total_variation - table.postclamp_total_variation).abs().max()),
              float((table.startup_motion + table.restart_motion - table.startup_restart_motion).abs().max()),
              float((100 * table.postclamp_total_variation / table.distance_m - table.TV_L_per_100m).abs().max()),
              float((table.postclamp_total_variation / table.duration_s - table.TV_t_per_s).abs().max())]
    report = {"definition": "L1 adjacent applied-command differences only within the same contiguous supported segment in original row order",
              "startup_definition": "L1 magnitude of the first applied command from reset zero",
              "restart_definition": "Sum of L1 applied-command magnitudes at subsequent contiguous-segment starts from reset zero",
              "normalization": "100 * supported TV / supported observed distance; zero-length files have undefined normalized TV",
              "public_aggregate_rows": len(table), "public_identity_max_abs_error": max(errors),
              "public_aggregate_checks_passed": len(table) == 40 and max(errors) <= 1e-10,
              "raw_trajectory_reconstruction_executed": args.canonical_step_log is not None,
              "scope": "Aggregate identities are consistency checks, not independent reconstruction of private trajectories or physical validation."}
    if args.canonical_step_log is not None:
        logs = pd.read_csv(args.canonical_step_log)
        records, raw_errors = [], []
        for (control, pass_id), g in logs.groupby(["control", "pass_id"], sort=False):
            k_numeric = g.k.to_numpy(float)
            if not np.isfinite(k_numeric).all() or not np.equal(k_numeric, np.floor(k_numeric)).all():
                raise ValueError("Canonical row indices must be finite integers")
            k = k_numeric.astype(int)
            if not np.all(np.diff(k) == 1) or g.segment_id.isna().any():
                raise ValueError("Incomplete, unordered or unlabeled canonical step group")
            seg = g.segment_id.to_numpy()
            eligible = seg[1:] == seg[:-1]
            u = g[["u_" + c for c in CHANNELS]].to_numpy(float)
            if not np.isfinite(u).all():
                raise ValueError("Canonical applied commands must be finite")
            deltas = np.diff(u, axis=0)
            tv = float(np.abs(deltas[eligible]).sum())
            startup = float(np.abs(u[0]).sum())
            restart = float(np.abs(u[1:][~eligible]).sum())
            rms = float(np.sqrt(np.mean(deltas[eligible] ** 2))) if eligible.any() else 0.0
            policy = normalize_policy(control)
            row = table.loc[table.policy.eq(policy) & table.pass_id.eq(pass_id)]
            if len(row) != 1:
                raise ValueError("Canonical trajectory does not have one published aggregate")
            p = row.iloc[0]
            values = {"control_total_variation": tv, "postclamp_total_variation": tv,
                      "control_delta_rms": rms, "startup_motion": startup, "restart_motion": restart,
                      "startup_restart_motion": startup + restart,
                      "delta_u_over_peak_action": float(np.abs(deltas[eligible]).max(initial=0) / (np.abs(u).max(initial=0) + 1e-9))}
            for name, prefix in [("preclamp_total_variation", "u_preclamp"), ("mpc_raw_total_variation", "u_mpc"), ("mpc_component_total_variation", "u_mpc_component")]:
                arr = g[[prefix + "_" + c for c in CHANNELS]].to_numpy(float)
                if not np.isfinite(arr).all():
                    raise ValueError("Canonical component commands must be finite")
                values[name] = float(np.abs(np.diff(arr, axis=0)[eligible]).sum())
            raw_errors.extend(abs(float(p[name]) - value) for name, value in values.items())
            records.append({"policy": policy, "pass_id": pass_id, "n_rows": len(g), "n_contiguous_segments": int((~eligible).sum() + 1),
                            "supported_total_variation": tv, "legacy_full_file_total_variation": float(np.abs(deltas).sum()),
                            "excluded_boundary_motion": float(np.abs(deltas[~eligible]).sum()),
                            "startup_motion": startup, "restart_motion": restart, "startup_restart_motion": startup + restart})
        report["private_log_check"] = {"sha256": hashlib.sha256(args.canonical_step_log.read_bytes()).hexdigest(),
                                       "rows": len(logs), "groups": len(records), "max_abs_published_difference": max(raw_errors),
                                       "group_aggregates": records}
        report["private_log_checks_passed"] = len(records) == 40 and max(raw_errors) <= 1e-10
    report["passed"] = report["public_aggregate_checks_passed"] and report.get("private_log_checks_passed", True)
    if args.audit_json is not None:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "private_log_check"}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
