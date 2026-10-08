"""Post-pipeline verification for the publication package.

Checks (all offline; the raw production data are not required):
  1. Required publication files exist.
  2. MANIFEST_SHA256.csv hashes match every file it lists.
  3. The frozen controller lock matches the four publication policies.
  4. Key publication numbers cross-match their sources
     (Table 3 <-> condition-weighted summary, Table 4 <-> bootstrap file,
     rate multiplier 1.00 anchor, OPC UA boundary count).
  5. All manuscript figure assets exist.
  6. Fault-diagnostic summary and seed-sensitivity files exist and are sane.
  7. Predictor timeline regression tests execute without private data, and the
     aggregate-only training audit retains its explicitly incomplete provenance.
  8. Exact decimal upper-tail regression tests and independent public pass-level
     tail-cardinality checks execute without modifying files.
  9. Supported-segment command-motion tests execute, and independent public
     motion identities agree with the current published file metrics.

This script verifies the auditable consistency of the publication package; it
does not re-run the proprietary-data replay itself.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "code"))

from scale_lock import SCALE_FIELDS, load_scale_lock  # noqa: E402
from package_manifest import package_files  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_tables_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    columns: list[str],
) -> bool:
    """Compare independently assembled tables without positional assumptions."""
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        return False
    first = left.set_index(keys).sort_index()
    second = right.set_index(keys).sort_index()
    if not first.index.equals(second.index):
        return False
    a = first[columns].to_numpy(dtype=float)
    b = second[columns].to_numpy(dtype=float)
    return bool(
        np.isfinite(a).all()
        and np.isfinite(b).all()
        and np.allclose(a, b, rtol=0.0, atol=1e-12)
    )


def main() -> int:
    failures: list[str] = []
    checks_passed = 0

    def check(condition: bool, label: str) -> None:
        nonlocal checks_passed
        if condition:
            checks_passed += 1
        else:
            failures.append(label)

    # 1. Required files.
    required = [
        "results/canonical_nominal_condition_weighted.csv",
        "results/canonical_nominal_pass_metrics.csv",
        "results/canonical_episode_condition_weighted.csv",
        "results/canonical_relative_effects.csv",
        "results/paired_effect_bootstrap.csv",
        "results/bootstrap_non_dominated_probability.csv",
        "results/condition_level_effects.csv",
        "results/leave_one_condition_out.csv",
        "results/response_family_effects.csv",
        "results/response_family_condition_weighted.csv",
        "results/Table_S12_rate_sensitivity.csv",
        "results/Table_S12_rate_sensitivity_anchor_audit.json",
        "results/Table_S12a_amplitude_quality.csv",
        "results/Table_S12b_amplitude_channels.csv",
        "results/Table_S12c_anti_windup.csv",
        "results/Table_S12d_state_guardrail.csv",
        "results/opcua_loopback_summary.csv",
        "results/expanded_batch_weighted_absolute.csv",
        "results/expanded_batch_relative_effects.csv",
        "results/data2_leave_one_batch_out.csv",
        "results/repeat_sequence_extension_metrics.csv",
        "results/fixed_dt_vs_segment_local_dt.csv",
        "results/source_sequence_relocking_nominal_summary.csv",
        "results/source_sequence_relocking_selected_configs.csv",
        "results/time_protocol_numeric_QA.csv",
        "results/data1_scale_lock_application.csv",
        "results/data1_scale_lock_catalog.csv",
        "results/coverage_sensitivity_hard_checks.json",
        "results/command_innovation_energy_audit.csv",
        "results/maturity_headroom_summary.csv",
        "results/farch_forward_summary.csv",
        "results/farch_forward_by_pass.csv",
        "results/farch_forward_by_pass_segment.csv",
        "results/farch_recorded_envelopes.csv",
        "results/farch_recorded_envelope_occupancy.csv",
        "results/farch_recorded_envelope_occupancy_by_pass.csv",
        "results/farch_forward_audit.json",
        "results/replay_flatness_distribution_summary.csv",
        "results/replay_flatness_distribution_stratified.csv",
        "results/replay_flatness_distribution_audit.json",
        "results/output_guardrail_activation_by_condition.csv",
        "results/output_guardrail_activation_by_segment.csv",
        "results/output_guardrail_activation_summary.csv",
        "results/recorded_band_decomposition_by_file.csv",
        "results/recorded_band_decomposition_by_condition.csv",
        "results/recorded_band_decomposition_summary.csv",
        "results/recorded_band_guardrail_row_composition.csv",
        "results/recorded_band_decomposition_audit.json",
        "results/strict_c7_component_ablation.csv",
        "results/pid_full_grid.csv",
        "results/pid_locked_params.json",
        "results/pid_selection_audit.json",
        "results/data2_command_audit_summary.csv",
        "results/fault_diagnostic_summary.csv",
        "results/fault_diagnostic_confusion_matrix.csv",
        "results/fault_event_log.csv",
        "results/local_opcua_loopback_results.json",
        "results/prediction_horizon_audit.json",
        "results/predictor_lopo_one_se_summary.csv",
        "results/predictor_timeline_audit.json",
        "results/predictor_training_provenance_audit.json",
        "tests/test_predictor_timeline.py",
        "tests/test_tail_statistics.py",
        "tests/test_supported_command_motion.py",
        "code/supported_command_motion.py",
        "code/supported_motion_audit.py",
        "results/supported_motion_audit.json",
        "code/tail_statistics.py",
        "results/seed_sensitivity_summary.csv",
        "results/stress_clustered_effects.csv",
        "config/controller_lock.json",
        "config/locked_pid.json",
        "config/reference_run/03_nominal/stage_allocation_ablation_manifest.json",
        "config/selected_controller_configs.json",
        "config/model_selection.json",
        "config/data1_scale_lock.json",
        "MANIFEST_SHA256.csv",
    ]
    missing = [item for item in required if not (root / item).exists()]
    check(not missing, f"required files present (missing: {missing})")

    # 2. Manifest hashes.
    if not missing:
        manifest = pd.read_csv(root / "MANIFEST_SHA256.csv")
        bad = []
        for _, row in manifest.iterrows():
            path = root / row["relative_path"]
            if not path.exists():
                bad.append(f"{row['relative_path']} missing")
                continue
            if sha256(path) != row["sha256"]:
                bad.append(f"{row['relative_path']} hash mismatch")
        check(not bad, f"manifest hashes (mismatch: {bad[:5]})")
        discovered = {
            path.relative_to(root).as_posix()
            for path in package_files(root)
        }
        listed = set(manifest.relative_path.astype(str))
        check(
            listed == discovered,
            f"manifest coverage (missing={sorted(discovered-listed)[:5]}, extra={sorted(listed-discovered)[:5]})",
        )

    # 3. Frozen lock.
    lock = json.loads((root / "config/controller_lock.json").read_text(encoding="utf-8"))
    labels = {item["label"] for item in lock["selected"]}
    check(
        labels == {"C7-Core-g0.75-mpc0.14", "PID-g1.20-du0.60", "ADRC-g1.20-du0.95", "C2-g1.20-du0.80"},
        "frozen controller labels",
    )
    c2 = [item for item in lock["selected"] if item["label"] == "C2-g1.20-du0.80"][0]
    check(c2.get("use_filter") is False, "C2 lock executes without EKF observer (use_filter=false)")

    scale_lock = load_scale_lock(root / "config/data1_scale_lock.json")
    scale_entries = pd.DataFrame(scale_lock["entries"])
    check(len(scale_entries) == 13 and scale_entries.pass_id.nunique() == 13,
          "Data1 scale catalogue contains exactly 13 unique passes")
    check(
        np.isfinite(scale_entries[list(SCALE_FIELDS)].to_numpy(float)).all()
        and (scale_entries[list(SCALE_FIELDS)].to_numpy(float) > 0).all(),
        "all six frozen scale fields are finite and positive",
    )

    # 4. Publication-number cross-matches.
    canonical = pd.read_csv(root / "results/canonical_nominal_condition_weighted.csv").set_index("policy")
    canonical.index = canonical.index.map(lambda name: {"C7": "C7-Core"}.get(name, name))
    check(len(canonical) == 4, "four canonical policies")
    bootstrap = pd.read_csv(root / "results/paired_effect_bootstrap.csv")
    check(len(bootstrap) == 12, "twelve paired bootstrap rows")
    for _, row in bootstrap.iterrows():
        check(row.point_pct >= row.ci_low_pct - 1e-9 and row.point_pct <= row.ci_high_pct + 1e-9,
              f"bootstrap point inside CI: {row.metric} {row.comparison}")
    check((bootstrap.point_pct.abs() < 60).all(), "bootstrap effects within sane range")

    band_file = pd.read_csv(
        root / "results/recorded_band_decomposition_by_file.csv"
    )
    band_condition = pd.read_csv(
        root / "results/recorded_band_decomposition_by_condition.csv"
    )
    band_summary = pd.read_csv(
        root / "results/recorded_band_decomposition_summary.csv"
    ).set_index("policy")
    band_guardrail = pd.read_csv(
        root / "results/recorded_band_guardrail_row_composition.csv"
    )
    band_audit = json.loads(
        (root / "results/recorded_band_decomposition_audit.json").read_text(
            encoding="utf-8"
        )
    )
    check(
        len(band_file) == 40
        and len(band_condition) == 32
        and set(band_summary.index) == {"PID", "OC-PID", "C2", "C7-Core"}
        and band_summary.n_conditions.astype(int).eq(8).all(),
        "recorded-band decomposition has 40 policy-pass and 32 policy-condition rows",
    )
    check(
        np.allclose(
            band_file.two_sided_mean_distance,
            band_file.below_mean_distance + band_file.above_mean_distance,
            atol=1e-12, rtol=0,
        )
        and np.allclose(
            band_file.two_sided_cvar95,
            band_file.below_cvar95_component + band_file.above_cvar95_component,
            atol=1e-12, rtol=0,
        ),
        "recorded-band mean and common-tail components are exactly additive",
    )
    for policy in ["PID", "OC-PID", "C2", "C7-Core"]:
        check(
            abs(
                float(band_summary.loc[policy, "recorded_band_departure_frequency"])
                - float(canonical.loc[policy, "S_out"])
            ) < 1e-12
            and abs(
                float(band_summary.loc[policy, "two_sided_mean_distance"])
                - float(canonical.loc[policy, "S_excess_mean"])
            ) < 1e-12
            and abs(
                float(band_summary.loc[policy, "two_sided_cvar95"])
                - float(canonical.loc[policy, "S_excess_cvar95"])
            ) < 1e-12,
            f"recorded-band decomposition reproduces legacy anchors {policy}",
        )
    # Numerical values depend on the implemented predictor timeline. Do not
    # freeze old values as "truth": independently rebuild the declared equal-
    # file/equal-condition aggregation from the released per-file ledger.
    band_metrics = [
        "below_frequency", "above_frequency", "recorded_band_departure_frequency",
        "below_mean_distance", "above_mean_distance", "two_sided_mean_distance",
        "below_cvar95_component", "above_cvar95_component", "two_sided_cvar95",
        "S_normalized_mean", "S_normalized_median", "S_normalized_rms",
    ]
    band_counts = ["n_rows", "below_count", "inside_count", "above_count"]
    file_groups = band_file.groupby(["policy", "condition_id"], sort=True)
    rebuilt_condition = file_groups[band_metrics].mean().join(
        file_groups[band_counts].sum()
    ).join(file_groups.size().rename("n_files")).reset_index()
    condition_groups = rebuilt_condition.groupby("policy", sort=True)
    rebuilt_summary = condition_groups[band_metrics].mean().join(
        condition_groups[[*band_counts, "n_files"]].sum()
    ).join(condition_groups.size().rename("n_conditions")).reset_index()
    check(
        numeric_tables_equal(
            rebuilt_condition, band_condition, ["policy", "condition_id"],
            [*band_metrics, *band_counts, "n_files"],
        )
        and numeric_tables_equal(
            rebuilt_summary, band_summary.reset_index(), ["policy"],
            [*band_metrics, *band_counts, "n_files", "n_conditions"],
        )
        and (band_file["below_count"] + band_file["inside_count"]
             + band_file["above_count"]).eq(band_file["n_rows"]).all(),
        "recorded-band components independently reproduce file-to-condition-to-summary aggregation",
    )
    s_guardrail_rows = band_guardrail.loc[
        band_guardrail.row_scope.eq("after_S_guardrail")
    ]
    any_guardrail_rows = band_guardrail.loc[
        band_guardrail.row_scope.eq("after_any_channel_guardrail")
    ]
    alignment = band_audit.get("guardrail_alignment", {})
    original_flags = alignment.get("original_flag_counts", {})
    reached_flags = alignment.get("aligned_reached_row_counts", {})
    reached_s_count = int(s_guardrail_rows.scope_rows.sum())
    reached_any_count = int(any_guardrail_rows.scope_rows.sum())
    positive_s_count = int(s_guardrail_rows.positive_8_boundary_rows.sum())
    negative_s_count = int(s_guardrail_rows.negative_8_boundary_rows.sum())
    check(
        set(band_guardrail.row_scope) == {
            "after_S_guardrail", "after_any_channel_guardrail"
        }
        and not band_guardrail.duplicated(["policy", "condition_id", "row_scope"]).any()
        and (band_guardrail.below_count + band_guardrail.inside_count
             + band_guardrail.above_count).eq(band_guardrail.scope_rows).all()
        and (band_guardrail.scope_rows >= 0).all()
        and (band_guardrail.scope_rows <= band_guardrail.condition_rows).all()
        and reached_s_count == positive_s_count + negative_s_count
        and reached_s_count == original_flags.get("S_active")
        == reached_flags.get("S_active_reached")
        and reached_any_count == original_flags.get("any_active")
        == reached_flags.get("any_active_reached")
        and original_flags.get("transition_evaluated")
        == reached_flags.get("transition_reached")
        and positive_s_count == alignment.get("aligned_S_rows_at_positive_8")
        and negative_s_count == alignment.get("aligned_S_rows_at_negative_8")
        and band_audit.get("status") == "PASS"
        and alignment.get("segment_first_rows_with_aligned_flag") == 0,
        "shifted guardrail ledger reconciles transition/reached-row counts and boundary signs without segment leakage",
    )

    pass_metrics = pd.read_csv(root / "results/canonical_nominal_pass_metrics.csv")
    data1_ids = set(scale_entries.pass_id.astype(str))
    check(set(pass_metrics.scale_match_pass_id.astype(str)).issubset(data1_ids),
          "canonical Data2 scale sources are all Data1 passes")
    check(
        pass_metrics.scale_lock_sha256.astype(str).eq(scale_lock["payload_sha256"]).all(),
        "canonical metrics carry the frozen scale-lock hash",
    )
    catalogue = scale_entries.set_index("pass_id")
    numeric_match = True
    for _, row in pass_metrics.iterrows():
        source = str(row.scale_match_pass_id)
        if source not in catalogue.index:
            numeric_match = False
            break
        for field in SCALE_FIELDS:
            if abs(float(row[field]) - float(catalogue.loc[source, field])) > 1e-12:
                numeric_match = False
                break
    check(numeric_match, "canonical scale values match their labelled Data1 catalogue rows")
    scale_application = pd.read_csv(root / "results/data1_scale_lock_application.csv")
    expected_matches = {
        "P05": "P23", "P06": "P24", "P07": "P24", "P08": "P01",
        "P09": "P02", "P10": "P03", "P11": "P04", "P15": "P14",
        "P16": "P14", "P17": "P12", "P18": "P13", "P19": "P14", "P20": "P14",
    }
    actual_matches = scale_application.set_index("pass_id")["scale_match_pass_id"].astype(str).to_dict()
    check(all(actual_matches.get(pid) == source for pid, source in expected_matches.items()),
          "evaluation-pass Data1 scale-source mapping")

    rate = pd.read_csv(root / "results/Table_S12_rate_sensitivity.csv")
    anchor = rate[rate["Rate multiplier"].eq(1.0)].set_index("Policy")
    for policy in ["PID", "OC-PID", "C2", "C7-Core"]:
        check(
            abs(float(anchor.loc[policy, "Mean two-sided band distance"]) - float(canonical.loc[policy, "S_excess_mean"])) < 1e-12,
            f"rate 1.00 anchor mean two-sided band distance {policy}",
        )
        check(
            abs(float(anchor.loc[policy, "TV/100 m"]) - float(canonical.loc[policy, "TV_L_per_100m"])) < 1e-12,
            f"rate 1.00 anchor TV {policy}",
        )
        check(
            abs(float(anchor.loc[policy, "Amplitude active"]) - float(canonical.loc[policy, "amplitude_projection_ratio"])) < 1e-12,
            f"rate 1.00 anchor amplitude {policy}",
        )

    opc = pd.read_csv(root / "results/opcua_loopback_summary.csv")
    check(str(opc.loc[opc["Item"].eq("Command-boundary violations"), "Result"].iloc[0]) in {"0", "0.0"},
          "OPC UA boundary count zero")

    corridor = json.loads((root / "results/coverage_sensitivity_hard_checks.json").read_text(encoding="utf-8"))
    corridor_delta = max(float(value) for value in corridor[
        "coverage_095_matches_canonical_by_metric_max_abs_delta"
    ].values())
    check(np.isfinite(corridor_delta) and corridor_delta <= 1e-12,
          "95% corridor metrics exactly match canonical eight-condition aggregation")

    time_qa = pd.read_csv(root / "results/time_protocol_numeric_QA.csv")
    check(time_qa.effective_source_count_sum_equals_rows.astype(bool).all(),
          "effective time-source counts reconcile to every input row")
    source_cols = [
        "effective_raw_timestamp_count", "effective_distance_over_speed_count",
        "segment_start_local_median_count", "segment_start_global_median_count",
        "segment_local_median_fallback_count",
    ]
    check((time_qa[source_cols].sum(axis=1) == time_qa.rows).all(),
          "numeric time-source count sum equals rows")

    guard = pd.read_csv(root / "results/output_guardrail_activation_summary.csv")
    check(
        (guard.event_count <= guard.eligible_transitions).all()
        and (guard.T_count <= guard.eligible_transitions).all()
        and (guard.h_count <= guard.eligible_transitions).all()
        and (guard.S_count <= guard.eligible_transitions).all()
        and (guard.event_count >= guard[["T_count", "h_count", "S_count"]].max(axis=1)).all()
        and (guard.event_count <= guard[["T_count", "h_count", "S_count"]].sum(axis=1)).all(),
        "output-guardrail event/channel counts reconcile to eligible transitions",
    )
    guard_segment = pd.read_csv(
        root / "results/output_guardrail_activation_by_segment.csv"
    )
    canonical_guard = guard.loc[guard.output_clip_scale.eq(8.0)].set_index("policy")
    segment_totals = guard_segment.groupby("policy")[[
        "eligible_transitions", "event_count", "T_count", "h_count", "S_count"
    ]].sum()
    check(
        set(segment_totals.index) == set(canonical_guard.index)
        and all(
            int(segment_totals.loc[policy, column])
            == int(canonical_guard.loc[policy, column])
            for policy in segment_totals.index
            for column in [
                "eligible_transitions", "event_count", "T_count", "h_count", "S_count"
            ]
        ),
        "canonical output-guardrail segment ledger reconciles to scale-8 summary",
    )
    guard_condition = pd.read_csv(
        root / "results/output_guardrail_activation_by_condition.csv"
    )
    scale8_conditions = guard_condition.loc[guard_condition.output_clip_scale.eq(8.0)]
    scope_to_channel = {
        "after_S_guardrail": "S_count",
        "after_any_channel_guardrail": "event_count",
    }
    guard_scope_match = True
    for scope, count_column in scope_to_channel.items():
        composition = band_guardrail.loc[band_guardrail.row_scope.eq(scope)]
        ledger = scale8_conditions[["policy", "condition_id", count_column]].rename(
            columns={count_column: "scope_rows"}
        )
        guard_scope_match = guard_scope_match and numeric_tables_equal(
            composition, ledger, ["policy", "condition_id"], ["scope_rows"]
        )
    check(
        guard_scope_match
        and reached_s_count == int(canonical_guard.S_count.sum())
        and reached_any_count == int(canonical_guard.event_count.sum())
        and original_flags.get("transition_evaluated")
        == int(canonical_guard.eligible_transitions.sum()),
        "reached-row guardrail composition independently reconciles condition and canonical transition ledgers",
    )

    # Execute the timeline regression tests, not merely a published PASS flag.
    # -B prevents package __pycache__ writes; no --audit-json is supplied here.
    timeline_audit = json.loads(
        (root / "results/predictor_timeline_audit.json").read_text(encoding="utf-8")
    )
    test_environment = os.environ.copy()
    test_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        timeline_run = subprocess.run(
            [sys.executable, "-B", str(root / "tests/test_predictor_timeline.py")],
            cwd=root, env=test_environment, text=True, capture_output=True,
            timeout=60, check=False,
        )
        timeline_output = timeline_run.stdout + timeline_run.stderr
        test_count_match = re.search(r"Ran\s+(\d+)\s+tests?", timeline_output)
        actual_tests = int(test_count_match.group(1)) if test_count_match else 0
        timeline_passed = timeline_run.returncode == 0 and actual_tests >= 8
    except (OSError, subprocess.TimeoutExpired):
        actual_tests = 0
        timeline_passed = False
    check(
        timeline_passed
        and timeline_audit.get("protocol")
        == "observation/control row k -> target archived row k+1"
        and timeline_audit.get("passed") is True
        and timeline_audit.get("tests_run") == actual_tests
        and timeline_audit.get("failures") == 0
        and timeline_audit.get("errors") == 0
        and timeline_audit.get("checkpoint_sha256")
        == sha256(root / "models/final_thickness_model.pt"),
        "predictor timeline is independently executed offline with the unchanged frozen checkpoint",
    )

    # Aggregate consistency is not complete historical training provenance.
    # Recompute the available evidence while explicitly retaining unknowns.
    from predictor_training_audit import audit_published

    training_audit = json.loads(
        (root / "results/predictor_training_provenance_audit.json").read_text(
            encoding="utf-8"
        )
    )
    recomputed_training_audit = audit_published(root)
    incomplete_status = "aggregate_consistent_but_historical_training_provenance_incomplete"
    unknown_fields = [
        "actual_frozen_training_updates", "actual_frozen_training_seed",
        "actual_frozen_training_device", "actual_frozen_training_optimizer",
    ]
    check(
        training_audit.get("status") == incomplete_status
        == recomputed_training_audit.get("status")
        and training_audit.get("checks") == recomputed_training_audit.get("checks")
        and all(item.get("passed") is True for item in training_audit.get("checks", []))
        and training_audit.get("source_summary_sha256")
        == sha256(root / "results/predictor_lopo_one_se_summary.csv")
        and training_audit.get("frozen_checkpoint_sha256")
        == sha256(root / "models/final_thickness_model.pt")
        and training_audit.get("checkpoint_fields_present")
        == recomputed_training_audit.get("checkpoint_fields_present")
        and training_audit.get("historical_fold_seed_scores_available") is False
        and training_audit.get("historical_training_curves_available") is False
        and all(field in training_audit and training_audit[field] is None
                for field in unknown_fields)
        and len(training_audit.get("limitations", [])) >= 4,
        "predictor aggregates are consistent while historical fold/training provenance remains explicitly incomplete",
    )
    architecture = pd.read_csv(root / "results/strict_c7_component_ablation.csv")
    check(set(architecture.Variant) == {"C7-Core", "C7-NoMPC", "C7-NoBlend"},
          "strict C7 component ablation has three registered variants")
    farch_summary = pd.read_csv(root / "results/farch_forward_summary.csv")
    farch_occupancy = pd.read_csv(
        root / "results/farch_recorded_envelope_occupancy.csv"
    )
    farch_audit = json.loads(
        (root / "results/farch_forward_audit.json").read_text(encoding="utf-8")
    )
    h1_check = farch_audit["hard_checks"][
        "rolling_origin_h1_vs_stored_innovations"
    ]
    check(
        len(farch_summary) == 48
        and set(farch_summary.aggregation) == {"pooled_origins", "equal_pass_mean"}
        and set(farch_summary.guardrail_label) == {"scale_8", "infinite"}
        and set(farch_summary.horizon_updates.astype(int)) == {1, 5, 10, 20}
        and set(farch_summary.channel) == {"T", "h", "S"}
        and len(farch_occupancy) == 96
        and set(farch_occupancy.envelope_scope)
        == {"frozen_all_data1", "leave_one_pass_out"}
        and farch_audit.get("diagnostic_id")
        == "data1_farch_zero_innovation_forward_discrepancy_v1"
        and farch_audit.get("inputs", {}).get("batch_root") == "<BATCH_ROOT>"
        and farch_audit.get("counts", {}).get("passes_evaluated") == 13
        and farch_audit.get("counts", {}).get("prediction_rows") == 115528
        and h1_check.get("bitwise_equal") is True
        and h1_check.get("within_machine_precision") is True
        and float(h1_check.get("max_abs_difference", 1.0)) == 0.0
        and farch_audit["hard_checks"].get("all_predictions_finite") is True
        and farch_audit["hard_checks"].get("all_targets_within_segment") is True
        and farch_audit["hard_checks"].get(
            "all_required_guardrail_horizon_pairs_present"
        ) is True
        and farch_audit["hard_checks"]["frozen_recorded_envelope"].get(
            "recomputed_matches_frozen"
        ) is True,
        "Data1 zero-innovation F_arch forward discrepancy audit is complete and segment disciplined",
    )
    replay_distribution = pd.read_csv(
        root / "results/replay_flatness_distribution_summary.csv"
    ).set_index("series")
    replay_distribution_audit = json.loads(
        (root / "results/replay_flatness_distribution_audit.json").read_text(
            encoding="utf-8"
        )
    )
    expected_distribution_series = {
        "Recorded Data2", "PID", "OC-PID", "C2", "C7-Core"
    }
    distribution_shares = replay_distribution[
        ["central_share_locked", "moderate_share_locked", "extreme_share_locked"]
    ].sum(axis=1)
    check(
        set(replay_distribution.index) == expected_distribution_series
        and replay_distribution.n_rows.astype(int).eq(5513).all()
        and np.allclose(distribution_shares.to_numpy(float), 1.0, rtol=0.0, atol=2e-15)
        and replay_distribution_audit.get("all_checks_passed") is True
        and replay_distribution_audit.get("same_row_keys_for_every_policy") is True
        and replay_distribution_audit.get("canonical_residual_gain") == 0.82
        and replay_distribution_audit.get("canonical_innovations_and_commands_retained") is True
        and replay_distribution_audit.get("maturity_thresholds_recomputed_from_data1_only") is True
        and replay_distribution_audit.get("published_maturity_summary_reproduced") is True
        and replay_distribution_audit.get("published_maturity_thresholds_reproduced") is True,
        "canonical replay flatness distribution uses identical 5513-row keys and Data1-locked maturity bands",
    )
    pid_audit = json.loads((root / "results/pid_selection_audit.json").read_text(encoding="utf-8"))
    pid_locked_result = json.loads(
        (root / "results/pid_locked_params.json").read_text(encoding="utf-8")
    )
    pid_grid = pd.read_csv(root / "results/pid_full_grid.csv")
    pid_shell = pid_audit.get("outer_shell", {})
    standalone_pid = json.loads(
        (root / "config/locked_pid.json").read_text(encoding="utf-8")
    )
    reference_pid = json.loads(
        (
            root / "config/reference_run/03_nominal/"
            "stage_allocation_ablation_manifest.json"
        ).read_text(encoding="utf-8")
    )["locked_pid"]
    pid_parameter_columns = [
        "kp_scale", "ki_scale", "kd_scale", "move_limit_scale"
    ]
    declared_pid_grid = {
        tuple(float(value) for value in candidate)
        for candidate in itertools.product(
            [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
            [0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.6, 0.8, 1.0, 1.2, 1.4],
        )
    }
    pid_numeric = pid_grid.loc[:, pid_parameter_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    observed_pid_grid = {
        tuple(float(value) for value in row)
        for row in pid_numeric.itertuples(index=False, name=None)
    }
    pid_keys_match_rows = all(
        str(key)
        == f"kp{values[0]:g}_ki{values[1]:g}_kd{values[2]:g}_ml{values[3]:g}"
        for key, values in zip(
            pid_grid.candidate_key.astype(str),
            pid_numeric.itertuples(index=False, name=None),
        )
    )
    winning_row = pid_grid.sort_values(
        ["data1_equal_pass_mean_RMS_c", "candidate_key"], kind="mergesort"
    ).iloc[0]
    winning_pid = {
        name: float(winning_row[name]) for name in pid_parameter_columns
    }
    published_locked_pid = {
        name: float(pid_locked_result[name]) for name in pid_parameter_columns
    }
    check(
        pid_audit.get("candidate_count") == 1050
        and pid_audit.get("data1_pass_count") == 13
        and pid_audit.get("selection_is_leave_one_pass_out") is False
        and pid_audit.get("selection_protocol_id") == "canonical_pid_replay_segment_gap_v2"
        and pid_audit.get("selection_metric", {}).get("field") == "composite_normalized_RMS"
        and pid_audit.get("data_access_scope", {}).get("included_batches") == ["I", "IV", "VII"]
        and pid_audit.get("data_access_scope", {}).get("data2_files_opened_by_selection") is False
        and pid_audit.get("selected_matches_frozen_lock") is True
        and pid_audit.get("selected")
        == pid_audit.get("frozen_manifest_lock")
        == standalone_pid
        == reference_pid
        == winning_pid
        == published_locked_pid
        and abs(
            float(winning_row["data1_equal_pass_mean_RMS_c"])
            - float(pid_audit.get("selected_score_RMS_c", np.inf))
        ) <= 1e-12
        and abs(
            float(winning_row["data1_equal_pass_mean_RMS_c"])
            - float(pid_locked_result.get("score_RMS_c", np.inf))
        ) <= 1e-12
        and pid_locked_result.get("score_metric") == "composite_normalized_RMS"
        and pid_locked_result.get("selection_protocol_id")
        == "canonical_pid_replay_segment_gap_v2"
        and pid_audit.get("selected_all_replay_contract_checks_pass") is True
        and float(pid_audit.get("selected_max_composite_reconstruction_abs_delta", 1.0)) <= 1e-12
        and float(pid_audit.get("selected_grid_vs_replay_score_abs_delta", 1.0)) <= 1e-12
        and len(pid_grid) == 1050
        and not pid_grid.candidate_key.astype(str).duplicated().any()
        and np.isfinite(pid_numeric.to_numpy(dtype=float)).all()
        and observed_pid_grid == declared_pid_grid
        and pid_keys_match_rows
        and set(pid_grid.selection_protocol_id.astype(str))
        == {"canonical_pid_replay_segment_gap_v2"}
        and abs(float(pid_shell.get("amplitude_bound_each_channel", -1.0)) - 0.55) <= 1e-15
        and abs(float(pid_shell.get("outer_du_scale", -1.0)) - 0.60) <= 1e-15
        and pid_shell.get("nominal_move_vector") == [0.039, 0.0312, 0.039]
        and abs(float(pid_shell.get("legacy_move_limit_reference_dt_s", -1.0)) - 0.50) <= 1e-15
        and abs(float(pid_shell.get("outer_du_hard_scale", -1.0)) - 2.0) <= 1e-15,
        "canonical-replay equal-pass Data1 PID grid reproduces the frozen inner lock and fixed outer shell",
    )

    # 5. Figures.
    # Execute the production tail functions, then independently reconcile the
    # public canonical per-pass counts. No audit files or caches are written.
    tail_message = "not run"
    tail_ok = False
    try:
        tail_run = subprocess.run(
            [sys.executable, "-B", str(root / "tests/test_tail_statistics.py")],
            cwd=root, capture_output=True, text=True, timeout=60,
        )
        tail_output = tail_run.stdout + tail_run.stderr
        tail_tests = re.search(r"Ran (\d+) tests?", tail_output)
        from public_statistics_audit import tail_cardinality_audit
        tail_audit = tail_cardinality_audit(root / "results")
        tail_ok = (
            tail_run.returncode == 0
            and tail_tests is not None
            and int(tail_tests.group(1)) >= 9
            and tail_audit["passed"] is True
            and tail_audit["pass_policy_rows"] == 40
            and tail_audit["max_count_difference"] == 0.0
        )
        tail_message = f"rc={tail_run.returncode}, canonical count mismatches={len(tail_audit['mismatches'])}"
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        tail_message = str(exc)
    check(tail_ok, f"exact decimal upper-tail regression and independent pass-count audit ({tail_message})")

    figure_dir = root / "figures"
    figures = ["Figure_1.png", "Figure_S1.png", "Figure_2.png",
               "Figure_S2.png", "Figure_3.png", "Figure_4.png", "Figure_S3.png"]
    missing_figs = [f for f in figures if not (figure_dir / f).exists()]
    check(not missing_figs, f"figure assets (missing: {missing_figs})")

    # 6. Fault and seed files.
    faults = pd.read_csv(root / "results/fault_diagnostic_summary.csv")
    fault_events = pd.read_csv(root / "results/fault_event_log.csv")
    fault_confusion = pd.read_csv(root / "results/fault_diagnostic_confusion_matrix.csv")
    expected_faults = {"bad_status", "model_failure", "stale", "input_unit_consistency"}
    check(
        len(faults) == 4
        and set(faults.fault_family) == expected_faults
        and int(faults.events.sum()) == len(fault_events) == int(fault_confusion.events.sum()) == 604,
        "four fault classes and reconciled 604-event ledger",
    )
    fault_by_name = faults.set_index("fault_family")
    expected_stale_rate = 199.0 / 200.0
    check(
        float(faults.bounded_command_rate.min()) == 1.0
        and float(fault_by_name.loc[["bad_status", "model_failure", "input_unit_consistency"], "detection_rate"].min()) == 1.0
        and float(fault_by_name.loc[["bad_status", "model_failure", "input_unit_consistency"], "localization_accuracy"].min()) == 1.0
        and abs(float(fault_by_name.loc["stale", "detection_rate"]) - expected_stale_rate) < 1e-12
        and abs(float(fault_by_name.loc["stale", "localization_accuracy"]) - expected_stale_rate) < 1e-12,
        "fault detection/localization and one-cycle stale onset match protocol",
    )
    seeds = pd.read_csv(root / "results/seed_sensitivity_summary.csv")
    check(len(seeds) >= 2 and float(seeds.max_abs_delta_pct_of_canonical.max()) < 5.0,
          "seed stability within 5% of canonical values")
    stress = pd.read_csv(root / "results/stress_clustered_effects.csv")
    check(
        len(stress) == 160
        and stress.scenario.nunique() == 20
        and set(stress.baseline) == {"PID", "ADRC"}
        and set(stress.metric) == {"RMS_c", "TV", "S_excess_mean", "S_excess_p95"}
        and stress.n_conditions.astype(int).eq(8).all()
        and stress.invalid_c7.astype(int).eq(0).all()
        and stress.invalid_baseline.astype(int).eq(0).all(),
        "stress publication table contains the current 20-scenario by two-baseline by four-metric analysis",
    )

    motion_env = os.environ.copy()
    motion_env["PYTHONDONTWRITEBYTECODE"] = "1"
    motion_env["MPLBACKEND"] = "Agg"
    motion_run = subprocess.run(
        [sys.executable, "-B", str(root / "tests/test_supported_command_motion.py")],
        cwd=root, capture_output=True, text=True, env=motion_env,
    )
    motion_output = motion_run.stdout + motion_run.stderr
    motion_count = re.search(r"Ran\s+(\d+)\s+tests?", motion_output)
    check(motion_run.returncode == 0 and motion_count is not None and int(motion_count.group(1)) >= 17,
          "executed supported-segment motion regression tests including startup and actual calculation paths")
    motion_audit_run = subprocess.run(
        [sys.executable, "-B", str(root / "code/supported_motion_audit.py")],
        cwd=root, capture_output=True, text=True, env=motion_env,
    )
    check(motion_audit_run.returncode == 0,
          "independent current public supported-motion aggregate identities")
    motion_receipt = json.loads((root / "results/supported_motion_audit.json").read_text(encoding="utf-8"))
    check(motion_receipt.get("passed") is True and motion_receipt.get("raw_trajectory_reconstruction_executed") is True
          and motion_receipt.get("private_log_check", {}).get("groups") == 40
          and motion_receipt.get("private_log_check", {}).get("max_abs_published_difference", float("inf")) <= 1e-10,
          "recorded canonical private-log motion reconstruction, distinct from current offline checks")

    if failures:
        print("FAIL:", len(failures), "failed checks")
        for item in failures:
            print("  -", item)
        return 1
    print(f"PASS: {checks_passed} checks verified (files, manifest hashes, lock, anchors, figures, faults, seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
