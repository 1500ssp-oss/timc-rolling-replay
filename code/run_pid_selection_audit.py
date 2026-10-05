"""Recompute and audit the Data1 PID candidate grid through the released replay.

Every candidate is evaluated by the same ``run_exposure_control`` path used by
``batch_sweep.py``.  The candidate changes only the locked PID's gain scales
and internal saturation bounds; the canonical PID measurement path,
seed-dependent segment-start estimator initialization, segmented
archived-update clock, numerical state guardrail, amplitude bound and outer
slew projection remain fixed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PID_SELECTION_PROTOCOL_ID = "canonical_pid_replay_segment_gap_v2"
PID_SELECTION_CONFIG_LABEL = "PID-g1.20-du0.60"
DATA1_SELECTION_PASS_ORDER = (
    "P14", "P13", "P12", "P04", "P03", "P02", "P01",
    "P26", "P25", "P24", "P23", "P22", "P21",
)
CHANNEL_RMS_FIELDS = (
    "T_error_normalized_RMS",
    "h_error_normalized_RMS",
    "S_error_normalized_RMS",
)
COMPOSITE_FIELD = "composite_normalized_RMS"


_WORKER_RUNNER = None
_WORKER_SUITE = None
_WORKER_DATA1 = None
_WORKER_CONFIG = None
_WORKER_ENVELOPE = None
_WORKER_SEED_INDEX = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_pid_config(runner, package: Path):
    manifest_path = (
        package / "config" / "reference_run" / "03_nominal" /
        "stage_allocation_ablation_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        item for item in manifest["configs"]
        if item.get("label") == PID_SELECTION_CONFIG_LABEL
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one {PID_SELECTION_CONFIG_LABEL!r} configuration, found {len(matches)}"
        )
    config = runner.ExposureConfig(**matches[0])
    expected = {
        "base_control": "C0",
        "use_filter": False,
        "force_common_outer_limits": True,
        "pid_gain_scale": 1.20,
        "outer_du_scale": 0.60,
        "clamp_scale": 1.0,
        "legacy_move_limit_reference_dt_s": 0.50,
        "outer_du_hard_scale": 2.0,
        "distance_step_reference_m": 0.81298828125,
        "gap_ratio_low": 0.40,
        "gap_ratio_high": 3.00,
        "output_clip_scale": 8.0,
    }
    for name, value in expected.items():
        observed = getattr(config, name)
        if isinstance(value, float):
            if not np.isclose(float(observed), value, rtol=0.0, atol=1e-15):
                raise AssertionError(f"Canonical PID {name}={observed!r}, expected {value!r}")
        elif observed != value:
            raise AssertionError(f"Canonical PID {name}={observed!r}, expected {value!r}")
    if config.amplitude_bound is not None:
        raise AssertionError("Canonical PID amplitude bound must resolve from clamp_scale")
    if config.mpc_share != 0.0 or config.ff_share != 0.0:
        raise AssertionError("Canonical PID selection configuration must not use MPC/feedforward")
    return config, manifest


def _load_data(batch_root: str):
    code = Path(__file__).resolve().parent
    package = code.parent
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    tag = str(os.getpid())
    runner = load_module(f"pid_audit_runner_{tag}", code / "controller_replay.py")
    archive = load_module(f"pid_audit_archive_{tag}", code / "batch_archive.py")
    archive.BATCH_ROOT = Path(batch_root)
    runner.protocol.PROJECT_DIR = code / "engine_core"
    suite, replay = runner.protocol.load_suite_modules()
    passes, inventory = archive.load_batch_passes(
        runner.protocol, suite, replay, included_batches={"I", "IV", "VII"}
    )
    if len(inventory) != 13 or set(inventory["source"].astype(str)) != {"Data1"}:
        raise AssertionError("PID selection loader must contain exactly 13 Data1 rows")
    by_id = {rp.pass_id: rp for rp in passes if rp.pass_id in archive.DATA1_IDS}
    if set(by_id) != set(DATA1_SELECTION_PASS_ORDER):
        raise AssertionError(
            f"Expected Data1 passes {list(DATA1_SELECTION_PASS_ORDER)}, found {sorted(by_id)}"
        )
    data1 = [by_id[pass_id] for pass_id in DATA1_SELECTION_PASS_ORDER]
    config, manifest = _canonical_pid_config(runner, package)
    envelope = pd.read_csv(
        package / "config" / "reference_run" / "03_nominal" / "audit" /
        "data1_stage_quantile_envelope_p025_p975.csv"
    )
    seed_index = {pass_id: index for index, pass_id in enumerate(DATA1_SELECTION_PASS_ORDER)}
    return runner, suite, data1, config, envelope, seed_index, manifest


def canonical_data1_seed(runner, pass_id: str, seed_index: dict[str, int]) -> int:
    """Return the exact Data1 seed used by ``batch_sweep.py`` at repeat zero."""
    return int(runner.RNG_SEED + 2000 * (seed_index[pass_id] + 1))


def score_pid_pass(
    runner,
    suite,
    replay_pass,
    config,
    envelope: pd.DataFrame,
    params,
    seed: int,
    *,
    verify_replay_contract: bool = False,
) -> dict[str, Any]:
    """Score one candidate/pass through the canonical batch-sweep code path."""
    curve, diag = runner.run_exposure_control(
        suite, replay_pass, config, int(seed), params, 0
    )
    metrics = runner.suite_metrics_with_config(
        suite, replay_pass, curve, diag, config, envelope
    )
    channel_rms = [float(metrics[name]) for name in CHANNEL_RMS_FIELDS]
    recomputed = float(np.mean(channel_rms))
    composite = float(metrics[COMPOSITE_FIELD])
    if not np.isclose(composite, recomputed, rtol=0.0, atol=1e-12):
        raise AssertionError(
            f"{replay_pass.pass_id}: composite RMS {composite} != channel mean {recomputed}"
        )
    result: dict[str, Any] = {
        "pass_id": replay_pass.pass_id,
        "seed": int(seed),
        "T_normalized_RMS": channel_rms[0],
        "h_normalized_RMS": channel_rms[1],
        "S_normalized_RMS": channel_rms[2],
        "composite_normalized_RMS": composite,
        "composite_recomputed_from_channels": recomputed,
        "composite_reconstruction_abs_delta": abs(composite - recomputed),
        "logged_row_count": int(len(curve)),
        "reported_segment_count": int(diag["segments"]),
    }
    if not verify_replay_contract:
        return result

    elapsed, dt_s, distance_step, segment_ids, sources = runner.replay_timebase(
        replay_pass.df, config
    )
    del elapsed, distance_step
    expected_segments = int(segment_ids.max() + 1) if len(segment_ids) else 0
    logged_segments = curve["segment_id"].to_numpy(dtype=int)
    logged_sources = curve["effective_time_source"].astype(str).to_numpy()
    expected_sources = np.asarray(sources, dtype=str)
    expected_valid_transitions = int(len(segment_ids) - expected_segments)
    logged_valid_transitions = int(curve["output_transition_evaluated"].sum())
    source_counts = pd.Series(logged_sources).value_counts().sort_index().to_dict()
    expected_source_counts = pd.Series(expected_sources).value_counts().sort_index().to_dict()

    base_move = np.array([0.065, 0.052, 0.065], dtype=float)
    nominal_move = base_move * float(config.outer_du_scale)
    expected_du_limit = np.minimum(
        dt_s[:, None] * nominal_move[None, :] /
        float(config.legacy_move_limit_reference_dt_s),
        nominal_move[None, :] * float(config.outer_du_hard_scale),
    )
    actual_du = np.abs(curve[["du_speed", "du_gap", "du_shape"]].to_numpy(float))
    max_du_violation = float(np.max(actual_du - expected_du_limit)) if len(curve) else 0.0
    amplitude_bound = 0.55 * float(config.clamp_scale)
    actual_u = np.abs(curve[["u_speed", "u_gap", "u_shape"]].to_numpy(float))
    max_amplitude_violation = float(np.max(actual_u - amplitude_bound)) if len(curve) else 0.0

    checks = {
        "row_count_matches_input": len(curve) == len(replay_pass.df),
        "segment_ids_match_formal_timebase": np.array_equal(logged_segments, segment_ids),
        "timebase_sources_match_formal_timebase": np.array_equal(logged_sources, expected_sources),
        "segment_count_matches_formal_timebase": int(diag["segments"]) == expected_segments,
        "valid_transition_count_matches_segments": logged_valid_transitions == expected_valid_transitions,
        "outer_slew_limits_respected": max_du_violation <= 1e-12,
        "outer_amplitude_bound_respected": max_amplitude_violation <= 1e-12,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"{replay_pass.pass_id}: replay contract failed: {failed}")
    result.update(
        {
            "formal_segment_count": expected_segments,
            "logged_valid_transition_count": logged_valid_transitions,
            "formal_valid_transition_count": expected_valid_transitions,
            "logged_timebase_source_counts": {str(k): int(v) for k, v in source_counts.items()},
            "formal_timebase_source_counts": {
                str(k): int(v) for k, v in expected_source_counts.items()
            },
            "max_outer_slew_violation": max(0.0, max_du_violation),
            "max_outer_amplitude_violation": max(0.0, max_amplitude_violation),
            "checks": checks,
        }
    )
    return result


def _worker_init(batch_root: str) -> None:
    global _WORKER_RUNNER, _WORKER_SUITE, _WORKER_DATA1
    global _WORKER_CONFIG, _WORKER_ENVELOPE, _WORKER_SEED_INDEX
    (
        _WORKER_RUNNER,
        _WORKER_SUITE,
        _WORKER_DATA1,
        _WORKER_CONFIG,
        _WORKER_ENVELOPE,
        _WORKER_SEED_INDEX,
        _,
    ) = _load_data(batch_root)


def _worker_score(values: tuple[float, float, float, float]) -> dict[str, Any]:
    if any(
        item is None
        for item in (
            _WORKER_RUNNER, _WORKER_SUITE, _WORKER_DATA1,
            _WORKER_CONFIG, _WORKER_ENVELOPE, _WORKER_SEED_INDEX,
        )
    ):
        raise RuntimeError("PID audit worker was not initialized")
    params = _WORKER_RUNNER.protocol.PidParams(*values)
    pass_scores = []
    for replay_pass in _WORKER_DATA1:
        seed = canonical_data1_seed(
            _WORKER_RUNNER, replay_pass.pass_id, _WORKER_SEED_INDEX
        )
        pass_scores.append(
            score_pid_pass(
                _WORKER_RUNNER,
                _WORKER_SUITE,
                replay_pass,
                _WORKER_CONFIG,
                _WORKER_ENVELOPE,
                params,
                seed,
            )["composite_normalized_RMS"]
        )
    row: dict[str, Any] = {
        "selection_protocol_id": PID_SELECTION_PROTOCOL_ID,
        "kp_scale": values[0],
        "ki_scale": values[1],
        "kd_scale": values[2],
        "move_limit_scale": values[3],
        "candidate_key": params.key,
        "data1_equal_pass_mean_RMS_c": float(np.mean(pass_scores)),
    }
    for replay_pass, value in zip(_WORKER_DATA1, pass_scores):
        row[f"pass_{replay_pass.pass_id}_RMS_c"] = float(value)
    return row


def _validate_existing_grid(grid: pd.DataFrame, candidates: list[tuple[float, ...]]) -> None:
    parameter_columns = (
        "kp_scale",
        "ki_scale",
        "kd_scale",
        "move_limit_scale",
    )
    required = {
        "selection_protocol_id",
        "candidate_key",
        "data1_equal_pass_mean_RMS_c",
        *parameter_columns,
        *[f"pass_{pass_id}_RMS_c" for pass_id in DATA1_SELECTION_PASS_ORDER],
    }
    missing = sorted(required - set(grid.columns))
    if missing:
        raise AssertionError(
            "Existing PID grid predates the canonical replay scorer; missing columns: "
            + ", ".join(missing)
        )
    protocol_ids = set(grid["selection_protocol_id"].astype(str))
    if protocol_ids != {PID_SELECTION_PROTOCOL_ID}:
        raise AssertionError(
            f"Existing PID grid protocol ids {sorted(protocol_ids)} do not match "
            f"{PID_SELECTION_PROTOCOL_ID!r}"
        )
    if len(grid) != len(candidates):
        raise AssertionError(
            f"Expected {len(candidates)} completed candidates, found {len(grid)}"
        )
    if grid["candidate_key"].astype(str).duplicated().any():
        duplicates = sorted(
            grid.loc[
                grid["candidate_key"].astype(str).duplicated(keep=False),
                "candidate_key",
            ].astype(str).unique()
        )
        raise AssertionError(
            "Existing PID grid contains duplicate candidate keys: "
            + ", ".join(duplicates[:10])
        )

    numeric = grid.loc[:, parameter_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise AssertionError("Existing PID grid contains non-finite candidate parameters")

    expected_tuples = {tuple(float(value) for value in candidate) for candidate in candidates}
    observed_tuples = {
        tuple(float(value) for value in row)
        for row in numeric.itertuples(index=False, name=None)
    }
    if observed_tuples != expected_tuples:
        missing_candidates = sorted(expected_tuples - observed_tuples)
        extra_candidates = sorted(observed_tuples - expected_tuples)
        raise AssertionError(
            "Existing PID grid candidate set differs from the declared grid; "
            f"missing={missing_candidates[:5]}, extra={extra_candidates[:5]}"
        )

    expected_keys = {
        f"kp{kp:g}_ki{ki:g}_kd{kd:g}_ml{move:g}"
        for kp, ki, kd, move in expected_tuples
    }
    observed_keys = set(grid["candidate_key"].astype(str))
    if observed_keys != expected_keys:
        raise AssertionError(
            "Existing PID grid key set differs from the declared candidate grid"
        )
    for key, values in zip(
        grid["candidate_key"].astype(str),
        numeric.itertuples(index=False, name=None),
    ):
        expected_key = (
            f"kp{values[0]:g}_ki{values[1]:g}_kd{values[2]:g}_ml{values[3]:g}"
        )
        if key != expected_key:
            raise AssertionError(
                f"Candidate key {key!r} does not encode its parameter row {expected_key!r}"
            )


def _outer_shell_audit(config) -> dict[str, Any]:
    base_move = [0.065, 0.052, 0.065]
    nominal = [value * float(config.outer_du_scale) for value in base_move]
    return {
        "candidate_move_limit_scope": "PID internal saturation bounds only",
        "amplitude_bound_each_channel": 0.55 * float(config.clamp_scale),
        "base_move_vector": base_move,
        "outer_du_scale": float(config.outer_du_scale),
        "nominal_move_vector": nominal,
        "legacy_move_limit_reference_dt_s": float(config.legacy_move_limit_reference_dt_s),
        "rowwise_move_limit_formula": "min(nominal_move*dt_control/0.50, nominal_move*2)",
        "outer_du_hard_scale": float(config.outer_du_hard_scale),
        "force_common_outer_limits": bool(config.force_common_outer_limits),
        "numerical_state_guardrail_scale": float(config.output_clip_scale),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    parser.add_argument(
        "--audit-existing-grid", action="store_true",
        help="rebuild only the audit JSON from a completed canonical-replay grid",
    )
    args = parser.parse_args()

    code = Path(__file__).resolve().parent
    package = code.parent
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    protocol = load_module("pid_audit_grid_protocol", code / "protocol.py")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates = [
        (p.kp_scale, p.ki_scale, p.kd_scale, p.move_limit_scale)
        for p in protocol.pid_candidate_grid()
    ]
    grid_path = out / "pid_full_grid.csv"
    if args.audit_existing_grid:
        if not grid_path.is_file():
            raise FileNotFoundError(grid_path)
        grid = pd.read_csv(grid_path)
        _validate_existing_grid(grid, candidates)
    else:
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            initializer=_worker_init,
            initargs=(str(args.batch_root.resolve()),),
        ) as pool:
            for index, row in enumerate(
                pool.map(_worker_score, candidates, chunksize=1), start=1
            ):
                rows.append(row)
                if index == 1 or index == len(candidates) or index % 50 == 0:
                    current = min(rows, key=lambda item: item["data1_equal_pass_mean_RMS_c"])
                    print(
                        f"[pid-grid] {index}/{len(candidates)} "
                        f"best={current['candidate_key']} "
                        f"RMS_c={current['data1_equal_pass_mean_RMS_c']:.6f}",
                        flush=True,
                    )
        grid = pd.DataFrame(rows).sort_values(
            ["data1_equal_pass_mean_RMS_c", "candidate_key"], kind="mergesort"
        )
        grid.to_csv(grid_path, index=False, encoding="utf-8-sig")

    best = grid.sort_values(
        ["data1_equal_pass_mean_RMS_c", "candidate_key"], kind="mergesort"
    ).iloc[0]
    observed = {
        key: float(best[key])
        for key in ["kp_scale", "ki_scale", "kd_scale", "move_limit_scale"]
    }
    (out / "pid_locked_params.json").write_text(
        json.dumps(
            observed
            | {
                "score_metric": COMPOSITE_FIELD,
                "score_RMS_c": float(best["data1_equal_pass_mean_RMS_c"]),
                "selection_protocol_id": PID_SELECTION_PROTOCOL_ID,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    runner, suite, data1, config, envelope, seed_index, manifest = _load_data(
        str(args.batch_root.resolve())
    )
    selected_params = runner.protocol.PidParams(**observed)
    selected_checks = []
    for replay_pass in data1:
        seed = canonical_data1_seed(runner, replay_pass.pass_id, seed_index)
        selected_checks.append(
            score_pid_pass(
                runner,
                suite,
                replay_pass,
                config,
                envelope,
                selected_params,
                seed,
                verify_replay_contract=True,
            )
        )
    selected_replay_mean = float(
        np.mean([item["composite_normalized_RMS"] for item in selected_checks])
    )
    selected_grid_score = float(best["data1_equal_pass_mean_RMS_c"])
    selected_score_delta = abs(selected_replay_mean - selected_grid_score)
    if selected_score_delta > 1e-12:
        raise AssertionError(
            "Selected grid score does not reproduce under the canonical replay: "
            f"grid={selected_grid_score}, replay={selected_replay_mean}"
        )

    frozen = {
        key: float(value) for key, value in manifest["locked_pid"].items()
    }
    matched = all(abs(observed[key] - frozen[key]) <= 1e-15 for key in frozen)
    audit = {
        "selection_protocol_id": PID_SELECTION_PROTOCOL_ID,
        "selection_design": (
            "complete 1050-candidate grid; equal arithmetic mean of 13 Data1 "
            "pass-level composite normalized RMS scores"
        ),
        "selection_is_leave_one_pass_out": False,
        "selection_metric": {
            "field": COMPOSITE_FIELD,
            "formula": (
                "mean(sqrt(mean((T/s_T)^2)), sqrt(mean((h/s_h)^2)), "
                "sqrt(mean((S/s_S)^2)))"
            ),
            "pass_aggregation": "equal arithmetic mean across the 13 Data1 passes",
        },
        "candidate_count": len(candidates),
        "data1_pass_count": len(data1),
        "data1_pass_order": list(DATA1_SELECTION_PASS_ORDER),
        "data_access_scope": {
            "included_batches": ["I", "IV", "VII"],
            "data2_files_opened_by_selection": False,
            "selection_loader_boundary": "batch_archive.load_batch_passes(included_batches=...)",
        },
        "workers": max(1, args.workers),
        "canonical_pid_config": asdict(config),
        "seed_rule": {
            "formula": "RNG_SEED + repeat + 2000*(Data1_order_index + 1)",
            "RNG_SEED": int(runner.RNG_SEED),
            "repeat": 0,
            "pass_seeds": {
                pass_id: canonical_data1_seed(runner, pass_id, seed_index)
                for pass_id in DATA1_SELECTION_PASS_ORDER
            },
        },
        "timebase": {
            "implementation": "segmented_archived_timebase.derive_segmented_archived_timebase",
            "innovation_boundaries": "innovation propagation is suppressed across segment boundaries",
            "distance_step_reference_m": float(config.distance_step_reference_m),
            "gap_ratio_low": float(config.gap_ratio_low),
            "gap_ratio_high": float(config.gap_ratio_high),
        },
        "outer_shell": _outer_shell_audit(config),
        "selected": observed,
        "selected_score_RMS_c": selected_grid_score,
        "selected_replayed_equal_pass_mean_RMS_c": selected_replay_mean,
        "selected_grid_vs_replay_score_abs_delta": selected_score_delta,
        "selected_replay_checks": selected_checks,
        "selected_max_composite_reconstruction_abs_delta": float(
            max(item["composite_reconstruction_abs_delta"] for item in selected_checks)
        ),
        "selected_all_replay_contract_checks_pass": bool(
            all(all(item["checks"].values()) for item in selected_checks)
        ),
        "frozen_manifest_lock": frozen,
        "selected_matches_frozen_lock": matched,
        **runner.protocol.scale_lock_manifest(),
    }
    (out / "pid_selection_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not matched:
        raise AssertionError(
            f"Recomputed Data1 PID selection {observed} does not match frozen lock {frozen}"
        )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
