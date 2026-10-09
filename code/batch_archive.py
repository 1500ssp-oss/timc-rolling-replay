from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raw_archive import resolve_archive


ENGINE_ROOT = Path(__file__).resolve().parent
FULL_PROJECT = ENGINE_ROOT / "engine_core"
CANONICAL_RUN = ENGINE_ROOT.parent / "config" / "reference_run"
CANONICAL_ANALYSIS = ENGINE_ROOT.parent / "results"
BATCH_ROOT = Path("production_batch_root")
DATA_MAP_PATH = ENGINE_ROOT.parent / "data_map" / "production_batch_map.csv"

BATCHES = ("I", "II", "III", "IV", "V", "VI", "VII")
DATA1_IDS = {f"P{i:02d}" for i in [1, 2, 3, 4, 12, 13, 14, 21, 22, 23, 24, 25, 26]}
CANONICAL_DATA2_IDS = {f"P{i:02d}" for i in [5, 6, 7, 8, 15, 16, 17, 18, 19, 20]}
REPEAT_EXTENSION_IDS = {"P09", "P10", "P11"}
TARGET_BATCHES = ["II", "III", "V", "VI"]

# This order reproduces the alphabetical ordering used by the locked Data2 run.
CANONICAL_SEED_INDEX = {
    "P16": 0,
    "P15": 1,
    "P20": 2,
    "P19": 3,
    "P18": 4,
    "P17": 5,
    "P08": 6,
    "P07": 7,
    "P06": 8,
    "P05": 9,
    "P09": 10,
    "P10": 11,
    "P11": 12,
}

POLICY_NAMES = {
    "C7-Core-g0.75-mpc0.14": "C7",
    "PID-g1.20-du0.60": "PID",
    "ADRC-g1.20-du0.95": "OC-PID",
    "C2-g1.20-du0.80": "C2",
}
PRIMARY_METRICS = [
    "composite_normalized_RMS",
    "S_out",
    "S_excess_mean",
    "S_excess_cvar95",
    "TV_L_per_100m",
    "amplitude_projection_ratio",
    "sat_ratio_logged",
    "integrated_severity_excess_m",
    "episodes_per_100m",
]
# Complete motion diagnostics used by the published expanded/repeat tables.
# Keep PRIMARY_METRICS unchanged because the fixed-dt comparison uses that list.
MOTION_METRICS = [
    "control_total_variation",
    "control_delta_rms",
    "delta_u_over_peak_action",
    "preclamp_total_variation",
    "postclamp_total_variation",
    "mpc_raw_total_variation",
    "mpc_component_total_variation",
    "TV_t_per_s",
    "TV_L_per_100m",
    "startup_motion",
    "restart_motion",
    "startup_restart_motion",
]
AGGREGATE_METRICS = list(dict.fromkeys(PRIMARY_METRICS + MOTION_METRICS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pass_id_from_name(name: str) -> str:
    token = Path(name).stem.split("_", 1)[0].upper()
    if len(token) != 3 or not token.startswith("P") or not token[1:].isdigit():
        raise ValueError(f"Cannot parse pass id from {name}")
    return token


def transition_from_name(name: str) -> str:
    token = Path(name).stem.split("_", 2)[1]
    if "to" not in token:
        raise ValueError(f"Cannot parse transition from {name}")
    entry, exit_ = token.split("to", 1)
    return f"{float(entry):.3f}->{float(exit_):.3f}"


def load_batch_passes(
    protocol,
    suite,
    replay,
    included_batches: set[str] | None = None,
) -> tuple[list[Any], pd.DataFrame]:
    """Load only the requested physical batch folders.

    ``None`` preserves the historical all-batch behavior.  Passing an explicit
    set is an I/O boundary, not a post-load filter; this is used by the Data1
    PID selection so no Data2 response file is opened during selection.
    """
    selected_batches = set(BATCHES) if included_batches is None else set(included_batches)
    unknown = selected_batches - set(BATCHES)
    if unknown:
        raise ValueError(f"Unknown batch ids: {sorted(unknown)}")
    if not selected_batches:
        raise ValueError("included_batches must contain at least one batch")
    archive_map, files_by_hash = resolve_archive(BATCH_ROOT, DATA_MAP_PATH)
    passes: list[Any] = []
    inventory: list[dict[str, Any]] = []
    for batch in BATCHES:
        if batch not in selected_batches:
            continue
        batch_rows = archive_map.loc[archive_map["batch"].eq(batch)].copy()
        if batch_rows.empty:
            raise AssertionError(f"Archive map contains no rows for batch {batch}")
        sources = set(batch_rows["source"].astype(str))
        if len(sources) != 1:
            raise AssertionError(f"Batch {batch} has inconsistent source labels: {sources}")
        source = next(iter(sources))
        staged: dict[str, dict[str, Any]] = {}
        with tempfile.TemporaryDirectory(prefix=f"rolling_batch_{batch}_") as temp_dir:
            temp_root = Path(temp_dir)
            for _, map_row in batch_rows.sort_values("pass_id").iterrows():
                pid = str(map_row["pass_id"])
                transition = str(map_row["transition"])
                file_path = files_by_hash[str(map_row["sha256"])]
                entry, exit_ = transition.split("->")
                canonical_name = f"{entry}_to_{exit_}__{pid}.csv"
                shutil.copy2(file_path, temp_root / canonical_name)
                staged[canonical_name] = {
                    "pass_id": pid,
                    "transition": transition,
                    "analysis_condition": str(map_row["analysis_condition"]),
                    "role": str(map_row["role"]),
                    "sha256": str(map_row["sha256"]),
                    "expected_rows": int(map_row["rows"]),
                }
            loaded = protocol.load_passes(suite, replay, temp_root, source)
        for rp in loaded:
            # The archived loader extracts the first two numbers in a filename
            # as thicknesses.  Pxx-prefixed archive names would therefore be
            # misread; the temporary canonical name prevents that parser leak.
            identity = staged[str(rp.p.pass_file)]
            pid = identity["pass_id"]
            transition = identity["transition"]
            analysis_condition = identity["analysis_condition"]
            role = identity["role"]
            file_alias = f"{pid}.csv"
            if len(rp.df) != identity["expected_rows"]:
                raise AssertionError(
                    f"{pid}: expected {identity['expected_rows']} rows, loaded {len(rp.df)}"
                )
            rp = replace(
                rp,
                source=source,
                pass_id=pid,
                condition_id=analysis_condition,
                trial_id=pid,
                file=file_alias,
            )
            passes.append(rp)
            inventory.append(
                {
                    "pass_id": pid,
                    "batch": batch,
                    "source": source,
                    "role": role,
                    "transition": transition,
                    "analysis_condition": analysis_condition,
                    "rows": len(rp.df),
                    "file": file_alias,
                    "sha256": identity["sha256"],
                }
            )
    inventory_frame = pd.DataFrame(inventory).sort_values("pass_id").reset_index(drop=True)
    observed_ids = set(inventory_frame.pass_id)
    if len(observed_ids) != len(inventory_frame):
        raise AssertionError("The selected batch archive contains duplicate pass ids.")
    if selected_batches == set(BATCHES):
        if observed_ids != {f"P{i:02d}" for i in range(1, 27)}:
            raise AssertionError("The batch archive is not an exact P01-P26 map.")
    elif selected_batches == {"I", "IV", "VII"}:
        if observed_ids != DATA1_IDS or len(inventory_frame) != 13:
            raise AssertionError(
                "The Data1-only archive is not the exact 13-pass I/IV/VII map."
            )
    return passes, inventory_frame


def add_episode_metrics(logs: pd.DataFrame, passes: list[Any], envelope: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scales = {rp.pass_id: float(rp.p.flatness_scale) for rp in passes}
    env = envelope.set_index("phase")
    events: list[dict[str, Any]] = []
    for (policy, pass_id, segment), group in logs.groupby(["policy", "pass_id", "segment_id"], sort=False):
        group = group.sort_values("k").reset_index(drop=True)
        scale = scales[str(pass_id)]
        upper = group.phase.map(env.S_norm_hi).fillna(float(env.S_norm_hi.median())).to_numpy(float)
        lower = group.phase.map(env.S_norm_lo).fillna(float(env.S_norm_lo.median())).to_numpy(float)
        z = group.S_control_error.to_numpy(float) / max(scale, 1e-12)
        excess = np.maximum.reduce([lower - z, z - upper, np.zeros(len(group))])
        mask = excess > 0.0
        start: int | None = None
        for i, flag in enumerate(np.r_[mask, False]):
            if flag and start is None:
                start = i
            if not flag and start is not None:
                part = group.iloc[start:i]
                recovery = np.nan
                inside = 0
                for q in range(i, len(group)):
                    inside = inside + 1 if not mask[q] else 0
                    if inside >= 2:
                        recovery = float(group.distance_step_m.iloc[i : q + 1].sum())
                        break
                events.append(
                    {
                        "policy": policy,
                        "pass_id": pass_id,
                        "segment_id": segment,
                        "start_k": int(group.k.iloc[start]),
                        "end_k": int(group.k.iloc[i - 1]),
                        "duration_s": float(part.timestamp_dt_s.sum()),
                        "distance_m": float(part.distance_step_m.sum()),
                        "peak_excess_norm": float(excess[start:i].max()),
                        "integrated_severity_excess_m": float(np.sum(excess[start:i] * part.distance_step_m.to_numpy(float))),
                        "recovery_distance_m": recovery,
                    }
                )
                start = None
    event_frame = pd.DataFrame(events)
    total_distance = (
        logs.groupby(["policy", "pass_id"], as_index=False).distance_step_m.sum().rename(columns={"distance_step_m": "total_distance_m"})
    )
    if event_frame.empty:
        per_pass = total_distance.assign(
            episode_count=0,
            peak_excess_norm=0.0,
            integrated_severity_excess_m=0.0,
            recovery_distance_m=np.nan,
        )
    else:
        per_pass = event_frame.groupby(["policy", "pass_id"], as_index=False).agg(
            episode_count=("start_k", "size"),
            peak_excess_norm=("peak_excess_norm", "max"),
            integrated_severity_excess_m=("integrated_severity_excess_m", "sum"),
            recovery_distance_m=("recovery_distance_m", "mean"),
        )
        per_pass = total_distance.merge(per_pass, on=["policy", "pass_id"], how="left")
        per_pass[["episode_count", "peak_excess_norm", "integrated_severity_excess_m"]] = per_pass[
            ["episode_count", "peak_excess_norm", "integrated_severity_excess_m"]
        ].fillna(0.0)
    per_pass["episodes_per_100m"] = 100.0 * per_pass.episode_count / per_pass.total_distance_m.clip(lower=1e-12)
    return event_frame, per_pass


def label_metrics(metrics: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    result["policy"] = result.control.map(POLICY_NAMES)
    if result.policy.isna().any():
        raise AssertionError(f"Unknown controls: {result.loc[result.policy.isna(), 'control'].unique()}")
    result = result.merge(inventory[["pass_id", "batch", "role", "transition"]], on="pass_id", how="left")
    if result.batch.isna().any():
        raise AssertionError("Some replay rows have no batch identity.")
    return result


def add_projection_metrics(metrics: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    raw_cols = ["u_target_unclipped_speed", "u_target_unclipped_gap", "u_target_unclipped_shape"]
    amp_cols = ["u_preclamp_speed", "u_preclamp_gap", "u_preclamp_shape"]
    work = logs.copy()
    raw = work[raw_cols].to_numpy(float)
    amplitude = work[amp_cols].to_numpy(float)
    work["amplitude_projected"] = np.any(np.abs(raw - amplitude) > 1e-12, axis=1)
    projection = work.groupby(["policy", "pass_id"], as_index=False).agg(
        amplitude_projection_ratio=("amplitude_projected", "mean"),
        slew_projection_ratio=("sat_flag", "mean"),
    )
    result = metrics.drop(columns=["amplitude_projection_ratio"], errors="ignore").merge(
        projection, on=["policy", "pass_id"], how="left"
    )
    result["sat_ratio_logged"] = result["slew_projection_ratio"]
    return result


def aggregate_batches(metrics: pd.DataFrame, batches: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    use = metrics[metrics.batch.isin(batches)].copy()
    available = [metric for metric in AGGREGATE_METRICS if metric in use]
    transition = use.groupby(["policy", "batch", "transition"], as_index=False)[available].mean()
    batch = transition.groupby(["policy", "batch"], as_index=False)[available].mean()
    overall = batch.groupby("policy", as_index=False)[available].mean()
    overall["n_batches"] = len(batches)
    overall["aggregation"] = "equal batch; equal transition within batch; repeated files first averaged within transition"
    return batch, overall


def pareto_set(frame: pd.DataFrame) -> str:
    objectives = frame[["composite_normalized_RMS", "S_excess_mean", "TV_L_per_100m"]].to_numpy(float)
    keep = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        dominated = np.all(objectives <= objectives[i], axis=1) & np.any(objectives < objectives[i], axis=1)
        if np.any(dominated):
            keep[i] = False
    return ";".join(sorted(frame.loc[keep, "policy"].astype(str)))


def leave_one_target_batch_out(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for omitted in TARGET_BATCHES:
        kept = [batch for batch in TARGET_BATCHES if batch != omitted]
        _, summary = aggregate_batches(metrics, kept)
        idx = summary.set_index("policy")
        pareto = pareto_set(summary)
        for baseline in ["PID", "OC-PID"]:
            for metric in ["S_excess_mean", "S_excess_cvar95", "composite_normalized_RMS", "TV_L_per_100m"]:
                base = float(idx.loc[baseline, metric])
                c7 = float(idx.loc["C7", metric])
                rows.append(
                    {
                        "omitted_batch": omitted,
                        "retained_batches": ";".join(kept),
                        "comparison": f"C7 vs {baseline}",
                        "metric": metric,
                        "c7": c7,
                        "baseline": base,
                        "relative_effect_pct": 100.0 * (c7 - base) / base if abs(base) > 1e-12 else np.nan,
                        "non_dominated_set": pareto,
                        "aggregation": "equal retained batch",
                    }
                )
    return pd.DataFrame(rows)


def repeat_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    use = metrics[metrics.pass_id.isin(REPEAT_EXTENSION_IDS)].copy()
    cols = [metric for metric in AGGREGATE_METRICS if metric in use]
    detailed = use[["pass_id", "transition", "policy", *cols]].sort_values(["pass_id", "policy"])
    average = use.groupby("policy", as_index=False)[cols].mean()
    average.insert(0, "pass_id", "equal-transition mean")
    average.insert(1, "transition", "P09-P11")
    return pd.concat([detailed, average], ignore_index=True, sort=False)


def canonical_equivalence(metrics: pd.DataFrame) -> pd.DataFrame:
    # The reference per-pass metrics of the frozen canonical run are shipped as
    # a publication-level result file; the raw canonical replay metrics are not
    # part of the reproducibility package.
    reference = pd.read_csv(CANONICAL_ANALYSIS / "canonical_nominal_pass_metrics.csv")
    current = metrics[metrics.pass_id.isin(CANONICAL_DATA2_IDS)].copy()
    keys = ["policy", "pass_id"]
    compare_metrics = [
        "composite_normalized_RMS",
        "S_out",
        "S_excess_mean",
        "S_excess_cvar95",
        "TV_L_per_100m",
        "duration_s",
        "distance_m",
    ]
    reference = reference.groupby(keys, as_index=False)[compare_metrics].mean()
    current = current.groupby(keys, as_index=False)[compare_metrics].mean()
    joined = reference[keys + compare_metrics].merge(
        current[keys + compare_metrics], on=keys, suffixes=("_reference", "_current"), how="outer", indicator=True
    )
    rows: list[dict[str, Any]] = []
    for metric in compare_metrics:
        delta = joined[f"{metric}_current"] - joined[f"{metric}_reference"]
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(delta.notna().sum()),
                "max_abs_delta": float(delta.abs().max()),
                "mean_abs_delta": float(delta.abs().mean()),
                "all_rows_matched": bool(joined._merge.eq("both").all()),
            }
        )
    return pd.DataFrame(rows)


def time_protocol_execution_qa(engine_root: Path, output_path: Path) -> pd.DataFrame:
    rows = [
        ("timebase reconstruction", "rowwise", "derive_archived_timebase reconstructs positive dt from chronology and distance/speed", "archived_update_timebase.py"),
        ("segment reset", "rowwise", "PID, command, estimator state and emulator state reset at unsupported gaps", "controller_replay.py"),
        ("PID integral", "rowwise", "PID.step receives dt_k and multiplies the trial integral by dt_k", "controllers.py"),
        ("PID derivative", "rowwise", "PID.step divides the error increment by dt_k", "controllers.py"),
        ("anti-windup", "rowwise", "vectorwise trial integral uses dt_k before saturation acceptance", "controllers.py"),
        ("amplitude projection", "not-time-dependent", "absolute normalized command bound; no elapsed-time factor is mathematically required", "controller_replay.py"),
        ("slew projection", "rowwise", "legacy per-update move bound is converted to a rate, then multiplied by dt_k with a hard cap", "controller_replay.py"),
        ("predictor horizon", "archived-update-domain", "10 future archived updates; not 5.0 s", "controller_replay.py"),
        ("LSTM predictor input", "rowwise-feature", "effective_dt_s and distance_step_m are explicit features, while the recurrent index remains archived-update-domain", "controller_replay.py"),
        ("EKF/state transition", "archived-update-domain", "the inherited filter advances once per archived row and has no continuous-time dt discretization", "15_realdata_closed_loop_suite.py"),
        ("response emulator", "archived-update-domain", "the bounded response family advances once per archived update; it is not a physical-time plant model", "controller_replay.py"),
        ("duration metrics", "rowwise", "duration is sum(dt_k)", "controller_replay.py"),
        ("TV per second", "rowwise", "post-clamp TV divided by sum(dt_k)", "controller_replay.py"),
        ("TV per 100 m", "distance-domain", "post-clamp TV divided by reconstructed traveled distance", "controller_replay.py"),
        ("episode duration", "rowwise", "episode duration is sum(dt_k) within the event", "run_appendix_audit_experiments.py"),
        ("episode severity", "distance-domain", "normalized excess is integrated against row-wise distance increments", "run_appendix_audit_experiments.py"),
        ("projection trigger ratios", "per-archived-update", "reported as fractions of archived updates, not fractions of physical time", "controller_replay.py"),
        ("fixed 0.50 s", "counterfactual-only", "retained only as the documented move-limit rate reference and an explicit sensitivity comparator", "controller lock and sensitivity run"),
    ]
    frame = pd.DataFrame(rows, columns=["component", "time_semantics", "implementation_evidence", "source"])
    frame["canonical_claim"] = frame.time_semantics.map(
        {
            "rowwise": "uses row-wise effective dt",
            "rowwise-feature": "uses row-wise time/distance as predictors",
            "archived-update-domain": "discrete archived-update mapping; no fixed sampling-period claim",
            "not-time-dependent": "time independent by definition",
            "distance-domain": "uses row-wise reconstructed distance",
            "per-archived-update": "per-update descriptive rate",
            "counterfactual-only": "not used in canonical replay",
        }
    )
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame


def run_policy_replay(runner, passes: list[Any], configs: list[Any], locked_pid: Any, envelope: pd.DataFrame, fixed_dt: float | None = None):
    original_timebase = runner.replay_timebase
    if fixed_dt is not None:
        def fixed_timebase(frame, config):
            _, _, distance, segment, _ = original_timebase(frame, config)
            dt = np.full(len(frame), float(fixed_dt), dtype=float)
            elapsed = np.cumsum(dt) - dt[0]
            source = np.full(len(frame), f"counterfactual_fixed_{fixed_dt:.2f}s", dtype=object)
            return elapsed, dt, distance, segment, source
        runner.replay_timebase = fixed_timebase
    metric_rows: list[dict[str, Any]] = []
    step_frames: list[pd.DataFrame] = []
    try:
        for cfg in configs:
            for rp in passes:
                seed_index = CANONICAL_SEED_INDEX[rp.pass_id]
                # Pass-only seed: all policies share the same random realization.
                seed = runner.RNG_SEED + 2000 * (seed_index + 1)
                curve, diag = runner.run_exposure_control(runner_suite, rp, cfg, seed, locked_pid, 0)
                curve["pass_id"] = rp.pass_id
                curve["control"] = cfg.label
                curve["policy"] = POLICY_NAMES[cfg.label]
                metric_rows.append(runner.suite_metrics_with_config(runner_suite, rp, curve, diag, cfg, envelope))
                step_frames.append(curve)
                print(f"[replay] {'fixed' if fixed_dt is not None else 'rowwise'} {cfg.label} {rp.pass_id}", flush=True)
    finally:
        runner.replay_timebase = original_timebase
    return pd.DataFrame(metric_rows), pd.concat(step_frames, ignore_index=True, sort=False)


runner_suite = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--skip-fixed-dt", action="store_true")
    parser.add_argument("--batch-root", type=Path, default=None,
                        help="Folder containing the seven production-batch directories")
    args = parser.parse_args()
    if args.batch_root is not None:
        global BATCH_ROOT
        BATCH_ROOT = args.batch_root.resolve()
    out = Path(args.out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if str(ENGINE_ROOT) not in sys.path:
        sys.path.insert(0, str(ENGINE_ROOT))
    runner = load_module("rolling_batch_runner", ENGINE_ROOT / "controller_replay.py")
    runner.protocol.PROJECT_DIR = FULL_PROJECT
    global runner_suite
    runner_suite, replay = runner.protocol.load_suite_modules()

    all_passes, inventory = load_batch_passes(runner.protocol, runner_suite, replay)
    inventory.to_csv(out / "production_batch_map.csv", index=False, encoding="utf-8-sig")
    target_passes = [rp for rp in all_passes if rp.pass_id in CANONICAL_DATA2_IDS | REPEAT_EXTENSION_IDS]

    manifest_path = CANONICAL_RUN / "03_nominal" / "stage_allocation_ablation_manifest.json"
    canonical_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configs = [runner.ExposureConfig(**payload) for payload in canonical_manifest["configs"]]
    locked_pid = runner.protocol.PidParams(**canonical_manifest["locked_pid"])
    envelope_path = CANONICAL_RUN / "03_nominal" / "audit" / "data1_stage_quantile_envelope_p025_p975.csv"
    envelope = pd.read_csv(envelope_path)

    metrics, logs = run_policy_replay(runner, target_passes, configs, locked_pid, envelope)
    metrics = label_metrics(metrics, inventory)
    metrics = add_projection_metrics(metrics, logs)
    events, episode = add_episode_metrics(logs, target_passes, envelope)
    metrics = metrics.merge(episode[["policy", "pass_id", "integrated_severity_excess_m", "episodes_per_100m"]], on=["policy", "pass_id"], how="left")
    metrics.to_csv(out / "expanded_13pass_fixed_policy_metrics.csv", index=False, encoding="utf-8-sig")
    logs.to_csv(out / "expanded_13pass_fixed_policy_step_logs.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    events.to_csv(out / "expanded_13pass_episode_events.csv", index=False, encoding="utf-8-sig")
    episode.to_csv(out / "expanded_13pass_episode_metrics.csv", index=False, encoding="utf-8-sig")

    equivalence = canonical_equivalence(metrics)
    equivalence.to_csv(out / "canonical_10pass_replay_equivalence.csv", index=False, encoding="utf-8-sig")
    repeat_summary(metrics).to_csv(out / "repeat_sequence_extension_metrics.csv", index=False, encoding="utf-8-sig")
    batch_summary, expanded_summary = aggregate_batches(metrics, TARGET_BATCHES)
    batch_summary.to_csv(out / "expanded_26pass_batch_level_policy_summary.csv", index=False, encoding="utf-8-sig")
    expanded_summary.to_csv(out / "expanded_26pass_batch_weighted_policy_summary.csv", index=False, encoding="utf-8-sig")
    leave_one_target_batch_out(metrics).to_csv(out / "data2_leave_one_batch_out_policy_summary.csv", index=False, encoding="utf-8-sig")

    time_protocol_execution_qa(ENGINE_ROOT, out / "time_protocol_execution_QA.csv")
    if not args.skip_fixed_dt:
        canonical_passes = [rp for rp in target_passes if rp.pass_id in CANONICAL_DATA2_IDS]
        fixed_metrics, fixed_logs = run_policy_replay(runner, canonical_passes, configs, locked_pid, envelope, fixed_dt=0.50)
        fixed_metrics = label_metrics(fixed_metrics, inventory)
        fixed_metrics = add_projection_metrics(fixed_metrics, fixed_logs)
        fixed_events, fixed_episode = add_episode_metrics(fixed_logs, canonical_passes, envelope)
        fixed_metrics = fixed_metrics.merge(
            fixed_episode[["policy", "pass_id", "integrated_severity_excess_m", "episodes_per_100m"]],
            on=["policy", "pass_id"],
            how="left",
        )
        fixed_metrics.to_csv(out / "counterfactual_fixed_0p50s_10pass_metrics.csv", index=False, encoding="utf-8-sig")
        fixed_events.to_csv(out / "counterfactual_fixed_0p50s_episode_events.csv", index=False, encoding="utf-8-sig")
        row_summary = metrics[metrics.pass_id.isin(CANONICAL_DATA2_IDS)].groupby("policy", as_index=False)[PRIMARY_METRICS].mean()
        fixed_summary = fixed_metrics.groupby("policy", as_index=False)[PRIMARY_METRICS].mean()
        comparison = row_summary.merge(fixed_summary, on="policy", suffixes=("_rowwise", "_fixed0p50"))
        for metric in PRIMARY_METRICS:
            comparison[f"{metric}_relative_delta_pct"] = 100.0 * (
                comparison[f"{metric}_fixed0p50"] - comparison[f"{metric}_rowwise"]
            ) / comparison[f"{metric}_rowwise"].replace(0.0, np.nan)
        comparison.to_csv(out / "fixed_dt_vs_rowwise_dt_sensitivity.csv", index=False, encoding="utf-8-sig")

    run_manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "canonical_role": "23-pass main analysis retained; P09-P11 are fixed-policy repeat extension",
        "batch_root": str(BATCH_ROOT),
        "engine_root": str(ENGINE_ROOT),
        "full_project": str(FULL_PROJECT),
        "canonical_run": str(CANONICAL_RUN),
        "canonical_manifest_sha256": sha256(manifest_path),
        "envelope_sha256": sha256(envelope_path),
        "controller_configs": [asdict(config) for config in configs],
        "locked_pid": asdict(locked_pid),
        "target_batches": TARGET_BATCHES,
        "aggregation": "repeat within transition, equal transition within batch, equal target batch",
        "fixed_dt_status": "explicit counterfactual only" if not args.skip_fixed_dt else "not run",
    }
    (out / "batch_audit_run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "rows": len(metrics), "equivalence": equivalence.to_dict("records")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
