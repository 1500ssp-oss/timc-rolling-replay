"""Diagnose how canonical replay flatness distributions differ from records.

This is a descriptive, row-weighted diagnostic.  It does not validate causal
effects or the physical plant response.  Data1 fixes the normalization and the
three maturity bands; Data2 records and the four canonical replay policies are
then compared on exactly the same archived row keys.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
EXPECTED_DATA2_ROWS = 5513
POLICIES = ("PID", "OC-PID", "C2", "C7-Core")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def policy_name(row: pd.Series) -> str:
    base = str(row.get("base_control", ""))
    label = str(row.get("control", ""))
    if base == "C0" or label.startswith("PID"):
        return "PID"
    if base == "ADRC" or label.startswith("ADRC") or label.startswith("OC-PID"):
        return "OC-PID"
    if base == "C2" or label.startswith("C2"):
        return "C2"
    if base == "C7" or label.startswith("C7"):
        return "C7-Core"
    return label


def describe(values: pd.Series, center: float, q80: float, q95: float) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if x.empty:
        raise ValueError("Cannot summarize an empty flatness distribution")
    d = (x - center).abs()
    return {
        "n_rows": int(len(x)),
        "mean_norm": float(x.mean()),
        "median_norm": float(x.median()),
        "signed_mean_bias_from_zero": float(x.mean()),
        "signed_median_bias_from_zero": float(x.median()),
        "mean_shift_from_data1_median": float(x.mean() - center),
        "median_shift_from_data1_median": float(x.median() - center),
        "IQR": float(x.quantile(0.75) - x.quantile(0.25)),
        "MAD": float((x - x.median()).abs().median()),
        "P5": float(x.quantile(0.05)),
        "P95": float(x.quantile(0.95)),
        "central_share_locked": float((d <= q80).mean()),
        "moderate_share_locked": float(((d > q80) & (d <= q95)).mean()),
        "extreme_share_locked": float((d > q95).mean()),
    }


def assert_close(observed: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(float(observed), float(expected), rtol=0.0, atol=atol):
        raise AssertionError(f"{label}: observed {observed!r}, expected {expected!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path, help="Canonical run root")
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--extension-script", type=Path, default=CODE / "batch_archive.py")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    run = args.run.resolve()
    nominal = run / "03_nominal"
    logs_path = nominal / "stage_allocation_ablation_step_logs.csv.gz"
    if not logs_path.exists():
        raise FileNotFoundError(logs_path)

    protocol = load_module("replay_distribution_protocol", CODE / "protocol.py")
    suite, replay = protocol.load_suite_modules()
    extension = load_module("replay_distribution_batch", args.extension_script.resolve())
    extension.BATCH_ROOT = args.batch_root.resolve()
    all_passes, _ = extension.load_batch_passes(protocol, suite, replay)
    data1 = [rp for rp in all_passes if rp.pass_id in extension.DATA1_IDS]
    data2 = [rp for rp in all_passes if rp.pass_id in extension.CANONICAL_DATA2_IDS]
    if len(data1) != 13 or len(data2) != 10:
        raise AssertionError(f"Expected 13 Data1 and 10 canonical Data2 passes, got {len(data1)} and {len(data2)}")

    data1_values = np.concatenate(
        [np.asarray(rp.y_real[:, 2], float) / float(protocol.scale_vec(rp.p)[2]) for rp in data1]
    )
    if not np.all(np.isfinite(data1_values)):
        raise AssertionError("Data1 normalized flatness values are not finite")
    center = float(pd.Series(data1_values).median())
    absdev = pd.Series(np.abs(data1_values - center))
    q80 = float(absdev.quantile(0.80))
    q95 = float(absdev.quantile(0.95))

    pass_meta: dict[str, dict[str, Any]] = {}
    recorded_rows: list[dict[str, Any]] = []
    expected_keys: set[tuple[str, int, int]] = set()
    for rp in data2:
        scale = float(protocol.scale_vec(rp.p)[2])
        timebase = protocol.derive_segmented_archived_timebase(
            rp.df, protocol.DISTANCE_STEP_REFERENCE_M
        )
        segments = np.asarray(timebase.segment_id, dtype=int)
        if len(segments) != len(rp.y_real):
            raise AssertionError(f"{rp.pass_id}: timebase/measurement row mismatch")
        pass_meta[rp.pass_id] = {"condition_id": rp.condition_id, "scale": scale}
        for k, (segment, value) in enumerate(zip(segments, rp.y_real[:, 2], strict=True)):
            key = (rp.pass_id, int(segment), int(k))
            expected_keys.add(key)
            recorded_rows.append(
                {
                    "series": "Recorded Data2",
                    "policy": "Recorded Data2",
                    "pass_id": rp.pass_id,
                    "condition_id": rp.condition_id,
                    "segment_id": int(segment),
                    "k": int(k),
                    "value_norm": float(value) / scale,
                }
            )
    if len(expected_keys) != EXPECTED_DATA2_ROWS:
        raise AssertionError(
            f"Canonical Data2 must contain {EXPECTED_DATA2_ROWS} unique row keys, found {len(expected_keys)}"
        )

    logs = pd.read_csv(logs_path, compression="gzip")
    required = {
        "block", "base_control", "control", "pass_id", "condition_id", "segment_id",
        "k", "repeat_idx", "S_control_error",
    }
    missing = required - set(logs.columns)
    if missing:
        raise AssertionError(f"Canonical step log is missing columns: {sorted(missing)}")
    logs = logs.loc[logs["block"].eq("data1_controller_search")].copy()
    if set(pd.to_numeric(logs.repeat_idx, errors="raise").astype(int)) != {0}:
        raise AssertionError("Canonical distribution diagnostic requires the single repeat_idx=0 run")
    logs["policy"] = logs.apply(policy_name, axis=1)
    observed_policies = set(logs.policy)
    if observed_policies != set(POLICIES):
        raise AssertionError(f"Expected canonical policies {POLICIES}, found {sorted(observed_policies)}")

    replay_rows: list[dict[str, Any]] = []
    key_columns = ["pass_id", "segment_id", "k"]
    for policy in POLICIES:
        frame = logs.loc[logs.policy.eq(policy)].copy()
        if frame.duplicated(key_columns).any():
            raise AssertionError(f"{policy}: duplicate canonical row keys")
        keys = {
            (str(row.pass_id), int(row.segment_id), int(row.k))
            for row in frame[key_columns].itertuples(index=False)
        }
        if keys != expected_keys:
            missing_keys = len(expected_keys - keys)
            extra_keys = len(keys - expected_keys)
            raise AssertionError(f"{policy}: row-key mismatch (missing={missing_keys}, extra={extra_keys})")
        values = pd.to_numeric(frame.S_control_error, errors="coerce")
        if not np.all(np.isfinite(values.to_numpy(float))):
            raise AssertionError(f"{policy}: non-finite flatness replay value")
        for row, value in zip(frame.itertuples(index=False), values, strict=True):
            pid = str(row.pass_id)
            scale = float(pass_meta[pid]["scale"])
            replay_rows.append(
                {
                    "series": policy,
                    "policy": policy,
                    "pass_id": pid,
                    "condition_id": str(pass_meta[pid]["condition_id"]),
                    "segment_id": int(row.segment_id),
                    "k": int(row.k),
                    "value_norm": float(value) / scale,
                }
            )

    combined = pd.DataFrame(recorded_rows + replay_rows)
    recorded = combined.loc[combined.series.eq("Recorded Data2"), "value_norm"]
    recorded_stats = describe(recorded, center, q80, q95)
    summary_rows: list[dict[str, Any]] = []
    for series, group in combined.groupby("series", sort=False):
        stats = describe(group.value_norm, center, q80, q95)
        stats.update(
            {
                "series": series,
                "row_weighting": "each archived row has equal weight",
                "data1_median_norm": center,
                "data1_absdev_p80": q80,
                "data1_absdev_p95": q95,
                "mean_shift_vs_recorded_data2": float(stats["mean_norm"] - recorded_stats["mean_norm"]),
                "median_shift_vs_recorded_data2": float(stats["median_norm"] - recorded_stats["median_norm"]),
            }
        )
        summary_rows.append(stats)
    summary = pd.DataFrame(summary_rows)
    summary = summary[[
        "series", "n_rows", "mean_norm", "median_norm",
        "signed_mean_bias_from_zero", "signed_median_bias_from_zero",
        "mean_shift_vs_recorded_data2", "median_shift_vs_recorded_data2",
        "mean_shift_from_data1_median", "median_shift_from_data1_median",
        "IQR", "MAD", "P5", "P95", "central_share_locked",
        "moderate_share_locked", "extreme_share_locked", "data1_median_norm",
        "data1_absdev_p80", "data1_absdev_p95", "row_weighting",
    ]]

    strata_rows: list[dict[str, Any]] = []
    groupings = {
        "pass": ["pass_id"],
        "condition": ["condition_id"],
        "segment": ["pass_id", "segment_id"],
    }
    for series, series_frame in combined.groupby("series", sort=False):
        for level, columns in groupings.items():
            for group_key, group in series_frame.groupby(columns, sort=False, dropna=False):
                keys = group_key if isinstance(group_key, tuple) else (group_key,)
                stats = describe(group.value_norm, center, q80, q95)
                row: dict[str, Any] = {
                    "series": series,
                    "stratum": level,
                    "pass_id": "",
                    "condition_id": "",
                    "segment_id": "",
                }
                row.update(dict(zip(columns, keys, strict=True)))
                row.update(stats)
                strata_rows.append(row)
    stratified = pd.DataFrame(strata_rows)

    maturity_path = run / "10_appendix_audit" / "10_maturity_headroom_summary.csv"
    maturity_reproduced = False
    if maturity_path.exists():
        maturity = pd.read_csv(maturity_path)
        row = maturity.loc[maturity.dataset.eq("Data2") & maturity.target.eq("S")]
        if len(row) != 1:
            raise AssertionError("Appendix maturity table has no unique Data2/S row")
        ref = row.iloc[0]
        assert int(ref.n_rows) == int(recorded_stats["n_rows"])
        for ours, theirs, label in (
            (recorded_stats["median_norm"], ref.median_norm, "Data2/S median"),
            (recorded_stats["IQR"], ref.IQR, "Data2/S IQR"),
            (recorded_stats["MAD"], ref.MAD, "Data2/S MAD"),
            (recorded_stats["central_share_locked"], ref.central_share_locked, "Data2/S central share"),
            (recorded_stats["moderate_share_locked"], ref.near_limit_share_locked, "Data2/S moderate share"),
            (recorded_stats["extreme_share_locked"], ref.extreme_share_locked, "Data2/S extreme share"),
        ):
            assert_close(ours, theirs, label)
        maturity_reproduced = True

    threshold_path = run / "10_appendix_audit" / "10_maturity_thresholds_locked_data1.json"
    threshold_reproduced = False
    if threshold_path.exists():
        published = json.loads(threshold_path.read_text(encoding="utf-8"))["S"]
        assert_close(center, published["data1_median"], "Data1/S median")
        assert_close(q80, published["central_absdev_p80"], "Data1/S p80")
        assert_close(q95, published["near_absdev_p95"], "Data1/S p95")
        threshold_reproduced = True

    manifest = json.loads((nominal / "stage_allocation_ablation_manifest.json").read_text(encoding="utf-8"))
    canonical_configs = [item for item in manifest.get("configs", []) if item.get("block") == "data1_controller_search"]
    residual_gains = [float(item.get("residual_gain", 0.82)) for item in canonical_configs]
    if len(canonical_configs) != 4 or any(not np.isclose(v, 0.82, rtol=0.0, atol=1e-15) for v in residual_gains):
        raise AssertionError(f"Canonical configs do not all use residual_gain=0.82: {residual_gains}")

    for row in summary.itertuples(index=False):
        share_sum = row.central_share_locked + row.moderate_share_locked + row.extreme_share_locked
        assert_close(share_sum, 1.0, f"{row.series} maturity shares", atol=2e-15)

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "replay_flatness_distribution_summary.csv", index=False, encoding="utf-8-sig")
    stratified.to_csv(out / "replay_flatness_distribution_stratified.csv", index=False, encoding="utf-8-sig")
    audit = {
        "diagnostic_scope": "descriptive row-weighted flatness distribution comparison; not physical-model or causal validation",
        "canonical_step_log": str(logs_path.relative_to(ROOT)).replace("\\", "/"),
        "expected_data2_row_keys": EXPECTED_DATA2_ROWS,
        "observed_data2_row_keys": len(expected_keys),
        "policies": list(POLICIES),
        "same_row_keys_for_every_policy": True,
        "canonical_residual_gain": 0.82,
        "canonical_innovations_and_commands_retained": True,
        "maturity_thresholds_recomputed_from_data1_only": True,
        "scale_lock": protocol.scale_lock_manifest(),
        "published_maturity_summary_reproduced": maturity_reproduced,
        "published_maturity_thresholds_reproduced": threshold_reproduced,
        "all_checks_passed": True,
    }
    (out / "replay_flatness_distribution_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"[done] replay flatness distribution diagnostic: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
