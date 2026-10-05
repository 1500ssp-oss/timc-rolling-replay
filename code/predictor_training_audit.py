"""Audit published aggregates, or regenerate one-SE tables from real fold rows.

The default audit does not reconstruct missing historical scores, infer the
released model's update budget, train a predictor, or alter frozen weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
COMPLEXITY = {"Persistence": 0, "LinearARX": 1, "GRU": 2, "LSTM": 3, "Transformer": 4}


def summarize_fold_scores(scores: pd.DataFrame, required_folds: int = 13) -> pd.DataFrame:
    required = {"target", "model", "fold", "seed", "MASE"}
    if not required.issubset(scores.columns):
        raise ValueError(f"missing fold-score columns: {sorted(required-set(scores.columns))}")
    if scores.duplicated(["target", "model", "fold", "seed"]).any():
        raise ValueError("duplicate target/model/fold/seed score rows")
    if not np.isfinite(pd.to_numeric(scores["MASE"], errors="coerce")).all():
        raise ValueError("MASE scores must be finite")
    unknown = set(scores["model"]) - set(COMPLEXITY)
    if unknown:
        raise ValueError(f"unsupported model complexity ordering: {sorted(unknown)}")
    rows = []
    for (target, model), group in scores.groupby(["target", "model"]):
        folds = group.groupby("fold")["MASE"].agg(["mean", "std", "count"])
        if len(folds) != required_folds:
            raise ValueError(f"{target}/{model}: expected {required_folds} folds, found {len(folds)}")
        expected_seeds = {0} if model in {"Persistence", "LinearARX"} else {101, 202, 303}
        for fold, sub in group.groupby("fold"):
            if set(sub["seed"].astype(int)) != expected_seeds:
                raise ValueError(f"{target}/{model}/fold {fold}: expected seeds {sorted(expected_seeds)}")
        if model == "Persistence" and not np.allclose(group["MASE"], 1.0, atol=1e-10, rtol=0):
            raise ValueError("persistence MASE is not 1; check denominator or degenerate target")
        sd = float(folds["mean"].std(ddof=1))
        rows.append({"target": target, "model": model, "mean_MASE": float(folds["mean"].mean()),
                     "fold_SD": sd, "n_folds": len(folds),
                     "mean_within_fold_seed_SD": float(folds["std"].mean()) if model not in {"Persistence", "LinearARX"} else np.nan,
                     "fold_SE": sd/np.sqrt(len(folds))})
    result = pd.DataFrame(rows)
    for target, group in result.groupby("target"):
        if set(group["model"]) != set(COMPLEXITY):
            raise ValueError(f"{target}: all five declared models are required")
        fold_sets = [set(scores.loc[scores.target.eq(target)&scores.model.eq(model), "fold"]) for model in COMPLEXITY]
        if any(folds != fold_sets[0] for folds in fold_sets[1:]):
            raise ValueError(f"{target}: models do not share the same held-out folds")
        best = group.loc[group["mean_MASE"].idxmin()]
        threshold = float(best["mean_MASE"] + best["fold_SE"])
        eligible = group.loc[group["mean_MASE"] <= threshold + 1e-12]
        selected = min(eligible["model"], key=COMPLEXITY.get)
        mask = result.target.eq(target)
        result.loc[mask, "eligible_one_se"] = result.loc[mask, "mean_MASE"] <= threshold + 1e-12
        result.loc[mask, "selected"] = result.loc[mask, "model"].eq(selected)
        result.loc[mask, "one_se_threshold"] = threshold
        result.loc[mask, "selection_note"] = ""
        result.loc[mask & result.model.eq(selected), "selection_note"] = (
            "Seeds averaged within each fold; equal-fold mean and fold SE; lowest-complexity eligible model "
            "under Persistence < LinearARX < GRU < LSTM < Transformer.")
    return result.sort_values(["target", "mean_MASE", "model"]).reset_index(drop=True)


def selection_records(summary: pd.DataFrame) -> list[dict]:
    records = []
    for target, group in summary.groupby("target"):
        selected = group.loc[group.selected.astype(bool)].iloc[0]
        best = group.loc[group.mean_MASE.idxmin()]
        records.append({"target": target, "selected_model": selected["model"],
                        "mean_MASE": float(selected.mean_MASE), "one_se_threshold": float(best.mean_MASE+best.fold_SE),
                        "best_model": best["model"], "note": "Generated from actual fold/seed rows; see fold score input provenance."})
    return records


def audit_published(root: Path) -> dict:
    summary_path = root / "results" / "predictor_lopo_one_se_summary.csv"
    summary = pd.read_csv(summary_path)
    config = json.loads((root / "config" / "model_selection.json").read_text(encoding="utf-8"))
    checks = []
    def check(name, valid):
        checks.append({"name": name, "passed": bool(valid)})
    check("15 distinct published target/model aggregate rows", len(summary)==15 and not summary.duplicated(["target","model"]).any())
    check("all aggregates declare 13 LOPO passes", summary.n_folds.eq(13).all())
    check("fold_SE = fold_SD / sqrt(n_folds)", np.allclose(summary.fold_SE, summary.fold_SD/np.sqrt(summary.n_folds), rtol=0, atol=1e-12))
    for target, group in summary.groupby("target"):
        best = group.loc[group.mean_MASE.idxmin()]
        threshold = float(best.mean_MASE+best.fold_SE)
        eligible = group.mean_MASE <= threshold + 1e-12
        selected_model = min(group.loc[eligible,"model"], key=COMPLEXITY.get)
        check(f"{target}: one-SE threshold from aggregate mean plus SE", np.allclose(group.one_se_threshold, threshold, rtol=0,atol=1e-12))
        check(f"{target}: eligibility flags", np.array_equal(group.eligible_one_se.to_numpy(bool),eligible.to_numpy()))
        selected = group.loc[group.selected.astype(bool)]
        check(f"{target}: one lowest-complexity selection", len(selected)==1 and selected.iloc[0]["model"]==selected_model)
        cfg = [item for item in config if item["target"]==target]
        check(f"{target}: config selection and threshold", len(cfg)==1 and cfg[0]["selected_model"]==selected_model and abs(cfg[0]["one_se_threshold"]-threshold)<=1e-12)
    checkpoint_path = root/"models"/"final_thickness_model.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    check("released checkpoint kind/target/window/features", payload.get("kind")=="GRU" and payload.get("target")=="exit_thickness_dev" and payload.get("window")==8 and len(payload.get("features",[]))==10)
    return {"status":"aggregate_consistent_but_historical_training_provenance_incomplete" if all(item["passed"] for item in checks) else "aggregate_inconsistency",
            "checks":checks, "source_summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "frozen_checkpoint_sha256":hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "checkpoint_fields_present":sorted(payload.keys()),
            "historical_fold_seed_scores_available":False, "historical_training_curves_available":False,
            "actual_frozen_training_updates":None, "actual_frozen_training_seed":None,
            "actual_frozen_training_device":None, "actual_frozen_training_optimizer":None,
            "historical_search_scope":"A bounded search of the available project records did not recover authentic historical per-fold/seed scores or training-run logs.",
            "limitations":["Aggregate agreement cannot reconstruct the missing per-fold/seed predictions or prove training adequacy.",
                           "The legacy script's --epochs counted one random minibatch update per iteration; its defaults are not evidence of the actual frozen checkpoint's budget.",
                           "The frozen predictor is retained unchanged; new training instrumentation applies to future executions only.",
                           "The master replay pipeline does not regenerate predictor fitting or the historical model-selection summary."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=ROOT)
    parser.add_argument("--fold-scores", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--required-folds", type=int, default=13)
    args = parser.parse_args()
    if args.fold_scores:
        if args.out_root is None:
            parser.error("--out-root is required when generating from --fold-scores")
        source = pd.read_csv(args.fold_scores)
        summary = summarize_fold_scores(source, args.required_folds)
        args.out_root.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out_root/"predictor_lopo_one_se_summary.csv", index=False, encoding="utf-8-sig")
        (args.out_root/"model_selection.json").write_text(json.dumps(selection_records(summary),indent=2),encoding="utf-8")
        receipt = {"source_fold_scores_sha256":hashlib.sha256(args.fold_scores.read_bytes()).hexdigest(),
                   "source_rows":len(source),"aggregation":"mean seeds within each held-out pass; equal-pass mean, ddof=1 fold SD and SD/sqrt(n) SE",
                   "complexity_order":COMPLEXITY,"training_provenance":"Refer to the matching model_run and checkpoint metadata; summary regeneration alone does not authenticate training."}
        (args.out_root/"predictor_summary_generation.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
        print(json.dumps(receipt,indent=2))
        return 0
    report = audit_published(args.package_root)
    out = args.out_root or args.package_root/"results"
    out.mkdir(parents=True, exist_ok=True)
    (out/"predictor_training_provenance_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return 0 if all(item["passed"] for item in report["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
