"""Independent offline recalculation from distributed publication tables.

No replay, publication, metric or bootstrap helpers are imported. The only
production functions exercised are AST-isolated response/solver definitions
in the numerical closed-form equivalence test. This audit does not reconstruct
proprietary raw records or verify plant-side effectiveness.
"""
from __future__ import annotations

import argparse
import ast
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["composite_normalized_RMS", "TV_L_per_100m", "S_excess_mean", "S_excess_cvar95"]
BASELINES = ["PID", "OC-PID", "C2"]
SEED = 20260712
DRAWS = 10_000
ATOL = 1e-10


def independent_upper_tail_count(n: int, quantile: float = 0.95) -> int:
    """Independent rational count; do not import the production tail helper."""
    q = Fraction(str(quantile))
    if not 0 <= q < 1 or n < 0 or int(n) != n:
        raise ValueError("tail count requires integer n >= 0 and 0 <= q < 1")
    ratio = (1 - q) * int(n)
    whole, remainder = divmod(ratio.numerator, ratio.denominator)
    return whole + int(remainder != 0)


def tail_cardinality_audit(results: Path) -> dict:
    """Check every public canonical pass tail against exact rational counts."""
    table = pd.read_csv(results / "recorded_band_decomposition_by_file.csv")
    expected = table.n_rows.map(independent_upper_tail_count).to_numpy(int)
    observed = table.tail_rows.to_numpy(float)
    difference = observed - expected
    mismatches = table.loc[difference != 0, ["policy", "pass_id", "n_rows", "tail_rows"]]
    return {"passed":bool(np.isfinite(observed).all() and (difference == 0).all()),
            "quantile":0.95,"valid_quantile_range":"0 <= q < 1",
            "tail_count_rule":"ceil((1-q)*N), q interpreted as its decimal value; no fractional boundary weighting",
            "pass_policy_rows":int(len(table)),
            "max_count_difference":float(np.max(np.abs(difference))) if len(table) else 0.0,
            "mismatches":mismatches.to_dict("records"),
            "scope":"Independent Fraction-based count check for distributed canonical pass-level band-tail records; no private trajectories required."}


def normalize_policy(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "policy" in frame:
        frame["policy"] = frame.policy.replace({"C7":"C7-Core", "ADRC":"OC-PID"})
    return frame


def equal_condition(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    files = frame.groupby(["policy", "condition_id", "file"])[cols].mean()
    return files.groupby(["policy", "condition_id"]).mean().groupby("policy").mean()


def comparison_row(frame: pd.DataFrame, metric: str, baseline: str) -> pd.Series:
    labels = frame.comparison.astype(str).replace({f"C7-Core vs {baseline}":f"C7 vs {baseline}"})
    rows = frame.loc[frame.metric.eq(metric) & labels.eq(f"C7 vs {baseline}")]
    if len(rows) != 1:
        raise ValueError(f"expected one publication row for {metric}/C7 vs {baseline}, found {len(rows)}")
    return rows.iloc[0]


def bootstrap_audit(metrics: pd.DataFrame, results: Path) -> dict:
    published = pd.read_csv(results/"paired_effect_bootstrap.csv")
    clustered = pd.read_csv(results/"batch_clustered_bootstrap.csv")
    if not published.bootstrap_draws.eq(DRAWS).all() or not clustered.bootstrap_draws.eq(DRAWS).all():
        raise ValueError(f"this audit implements the declared {DRAWS}-draw protocol")
    condition_rng = np.random.default_rng(SEED)
    batch_rng = np.random.default_rng(SEED)
    rows = []
    errors = []
    for metric in METRICS:
      for baseline in BASELINES:
        file_means = metrics.groupby(["batch", "condition_id", "file", "policy"])[metric].mean().unstack("policy")
        file_means = file_means.dropna(subset=[baseline,"C7-Core"])
        if (file_means[baseline].abs() <= 1e-12).any():
            raise ValueError(f"zero relative-effect denominator in {metric}/{baseline}")
        file_means["effect"] = 100*(file_means["C7-Core"]-file_means[baseline])/file_means[baseline]
        flat = file_means.reset_index()
        by_condition = {str(c): group.effect.to_numpy(float) for c,group in flat.groupby("condition_id")}
        condition_names = np.array(sorted(by_condition),dtype=object)
        by_batch = {str(batch):{str(c): group.effect.to_numpy(float) for c,group in bgroup.groupby("condition_id")}
                    for batch,bgroup in flat.groupby("batch")}
        batch_names = np.array(sorted(by_batch),dtype=object)
        point = float(file_means.effect.groupby(level="condition_id").mean().mean())
        condition_draws = np.empty(DRAWS)
        batch_draws = np.empty(DRAWS)
        for draw in range(DRAWS):
            sampled_values = []
            for condition in condition_rng.choice(condition_names,size=len(condition_names),replace=True):
                files = by_condition[str(condition)]
                sampled_values.append(float(condition_rng.choice(files,size=len(files),replace=True).mean()))
            condition_draws[draw] = np.mean(sampled_values)
            sampled_values = []
            for batch in batch_rng.choice(batch_names,size=len(batch_names),replace=True):
                conditions = np.array(sorted(by_batch[str(batch)]),dtype=object)
                for condition in batch_rng.choice(conditions,size=len(conditions),replace=True):
                    files = by_batch[str(batch)][str(condition)]
                    sampled_values.append(float(batch_rng.choice(files,size=len(files),replace=True).mean()))
            batch_draws[draw] = np.mean(sampled_values)
        condition_ci = np.percentile(condition_draws,[2.5,97.5])
        batch_ci = np.percentile(batch_draws,[2.5,97.5])
        p = comparison_row(published,metric,baseline)
        b = comparison_row(clustered,metric,baseline)
        error_values = [abs(point-float(p.point_pct)), abs(point-float(b.point_pct)),
                        float(np.max(np.abs(condition_ci-[p.ci_low_pct,p.ci_high_pct]))),
                        float(np.max(np.abs(batch_ci-[b.batch_clustered_ci_low_pct,b.batch_clustered_ci_high_pct]))),
                        abs(float(condition_draws.std(ddof=1))-float(p.bootstrap_se_pct)),
                        abs(float(batch_draws.std(ddof=1))-float(b.batch_clustered_se_pct))]
        if "bootstrap_mean_pct" in p:
            error_values.append(abs(float(condition_draws.mean())-float(p.bootstrap_mean_pct)))
        if "condition_bootstrap_ci_low_pct" in b:
            error_values.append(float(np.max(np.abs(condition_ci-[b.condition_bootstrap_ci_low_pct,b.condition_bootstrap_ci_high_pct]))))
        errors.extend(error_values)
        if int(p.n_conditions) != len(condition_names) or int(p.n_passes) != len(flat) or int(b.n_batches) != len(batch_names):
            raise ValueError("published inferential unit counts disagree with distributed pass rows")
        rows.append({"metric":metric,"comparison":f"C7-Core vs {baseline}","point_pct":point,
                     "condition_ci":condition_ci.tolist(),"batch_ci":batch_ci.tolist(),
                     "n_conditions":len(condition_names),"n_files":len(flat),"n_batches":len(batch_names),
                     "max_publication_abs_difference":max(error_values),
                     "batch_relative_effects_pct":flat.groupby(["batch","condition_id"]).effect.mean().groupby("batch").mean().to_dict()})
    maximum = float(max(errors))
    return {"passed":maximum <= ATOL,"seed":SEED,"draws_each":DRAWS,"comparisons":rows,
            "max_publication_abs_difference":maximum,
            "scope":"Independent recomputation of public file-relative estimands and both nested bootstrap protocols; not raw-data replay or extra production replication."}


def table_audit(metrics: pd.DataFrame, results: Path) -> dict:
    episode_cols = ["episodes_per_100m","integrated_severity_excess_m","peak_excess_norm"]
    episodes = equal_condition(metrics,episode_cols)
    published_ep = normalize_policy(pd.read_csv(results/"canonical_episode_condition_weighted.csv")).set_index("policy")
    episode_error = float((episodes-published_ep.loc[episodes.index,episode_cols]).abs().max().max())
    table3_cols = METRICS+["S_out"]
    aggregate = equal_condition(metrics,table3_cols)
    table3 = normalize_policy(pd.read_csv(results/"canonical_nominal_condition_weighted.csv")).set_index("policy")
    table3_error = float((aggregate-table3.loc[aggregate.index,table3_cols]).abs().max().max())
    file_band = normalize_policy(pd.read_csv(results/"recorded_band_decomposition_by_file.csv"))
    cols = ["below_frequency","above_frequency","recorded_band_departure_frequency",
            "below_mean_distance","above_mean_distance","two_sided_mean_distance",
            "below_cvar95_component","above_cvar95_component","two_sided_cvar95"]
    band = equal_condition(file_band,cols)
    published_band = normalize_policy(pd.read_csv(results/"recorded_band_decomposition_summary.csv")).set_index("policy")
    band_error = float((band-published_band.loc[band.index,cols]).abs().max().max())
    identity_errors = [float((band.below_frequency+band.above_frequency-band.recorded_band_departure_frequency).abs().max()),
                       float((band.below_mean_distance+band.above_mean_distance-band.two_sided_mean_distance).abs().max()),
                       float((band.below_cvar95_component+band.above_cvar95_component-band.two_sided_cvar95).abs().max()),
                       float((band.two_sided_mean_distance-aggregate.S_excess_mean).abs().max()),
                       float((band.two_sided_cvar95-aggregate.S_excess_cvar95).abs().max()),
                       float((band.recorded_band_departure_frequency-aggregate.S_out).abs().max())]
    pid_minus_c7 = band.loc["PID",["below_mean_distance","above_mean_distance","two_sided_mean_distance"]]-band.loc["C7-Core",["below_mean_distance","above_mean_distance","two_sided_mean_distance"]]
    anti = pd.read_csv(results/"Table_S12c_anti_windup.csv")
    effects = []
    for mode,group in anti.groupby("Mode"):
        wide = group.set_index("Policy").rename(index={"C7":"C7-Core"})
        for base in BASELINES:
            effects.append({"mode":str(mode),"baseline":base,
                "mean_distance_aggregate_ratio_pct":float(100*(wide.loc["C7-Core","Mean two-sided band distance"]/wide.loc[base,"Mean two-sided band distance"]-1)),
                "tail_distance_aggregate_ratio_pct":float(100*(wide.loc["C7-Core","Upper-tail band distance"]/wide.loc[base,"Upper-tail band distance"]-1))})
    max_error = max([episode_error,table3_error,band_error,*identity_errors])
    return {"passed":max_error<=ATOL,"max_publication_abs_difference":max_error,
            "table3_max_abs_difference":table3_error,"episode_max_abs_difference":episode_error,
            "episode_frequency_rank_ascending":episodes.episodes_per_100m.sort_values().to_dict(),
            "episodes_equal_condition":episodes.to_dict("index"),
            "band_max_abs_difference":band_error,"band_additivity_and_metric_crosscheck_max_error":max(identity_errors),
            "band_equal_condition":band.to_dict("index"),
            "PID_minus_C7_mean_distance_components":pid_minus_c7.to_dict(),
            "antiwindup_descriptive_aggregate_ratio_effects":effects,
            "scope":"File -> equal condition aggregation and additive historical-band identities; ratios in the anti-windup screen are descriptive, not Table 4 paired estimates."}


def isolated_functions(path: Path, names: set[str], namespace: dict) -> None:
    definitions = [node for node in ast.parse(path.read_text(encoding="utf-8-sig")).body
                   if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError(f"requested solver definitions not found in {path.name}")
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(path),"exec"),namespace)


def diagonal_mpc_audit(root: Path) -> dict:
    suite_namespace = {"np":np,"RealPass":Any}
    isolated_functions(root/"code"/"engine_core"/"15_realdata_closed_loop_suite.py",{"response_matrix"},suite_namespace)
    suite = SimpleNamespace(response_matrix=suite_namespace["response_matrix"])
    controller_namespace = {"np":np,"Any":Any}
    isolated_functions(root/"code"/"controller_replay.py",{"response_matrix_custom","solve_mpc_custom"},controller_namespace)
    solver = controller_namespace["solve_mpc_custom"]
    maximum_raw = maximum_applied = maximum_residual = 0.0
    count = 4000
    for index in range(count):
        q = (1+np.sin((index+1)*np.array([1.324,2.417,3.136])))/2
        scales = 10**(-5+8*q)
        settings = SimpleNamespace(tension_scale=scales[0],thickness_scale=scales[1],flatness_scale=scales[2],
                                   exit_thickness=float(.02+.73*q[0]),reduction_ratio=float(.6*q[1]))
        normalized_error = np.cos((index+1)*np.array([1.75,3.254,2.881]))*(.01+12*q[2])
        predictions = np.tile(normalized_error*scales,(10,1))
        weights = np.array([5.5,6.,5.5])
        penalty = np.array([16.,20.,18.])
        response = suite.response_matrix(settings,10,False)/scales.reshape(1,3,1)
        diagonal = np.stack([np.diag(matrix) for matrix in response])
        closed_move = -weights*diagonal.sum(axis=0)/(penalty+weights*np.square(diagonal).sum(axis=0))*normalized_error
        design = np.vstack([np.sqrt(weights).reshape(3,1)*matrix for matrix in response])
        rhs = np.tile(np.sqrt(weights)*normalized_error,10)
        hessian = design.T@design+np.diag(penalty)
        direct_move = -np.linalg.solve(hessian,design.T@rhs)
        maximum_raw = max(maximum_raw,float(np.max(np.abs(direct_move-closed_move))))
        maximum_residual = max(maximum_residual,float(np.max(np.abs(hessian@closed_move+design.T@rhs))))
        previous = np.sin((index+1)*np.array([2.123,.731,1.889]))*.55
        lower,upper = np.full(3,-.55),np.full(3,.55)
        actual = solver(suite,predictions,weights,settings,previous,lower,upper,"diag",False)
        limits = np.array([.095,.080,.095])
        expected = np.clip(previous+np.clip(closed_move,-limits,limits),lower,upper)
        maximum_applied = max(maximum_applied,float(np.max(np.abs(actual-expected))))
    return {"passed":maximum_raw<1e-12 and maximum_applied<1e-12,"cases":count,
            "max_unclipped_move_difference":maximum_raw,"max_internal_command_difference":maximum_applied,
            "max_normal_equation_residual":maximum_residual,
            "scope":"Canonical diagonal response; held ten-stage error, fixed weights, unit fixed move penalty; internal increment and command clipping included. Does not test all noncanonical modes."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root",type=Path,default=ROOT)
    parser.add_argument("--audit-json",type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    results = root/"results"
    metrics = normalize_policy(pd.read_csv(results/"canonical_nominal_pass_metrics.csv"))
    inventory = pd.read_csv(root/"data_map"/"production_batch_map.csv")
    canonical_inventory = inventory.loc[inventory.role.eq("canonical_data2")].set_index("pass_id")
    if set(metrics.pass_id)!=set(canonical_inventory.index):
        raise ValueError("public pass metrics do not match canonical Data2 membership")
    expected_conditions = metrics.pass_id.map(canonical_inventory.analysis_condition)
    expected_batches = metrics.pass_id.map(canonical_inventory.batch)
    if not metrics.condition_id.eq(expected_conditions).all() or not metrics.batch.eq(expected_batches).all():
        raise ValueError("published condition or batch identities differ from de-identified input map")
    numeric = metrics[METRICS+["episodes_per_100m","integrated_severity_excess_m","peak_excess_norm"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite canonical statistics")
    report = {"scope":"Independent public-table recomputation; not raw-data replay, historical training reconstruction or plant validation.",
              "numpy_version":np.__version__,"pandas_version":pd.__version__,
              "production_units":{"data2_batches":int(canonical_inventory.batch.nunique()),"data2_passes":len(canonical_inventory),
                                  "data2_conditions":int(canonical_inventory.analysis_condition.nunique()),
                                  "data1_batches":int(inventory.loc[inventory.role.eq("data1_source"),"batch"].nunique())},
              "tables":table_audit(metrics,results),"paired_bootstrap":bootstrap_audit(metrics,results),
              "tail_cardinality":tail_cardinality_audit(results),
              "diagonal_mpc":diagonal_mpc_audit(root)}
    report["passed"] = all(report[name]["passed"] for name in ["tables","paired_bootstrap","tail_cardinality","diagonal_mpc"])
    if args.audit_json is not None:
        args.audit_json.parent.mkdir(parents=True,exist_ok=True)
        args.audit_json.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
