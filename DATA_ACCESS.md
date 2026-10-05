# Data access

The raw production archive is not redistributed because it contains proprietary manufacturing records. The package includes the complete de-identified pass map, frozen configurations, publication-level result tables, and all analysis code. To rerun the replay, supply a root containing the 26 authorized raw pass CSV files. Files may be nested; the pipeline locates them recursively by the SHA-256 digests in `data_map/production_batch_map.csv`.

An authorized full-data rerun generates row-level step logs used by the episode, command, corridor and state-guardrail audits. Because those logs retain production-derived row trajectories, they are not included in the redistributed package. Aggregate QA and publication tables derived from them are included under `results/` and checked by `verify_results.py`.
