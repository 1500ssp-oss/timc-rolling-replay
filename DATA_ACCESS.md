# Data access

Proprietary raw manufacturing records are confidential and are not redistributed.
The public materials are frozen predictor weights, analysis code, versioned
configurations, de-identified input hashes, publication-level aggregate tables,
figures and audit summaries. Raw records, feature windows and production-derived
row-level trajectories are excluded.

An authorized replay requires legal access to all 26 original pass CSV files.
The files may be nested under a private root; `code/run_pipeline.py` locates them
by the SHA256 values in `data_map/production_batch_map.csv`. Original folder and
file names are not required. The pipeline loads the frozen model; it does not
reconstruct historical model fitting or selection.

Run the full replay in a separate private working copy. Its feature, prediction,
state, command, episode and step-log trajectories remain confidential. Do not
commit or upload raw inputs or generated trajectories. Published pass-,
condition- and batch-level summaries are aggregate evidence, rather than raw
production records.

Without the raw archive, `python verify_results.py` and the public-statistics
audit check file identities and selected arithmetic, configuration and
implementation contracts. They do not regenerate every result from raw data,
reconstruct historical training or validate a physical rolling mill.

`PROVENANCE_LIMITATIONS.txt` explains the distinct scopes of release-file hashes
and the source identifiers in individual diagnostic receipts.
