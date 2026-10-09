# Reproducibility / auditability package

Submission correction prepared on 9 October 2026. The original public snapshot
`3d020e942b56e363f940bc1d8c91d3140b002cf0` remains the historical source record.
This correction supplements that original release. It retains the frozen model
weights, settings, publication results and figures; it restores missing motion
diagnostic aggregation and includes the GitHub support files in the integrity
manifest. See `SUBMISSION_VERSION.md` and `PROVENANCE_LIMITATIONS.txt` for the
version distinction and the recovered historical Farch source versions.

This package contains the analysis code, frozen controller configuration, one final thickness predictor, publication-level result tables, and figure files for the manuscript. It deliberately excludes raw proprietary production records, exploratory notebooks, console logs, caches, intermediate checkpoints, and non-publication outputs.

Positioning: this is an **auditability package**. The raw production archive cannot be released, so the package supports (a) offline file-integrity verification and selected numerical/configuration cross-checks, and (b) a full replay rerun on an authorized machine where the 26 raw pass files are available (`code/run_pipeline.py`). The pipeline discovers those inputs recursively by their published SHA256 digests; original folder and file names are neither required nor distributed. `verify_results.py` checks the frozen lock, selected publication numbers, rate-table 1e-12 anchors, figure assets and MANIFEST hashes without needing the data. These checks do not independently reconstruct every reported value or validate the response emulator against a physical mill.

Historical-source recovery confirmed on 9 October 2026. Both source versions
named in the retained Farch audit have been recovered byte for byte and
verified against its recorded SHA256 values. They are retained separately
as historical evidence; current replay code and the original audit JSON
remain unchanged. This resolves the missing-source identity issue, without
claiming a new replay or complete historical training reconstruction.

## Reviewer quick start

Open a terminal in the extracted package root (the directory containing this
README). In a Python environment with `requirements.txt` installed, run:

```sh
python verify_results.py
python -m unittest discover -s tests -p "test_*.py" -v
python code/public_statistics_audit.py
```

These commands use only public files. They check integrity, regression cases
and selected published calculations; they do not replay confidential records.
Start reading at `code/run_pipeline.py` (workflow),
`code/controller_replay.py` (replay/controller execution),
`code/batch_archive.py` (pass aggregation), and `code/run_relocking.py`
(source-omission analysis). Frozen settings, predictor weights and reported
tables are in `config/`, `models/` and `results/`, respectively. The full-replay
command and input requirements are given below; `DATA_ACCESS.md` and
`PROVENANCE_LIMITATIONS.txt` state the access and historical-evidence limits.

## Evidence boundary

The primary evidence is timestamp-aware archived-update replay. The response-family, projection, fault and local OPC UA tests are sensitivity or integration screens; they are not plant-side closed-loop validation.

The data owner authorized research use of the records and publication of de-identified summaries, and retains the source production archive. The proprietary raw records are not available for external distribution. The distributed `data_map/production_batch_map.csv` contains de-identified SHA256 hashes of the fixed analysis inputs; raw production files are not distributed. Data1 supplies the predictor fit, the equal-pass inner-PID grid, the phase-specific recorded operating band and a frozen catalogue of six pass scales. PID-grid candidates are run through the canonical segmented PID replay, scored by the arithmetic mean of the three channelwise normalized RMS values, and differ only in the PID gain multipliers and internal saturation scale; the canonical gain scale, seed rule, measurement path and outer amplitude/slew shell remain fixed. The remaining controller-family, projection and response constants are versioned engineering settings held fixed within each released replay. Data2 is not used to estimate controller or emulator scales.

Rate calculations use the supported segment-local control interval, not an unsupported timestamp gap. Raw gaps are retained for audit; observed distance excludes unsupported segment boundaries. The value 0.50 s is the documented move-limit conversion reference and an explicitly labelled counterfactual comparison, not the replay sampling period.

## Supported command motion

`TV_L_per_100m` is 100 times the applied-command L1 adjacent variation,
restricted to transitions within the same contiguous supported segment, divided
by supported observed distance. Segment joins and reset-zero initialization
actions are excluded. Original row order is retained; a recurring segment label
does not reconnect separate runs. A one-row segment contributes zero variation.
Zero supported distance gives an undefined normalized TV, rather than an
artificial large value. The same transition rule applies to
`control_total_variation`, preclamp/postclamp and raw/component MPC variation,
`control_delta_rms`, and peak relative command change.

`startup_motion` is the applied L1 move from reset zero at the file's first
command. `restart_motion` sums such moves at later contiguous-segment starts;
`startup_restart_motion` is their sum. These unnormalized diagnostics are
distinct from supported TV. Logged `du_*` values remain the executed changes,
including initialization actions, and are never silently rewritten.

The actual production paths are tested in
`tests/test_supported_command_motion.py`. `code/supported_motion_audit.py`
independently checks public aggregate identities; with `--canonical-step-log`
on an authorized machine it also reconstructs the segmented numerators from
retained command rows. Public identities alone are not raw-data reconstruction.
The recorded `results/supported_motion_audit.json` contains aggregate evidence,
not proprietary trajectories.

For the historical supported-motion release of 30 September 2026, complete
retained canonical command logs were reaggregated.
Data1, expanded and fixed-period trajectories were regenerated in the verified
environment to keep their scientific summaries aligned with the current code.
Scenarios without complete logs were regenerated with the same raw-input
hashes, frozen weights, configurations and seeds, or reused only
after their execution signatures and full trajectories were shown to match an
available run. Non-motion outcomes were checked against the retained results;
original execution-timing measurements were preserved. Historical fixed-period
differences were recorded and their scientific aggregates were replaced by the
current replay results. This targeted refresh
is distinct from executing the entire historical fitting and selection process.
The 9 October 2026 correction retains those published results unchanged; it
does not perform a new raw-data replay or repeat that historical refresh.
The full replay command below generates the supported-motion audit anew.

## Executed implementation

The released replay pipeline enforces segment-local causal measurement playback with zero nominal synthetic measurement noise, shared stochastic settings across policies within a pass/scenario, C2 without the EKF observer, segment-disciplined predictor windows and innovations, forward-fill-only missing-data handling, strict upper-tail CVaR, a non-duplicated LinearARX design matrix, and sensitivity configurations generated from the canonical controller lock. See `DATA1_LOCKING.md` for the corresponding execution constraints and Data1 locking provenance.

## Recorded-band metric semantics

The retained legacy fields `S_out`, `S_excess_mean` and
`S_excess_cvar95` are compatibility names. They describe departure from the
two-sided, phase-specific Data1 recorded operating band; they are not
one-sided flatness-quality or safety metrics. The recorded `flatness_std`
measurement is non-negative with nominal reference zero, but the engineering
emulator propagates a signed replay coordinate and does not apply an absolute
value. Negative signed replay rows, when present,
are not physical predictions of negative dispersion. The executed row count is
reported in `results/replay_flatness_distribution_audit.json`; it is not a
physical-validation statistic. For a normalized signed
replay row `z` and recorded-band limits `[l, q]`, the released
decomposition is `below=max(l-z,0)`, `above=max(z-q,0)`, and
`total=below+above`. The lower and upper CVaR components use exactly the same
largest-5% row set selected by `total`, so their sum reproduces the retained
total upper-tail metric.

`code/recorded_band_decomposition.py` rebuilds these quantities from the
canonical step log, pass-specific frozen scale, and phase envelope. It writes
file-, condition-, and equal-condition policy summaries plus a guardrail-row
ledger to `results/recorded_band_decomposition_*.csv` and
`results/recorded_band_guardrail_row_composition.csv`. The guardrail flags are
transition flags: a flag logged at row k is shifted to the reached row k+1
within the same segment. The resulting composition is descriptive only and
does not assign a causal contribution to the numerical guardrail. In this
canonical run, the reached-row composition is reported in the released ledger;
the symmetric +/-8 boundary is a versioned numerical setting without physical
calibration.

## Environment

Python 3.11 or newer is recommended. Install `requirements.txt`. The released replay loads the frozen predictor on CPU; no retraining is performed by the master replay pipeline.

## Predictor timeline and training reproducibility

At control row k the predictor consumes archived features through k and targets
k+1. Ineligible segment-local windows fall back to the current measurement.
The ten internal planning stages repeat that one-step vector; they are not ten
learned future predictions. `tests/test_predictor_timeline.py` tests the executed
cache against synthetic windows and direct frozen-GRU inference without raw data.

The published model-selection aggregates and frozen checkpoint are retained.
The original per-fold/seed scores, fitting curves and actual checkpoint fitting
budget/seed/device could not be recovered. `TRAINING_AUDIT.md` and
`results/predictor_training_provenance_audit.json` distinguish aggregate
consistency from incomplete historical fitting provenance. Instrumented training
scripts record these fields for future runs; they do not retroactively recover
the historical fit. A full replay rerun uses the released frozen weights and
does not regenerate predictor training or the historical selection summary.

The original release recorded verification on Windows 11 with Python 3.14.5, NumPy 2.3.5, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.8.0, Matplotlib 3.10.9, PyTorch 2.13.0+cpu, and asyncua 2.0.1. These versions record the verified environment; `requirements.txt` retains lower bounds so that CPU/GPU and platform-specific PyTorch builds remain installable.

## Full rerun (requires the production batch archive)

From the package root:

```powershell
python code/run_pipeline.py --fresh --batch-root "C:\path\to\batch-archive"
```

With authorized access to the hash-matched inputs, this entry point executes
the current frozen-policy replay and writes its publication tables to `results/`
for comparison with the retained manuscript results. It does not reconstruct
historical predictor fitting or selection runs, and a new run is not a promise
of identical floating-point bytes across software/hardware environments.

The correction carries the existing supported-motion definitions through the
expanded, repeat-sequence and source-omission summaries. The original release
had added eleven diagnostic columns through internal postprocessing that was
not included in its public workflow. This reporting correction does not change
controller actions, parameters, response equations or the retained result CSVs.

Use a separate private working copy for a full rerun: its row-level outputs
remain confidential. Do not upload raw inputs or generated trajectories to
GitHub. The public package permits offline integrity and selected arithmetic
checks; possession of the model weights does not provide the confidential data.

The optional `--audit-published-pid-grid` reuses the complete published
1050-candidate PID grid, validates its candidate coverage and equal-pass scores,
and replays the selected candidate on raw Data1. Without that option the whole
PID grid is recomputed. The option does not skip any replay-dependent evaluation,
sensitivity, bootstrap, command or episode analysis.

## Offline verification (no data required)

```powershell
python verify_results.py
```

The offline verifier checks file integrity, registered numerical identities and
the actual prediction, tail-cardinality and supported-command-motion paths.
The regression suite runs without proprietary data:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Top-tail means use exactly `ceil((1-q)*N)` observations, with the displayed
decimal `q` evaluated without floating-point ceiling drift. Ties do not expand
the tail and no fractional boundary weighting is used.

## Primary replay (individual step)

From the package root, replace `<BATCH_ROOT>` with a directory tree containing the 26 authorized raw pass CSV files:

```powershell
python code/batch_sweep.py --out-root outputs/nominal --run-root config/reference_run --batch-root "<BATCH_ROOT>" --extension-script code/batch_archive.py --config-json config/selected_controller_configs.json --dataset Data2 --predictor-bank-path models/final_thickness_model.pt
```

## Relative outer-rate sensitivity

Table S12 uses relative multipliers of each policy-specific locked outer slew
scale, not common absolute gamma values. After running the sweep with
`config/implementation_sensitivity/rate_sensitivity_configs.json`, build and
audit the publication table with:

```powershell
python code/build_rate_sensitivity_table.py --metrics <RATE_RUN>/stage_allocation_ablation_metrics.csv --canonical results/canonical_nominal_condition_weighted.csv --output results/Table_S12_rate_sensitivity.csv --audit results/Table_S12_rate_sensitivity_anchor_audit.json
```

The script requires the multiplier 1.00 rows to reproduce canonical TV, mean
excess and amplitude-activation values within `1e-12`; it exits with an error
otherwise. This holds exactly because the replay seed depends only on the pass
index, so an identical configuration receives the same segment-start realization and, when an explicit noise sensitivity is active, the same measurement-noise realization.

The batch map defines the 13-pass source archive, the 10-pass primary evaluation archive and the three-pass repeat-sequence extension. Publication summaries use eight condition groups, with the speed-ramp and nominal traces for the 0.628 to 0.458 transition retained as separate conditions.

The pipeline verifies `config/data1_scale_lock.json` against the thirteen raw Data1 files before loading Data2. For each replay pass, the six controller/emulator scales are taken from the nearest Data1 recipe in Euclidean log(entry-gauge), log(exit-gauge) space, with a lexical pass-ID tie-break. The selected Data1 source pass and matching distance are published in `results/data1_scale_lock_application.csv`.

Full-data reruns generate row-level step logs for episode, command and corridor audits. Those logs retain production-derived row trajectories and are therefore not redistributed. The public package contains their aggregate time-source, projection, state-guardrail and result-consistency tables instead.

## Package layout

- `code/`: replay, model-selection, stress, command-audit, corridor-sensitivity, local OPC UA scripts, the master `run_pipeline.py`, and the publication assembly scripts (`publish_tables.py`, `publish_figures.py`).
- `config/`: versioned policy settings, model selection, reference envelope, the Data1-only scale catalogue, and lock-generated sensitivity configurations (`build_sensitivity_configs.py`).
- `models/`: final GRU thickness predictor used by C2 and C7-Core.
- `data_map/`: de-identified batch/pass identity map.
- `results/`: exact publication-level tables from the executed analysis pipeline.
- `figures/`: final manuscript figures.
- `MANIFEST_SHA256.csv`: integrity hashes (regenerated by the pipeline).
- `DATA1_LOCKING.md`: Data1 locking provenance and execution constraints.

Figure filenames follow their printed numbering: `Figure_1.png`--`Figure_4.png` in the main article and `Figure_S1.png`--`Figure_S3.png` in the supplementary material. Protocol identifiers and local loopback endpoints record execution provenance.

## Final editorial release (4 October 2026)

The scientific tables, frozen models/configurations and replay implementation are retained from the supported-motion release. Figure 2 now separates the RMS horizontal scale from the three larger-scale metrics; Figure S3 uses the same descending raw-gap pass order in both panels. The Table S1 source IDs resolve to complete relative paths and hashes in `results/claim_source_index.csv`. Bootstrap intervals remain pointwise, unadjusted exploratory summaries of the 12 metric–baseline comparisons; this presentation update does not add simultaneous coverage or a new scientific experiment.
