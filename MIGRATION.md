# Using this package after moving or cloning it

The maintenance revision fixes path resolution in the released replay code. It
retains the published weights, numerical configuration, input hash map, result
tables and figures. It does not recover the unavailable historical training
provenance. The original submission snapshot remains available at commit
`3d020e942b56e363f940bc1d8c91d3140b002cf0`.

## Public checks

With Python 3.11 or newer, the path regression tests use only the standard
library and require neither raw data nor the scientific dependencies:

```powershell
python -B tests/test_migration_paths.py
```

From a different working directory, use the full path to that test file. The
test locates the repository from its own file path. It creates only synthetic
temporary test files and removes them afterwards.

After installing `requirements.txt` in an environment outside this repository,
the existing offline checks are:

```powershell
python -B verify_results.py
python -B -m unittest discover -s tests -p "test_*.py" -v
```

Path tests and package hashes are checks of migration behavior and file
consistency. They do not demonstrate a complete scientific replay on a new
machine. See [README.md](README.md) for the recorded scientific environment and
[TRAINING_AUDIT.md](TRAINING_AUDIT.md) for historical fitting limits.

## Optional local path profile

Public files contain no workstation paths. An authorized data holder may copy
the example to the ignored local profile:

```powershell
Copy-Item migration_paths.example.json migration_paths.json
```

Set `external.archive_root` to a directory containing the 26 authorized raw
pass files. Files can be nested; their published SHA256 digests determine the
matches. `external.data1_root` and `external.data2_root` support the separate
Data1/Data2 utilities and are unnecessary for the master archive replay.
Values may be absolute or relative to the profile file's own directory. Null
or empty fields mean unset. Keep raw files outside this repository.

The default profile is `migration_paths.json` in the repository root. To use a
profile stored elsewhere, set `TIMC_PATHS_CONFIG` to its path. For example,
replace this generic placeholder with an authorized local file:

```powershell
$env:TIMC_PATHS_CONFIG = "<path-to-local-profile.json>"
```

Explicit `ROLLING_DATA1_DIR` or `ROLLING_DATA2_DIR` values override the
corresponding profile field. Relative environment values, including
`TIMC_PATHS_CONFIG`, are resolved against the caller's working directory before
subprocesses change directory. Package-owned paths recorded in configuration,
including the GRU weights and stress scenario configurations, are resolved
against the repository root. Explicit relative CLI paths continue to use the
caller's working directory.

`TIMC_DATA1_SCALE_LOCK` can identify a lock for standalone utilities. The master
replay requires it to resolve to this repository's
`config/data1_scale_lock.json`; it rejects a different lock. Historical disabled
training/data entry points remain disabled.

## Authorized replay

Run the following only on a machine with the scientific dependencies and
authorized raw data. Work from an independent copy, since the replay writes
publication outputs and `--fresh` clears that copy's `outputs` directory:

```powershell
# Explicit archive path, resolved against the caller's working directory:
python code/run_pipeline.py --fresh --batch-root "<authorized-archive-root>"

# Alternatively, use external.archive_root from the local profile:
python code/run_pipeline.py --fresh
```

From outside the repository, use the full path to `code/run_pipeline.py`. No
training, replay or sensitivity process was run to prepare this migration
patch. The frozen predictor is reused by the master replay; no retraining is
required by that command.

## Confidential files and integrity coverage

Do not commit raw production CSVs, row-level replay logs, local path profiles,
or environment directories. The ignore rules and shared manifest enumerator
exclude the default `data1/`, `data2/`, `production_batch_root/`, `private_data/`,
`raw_data/`, `batch_archive/`, `outputs/`, `.venv/`, `venv/` and cache directories,
local `migration_paths.json` / `migration_paths.local*.json`, and generated
log/trajectory archives. These rules do not detect confidential files stored
under arbitrary names; keep those files outside the repository and inspect
files before publishing. Frozen `models/` weights remain included.

The maintenance `MANIFEST_SHA256.csv` covers the current public package,
including `.gitattributes`, `.gitignore`, `SUBMISSION_VERSION.md` and the new
migration files. The writer and verifier share `code/package_manifest.py`.
The manifest does not hash itself. The original 174-entry manifest remains in
the immutable submission commit; this revision does not claim to retain all
original source files byte for byte.
