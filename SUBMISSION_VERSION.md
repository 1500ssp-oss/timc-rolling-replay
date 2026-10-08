# TIMC submission snapshot

Associated manuscript: **Timestamp-aware assessment of control policies using archived ultra-thin strip rolling data**.

The original reproducibility / auditability snapshot was frozen on 2026-10-04
and published on 2026-10-05 at commit
`3d020e942b56e363f940bc1d8c91d3140b002cf0`. At that commit, the 175 files
from the original submission code ZIP are retained byte for byte, with three
additional GitHub support files. The current maintenance revision changes
path handling, tests, documentation and manifest coverage; use the original
commit when citing the unchanged submission source snapshot.

Original `Reproducibility_Code.zip` SHA256:

```text
ac11cbed2887c0351530b6a17a6729dc4db55bf17aa9cf7c1a6e628a5d4cac27
```

The original `MANIFEST_SHA256.csv` covers the other 174 files in the frozen
package, excluding the three GitHub support files. The maintenance manifest
covers all current public package files, including those support files and new
migration files. For review, use a URL containing the full Git commit SHA
rather than the moving `main` branch.

## Maintenance revision, 2026-10-08

Package resources now resolve from the repository root; local profile values
resolve from their profile directory; explicit CLI and environment paths
resolve from the caller's working directory. Subprocesses retain these resolved
paths when their working directory changes. The master replay keeps the frozen
Data1 scale-lock requirement and uses the released predictor weights.

The 118 files under `config/`, `models/`, `data_map/`, `results/` and `figures/`
remain byte for byte identical to the submission commit. Standard-library
migration tests and hashes validate this patch without opening raw data. No
training or full replay was run for this maintenance revision. See
[MIGRATION.md](MIGRATION.md) for commands, confidentiality boundaries and the
optional example profile.

## Verification without private data

From the repository root, install the dependencies listed in `requirements.txt`, then run:

```text
python verify_results.py
python -m unittest discover -s tests -p "test_*.py" -v
```

These checks verify package integrity and selected numerical/configuration identities. A full replay rerun requires authorized access to the 26 proprietary raw pass files; the public package does not independently regenerate every reported result or the historical predictor fitting and selection process. See [README.md](README.md), [DATA_ACCESS.md](DATA_ACCESS.md), and [TRAINING_AUDIT.md](TRAINING_AUDIT.md) for the full scope and execution instructions.

## Public material and confidentiality

The code, frozen predictor weights, controller parameters, data-derived scales/quantiles, de-identified input hashes, publication-level result tables and figure assets are included within the author's confirmed publication scope. Proprietary raw production records and row-level production trajectories are confidential and are not distributed.

The evidence concerns archived-data replay and sensitivity/integration screens. It is not plant-side closed-loop validation, and response-emulator trajectories are not validated physical mill predictions.
