# TIMC submission snapshot

Associated manuscript: **Timestamp-aware assessment of control policies using archived ultra-thin strip rolling data**.

This repository distributes the reproducibility / auditability package frozen on 2026-10-04 and prepared for GitHub publication on 2026-10-05. The 175 files from the original submission code ZIP are retained byte for byte. The repository adds only this note, Git attributes to preserve file hashes across platforms, and ignore rules for local/private inputs and generated outputs.

Original `Reproducibility_Code.zip` SHA256:

```text
ac11cbed2887c0351530b6a17a6729dc4db55bf17aa9cf7c1a6e628a5d4cac27
```

`MANIFEST_SHA256.csv` covers the other 174 files in the frozen package. The three GitHub-specific support files are outside that original manifest. For review, use a URL containing the full Git commit SHA rather than the moving `main` branch; GitHub's archive download for that commit provides the same frozen package contents plus these three support files.

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
