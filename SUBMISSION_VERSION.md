# TIMC submission code package

Associated manuscript: **Timestamp-aware assessment of control policies using
archived ultra-thin strip rolling data**.

This package contains 179 files. `MANIFEST_SHA256.csv` records the paths, byte
counts and SHA256 hashes of the other 178 distributed files. The fixed GitHub
commit linked in the manuscript and code-availability note identifies this
submission version. Use that commit permalink when citing or downloading it.

Read `README.md` for the workflow, core modules, metric definitions and commands.
From the extracted package root, with the dependencies in `requirements.txt`
installed, run:

```sh
python verify_results.py
python -m unittest discover -s tests -p "test_*.py" -v
python code/public_statistics_audit.py
```

The package provides frozen-weight inference and checks on public files.
It includes code, versioned settings, model weights, de-identified input hashes,
publication-level tables, figures and diagnostic receipts. It excludes raw
production records and production-derived row trajectories. Authorized full
replay requires all 26 hash-matched inputs in a separate private working copy.

`DATA_ACCESS.md` describes input access. `PROVENANCE_LIMITATIONS.txt` describes
release integrity, diagnostic-receipt scope and the limits of verification.
Historical predictor fitting is not reconstructed by the replay pipeline.
The evidence is archived-data replay, without plant-side closed-loop validation.
