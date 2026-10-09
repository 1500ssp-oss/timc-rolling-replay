# TIMC submission correction

Correction prepared on 9 October 2026; the original public snapshot is retained.

Associated manuscript: **Timestamp-aware assessment of control policies using
archived ultra-thin strip rolling data**.

## Recovery-confirmed documentation update

The preceding correction is retained at fixed commit
`edae92663772cd426cabb7d66d1ba5c35838cad8`. This follow-up changes only the
four documentation files and their integrity-manifest entries after receipt
of the historical source recovery archive. All current Python source files,
tests, configurations, weights, publication results and figures are unchanged
from that correction. The historical source archive is stored separately
and is not part of the current executable code package.

## Original public source record

The original frozen submission remains available at commit
`3d020e942b56e363f940bc1d8c91d3140b002cf0`. This correction does not modify
that commit. The original 175-file code ZIP SHA256 was
`ac11cbed2887c0351530b6a17a6729dc4db55bf17aa9cf7c1a6e628a5d4cac27`.
Those identities describe the original release, not this corrected archive.

## Changes in this correction

The manifest now covers every distributed file except itself, including
`.gitattributes`, `.gitignore`, this version note and `PROVENANCE_LIMITATIONS.txt`.
The verifier's existing coverage rule is preserved. This removes the original
GitHub-directory coverage failure without exempting unlisted support files.

Expanded, repeat-sequence and source-omission aggregations now retain eleven
supported-motion diagnostics that the original results received through
historical internal postprocessing. Controller algorithms, locked parameters,
model weights, response equations, publication result CSVs and figures are
unchanged. This is a reporting and documentation correction.

The original Farch audit JSON is retained byte for byte. The two historical
source versions previously reported as missing have now been recovered and
independently verified against its recorded hashes. They differ from the
current implementation and are retained separately as historical evidence.
`PROVENANCE_LIMITATIONS.txt` documents this resolved source-identity issue
and the version distinction. No current code, result or historical hash
has been replaced; historical training is not recreated.

## Verification and confidentiality

From this package root, run `python verify_results.py` after installing the
listed requirements in an appropriate environment. The data-free regression
suite is `python -m unittest discover -s tests -p "test_*.py" -v`.
These checks are narrower than a full raw-data replay.

The original approved public weights, code, configurations, de-identified
aggregates and figures remain public materials. Raw production records and
production-derived row trajectories remain confidential. Full replay requires
authorized access to the 26 original inputs in a separate private working copy.
The evidence is archived-data replay, not plant-side closed-loop validation.

Use the fixed commit of this correction when citing the corrected code. The
original commit above remains the historical source link and does not contain
these corrections. No original commit, result or historical receipt is rewritten.
