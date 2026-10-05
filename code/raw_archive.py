"""Hash-locked discovery of authorized raw production files.

The released archive map deliberately omits original folder and file names.
Authorized users point the pipeline at a directory containing the raw CSV
files; each pass is then resolved solely by its published SHA-256 digest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_archive_map(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    required = {
        "pass_id",
        "batch",
        "source",
        "role",
        "transition",
        "rows",
        "sha256",
        "analysis_condition",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Archive map is missing columns: {missing}")
    if len(frame) != 26 or frame["pass_id"].nunique() != 26:
        raise ValueError("Archive map must contain exactly 26 unique passes")
    expected_ids = {f"P{index:02d}" for index in range(1, 27)}
    if set(frame["pass_id"]) != expected_ids:
        raise ValueError("Archive map pass IDs must be exactly P01-P26")
    hashes = frame["sha256"].str.lower()
    if hashes.duplicated().any() or not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Archive-map SHA-256 values must be unique 64-digit hex strings")
    frame = frame.copy()
    frame["sha256"] = hashes
    frame["rows"] = pd.to_numeric(frame["rows"], errors="raise").astype(int)
    return frame.sort_values("pass_id", kind="mergesort").reset_index(drop=True)


def locate_files_by_hash(batch_root: Path, archive_map: pd.DataFrame) -> dict[str, Path]:
    root = batch_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected = set(archive_map["sha256"].astype(str).str.lower())
    matches: dict[str, list[Path]] = {digest: [] for digest in expected}
    for path in sorted(root.rglob("*.csv"), key=lambda item: item.as_posix().lower()):
        digest = file_sha256(path)
        if digest in matches:
            matches[digest].append(path.resolve())
    missing = sorted(digest for digest, paths in matches.items() if not paths)
    duplicates = {
        digest: [path.as_posix() for path in paths]
        for digest, paths in matches.items()
        if len(paths) > 1
    }
    if missing or duplicates:
        raise RuntimeError(
            "Raw archive does not uniquely match the published SHA-256 map; "
            f"missing={missing[:5]}, duplicates={list(duplicates)[:5]}"
        )
    return {digest: paths[0] for digest, paths in matches.items()}


def resolve_archive(batch_root: Path, map_path: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    archive_map = load_archive_map(map_path)
    return archive_map, locate_files_by_hash(batch_root, archive_map)
