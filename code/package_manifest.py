"""Enumerate distributable package files without reading local/private inputs."""
from __future__ import annotations

import os
from pathlib import Path


EXCLUDED_DIRECTORIES = frozenset({
    ".git", "outputs", "__pycache__", ".venv", "venv", ".pytest_cache",
    "private_data", "raw_data", "batch_archive", "production_batch_root",
    "data1", "data2",
})
EXCLUDED_SUFFIXES = frozenset({".log", ".gz", ".pyc", ".pyo"})


def package_files(root: Path) -> list[Path]:
    """Return public files, pruning confidential and environment directories.

    The manifest cannot hash itself. Local path profiles are machine-specific;
    the public example profile remains included. Inputs stored under any other
    name should be kept outside this package, as documented in MIGRATION.md.
    """
    root = Path(root).resolve()
    files: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        subdirectories[:] = sorted(
            name for name in subdirectories if name not in EXCLUDED_DIRECTORIES
        )
        for name in sorted(filenames):
            path = Path(directory) / name
            local_profile = name == "migration_paths.json" or (
                name.startswith("migration_paths.local") and path.suffix == ".json"
            )
            if name == "MANIFEST_SHA256.csv" or local_profile:
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            if path.is_file() and not path.is_symlink():
                files.append(path)
    return sorted(files)
