"""Portable paths for the maintenance copy, without scientific dependencies.

Resources stored in package configuration are relative to PACKAGE_ROOT.
Explicit CLI paths and environment overrides are relative to the caller's
working directory. Profile values are relative to the profile file itself.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = PACKAGE_ROOT / "migration_paths.json"
PATH_ENV_NAMES = (
    "TIMC_PATHS_CONFIG", "ROLLING_DATA1_DIR", "ROLLING_DATA2_DIR",
    "TIMC_DATA1_SCALE_LOCK",
)


def external_path(value: str | os.PathLike[str], base_dir: Path | None = None) -> Path:
    """Resolve a user path once, before a child process changes directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() if base_dir is None else base_dir) / path
    return path.resolve()


def package_path(value: str | os.PathLike[str]) -> Path:
    """Resolve package-owned paths while retaining absolute user overrides."""
    return external_path(value, PACKAGE_ROOT)


def paths_profile(environ: Mapping[str, str] | None = None) -> tuple[Path, dict]:
    """Read an optional migration profile; an explicit missing profile is an error."""
    env = os.environ if environ is None else environ
    override = env.get("TIMC_PATHS_CONFIG", "").strip()
    path = external_path(override) if override else DEFAULT_PROFILE
    if not path.is_file():
        if override:
            raise FileNotFoundError(f"TIMC_PATHS_CONFIG does not exist: {path}")
        return path, {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("external", {}), dict):
        raise ValueError(f"Migration profile must contain an external object: {path}")
    return path, payload.get("external", {})


def configured_external_path(
    key: str,
    env_name: str | None = None,
    default: Path | None = None,
    *,
    required: bool = False,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Precedence: explicit environment, profile external field, package default."""
    env = os.environ if environ is None else environ
    override = env.get(env_name, "").strip() if env_name else ""
    if override:
        return external_path(override)
    profile, entries = paths_profile(env)
    value = entries.get(key)
    if value is not None and value != "":
        if not isinstance(value, str):
            raise ValueError(f"external.{key} must be a path string or null: {profile}")
        return external_path(value, profile.parent)
    if default is not None:
        return package_path(default)
    if required:
        detail = f"{env_name} or " if env_name else ""
        raise ValueError(f"Set {detail}external.{key} in the migration profile")
    return None


def data_directory(
    dataset: str, *, required: bool = False,
    environ: Mapping[str, str] | None = None,
) -> Path:
    if dataset not in {"Data1", "Data2"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    lower = dataset.lower()
    result = configured_external_path(
        f"{lower}_root", f"ROLLING_{dataset.upper()}_DIR",
        None if required else PACKAGE_ROOT / lower,
        required=required, environ=environ,
    )
    assert result is not None
    return result


def archive_directory(
    *, required: bool = False, environ: Mapping[str, str] | None = None,
) -> Path | None:
    return configured_external_path("archive_root", required=required, environ=environ)


def scale_lock_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    override = env.get("TIMC_DATA1_SCALE_LOCK", "").strip()
    return external_path(override) if override else package_path("config/data1_scale_lock.json")


def normalized_path_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy the environment with explicit path values made absolute for children."""
    result = dict(os.environ if environ is None else environ)
    for name in PATH_ENV_NAMES:
        value = result.get(name, "").strip()
        if value:
            result[name] = str(external_path(value))
    return result
