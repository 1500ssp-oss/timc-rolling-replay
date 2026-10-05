"""Data1-only pass-scale lock used by every publication replay.

The lock is a frozen catalogue of the six robust scales measured on the
thirteen Data1 passes.  A replay pass is assigned the nearest catalogue entry
using only its archived entry/exit gauge metadata in natural-log space.  No
Data2 response or quality value enters scale selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any


SCALE_FIELDS = (
    "tension_scale",
    "thickness_scale",
    "flatness_scale",
    "rollforce_scale",
    "radial_scale",
    "offcenter_scale",
)
LOCK_SCHEMA_VERSION = 1
LOCK_METHOD = "nearest Data1 recipe in Euclidean log(entry gauge), log(exit gauge) space; lexical pass_id tie-break"


def canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload_bytes(payload)).hexdigest()


def seal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed["payload_sha256"] = payload_sha256(sealed)
    return sealed


def load_scale_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != LOCK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported scale-lock schema: {payload.get('schema_version')}")
    if payload.get("method") != LOCK_METHOD:
        raise ValueError("Unexpected scale-lock matching method")
    expected = str(payload.get("payload_sha256", ""))
    observed = payload_sha256(payload)
    if expected != observed:
        raise ValueError(
            f"Scale-lock payload hash mismatch: expected {expected}, observed {observed}"
        )
    entries = payload.get("entries", [])
    if len(entries) != 13:
        raise ValueError(f"Scale lock must contain 13 Data1 passes, found {len(entries)}")
    ids = [str(item["pass_id"]) for item in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate pass_id in scale lock")
    for item in entries:
        if (
            not math.isfinite(float(item["entry_thickness_mm"]))
            or not math.isfinite(float(item["exit_thickness_mm"]))
            or float(item["entry_thickness_mm"]) <= 0
            or float(item["exit_thickness_mm"]) <= 0
        ):
            raise ValueError(f"Non-positive gauge in scale lock: {item['pass_id']}")
        for field in SCALE_FIELDS:
            if not math.isfinite(float(item[field])) or float(item[field]) <= 0:
                raise ValueError(f"Invalid {field} in scale lock: {item['pass_id']}")
    return payload


def select_entry(
    payload: dict[str, Any], entry_thickness: float, exit_thickness: float
) -> tuple[dict[str, Any], float]:
    entry = float(entry_thickness)
    exit_ = float(exit_thickness)
    if not math.isfinite(entry) or not math.isfinite(exit_) or entry <= 0 or exit_ <= 0:
        raise ValueError(f"Pass gauge must be positive, got {entry}->{exit_}")
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for item in payload["entries"]:
        distance = math.hypot(
            math.log(entry / float(item["entry_thickness_mm"])),
            math.log(exit_ / float(item["exit_thickness_mm"])),
        )
        ranked.append((float(distance), str(item["pass_id"]), item))
    distance, _, item = min(ranked, key=lambda row: (row[0], row[1]))
    return item, distance


def scale_override_and_provenance(
    payload: dict[str, Any], entry_thickness: float, exit_thickness: float
) -> tuple[dict[str, float], dict[str, Any]]:
    item, distance = select_entry(payload, entry_thickness, exit_thickness)
    override = {field: float(item[field]) for field in SCALE_FIELDS}
    provenance = {
        "scale_source": "Data1 gauge-nearest frozen catalogue",
        "scale_lock_id": str(payload["lock_id"]),
        "scale_lock_sha256": str(payload["payload_sha256"]),
        "scale_match_pass_id": str(item["pass_id"]),
        "scale_match_log_distance": float(distance),
    }
    return override, provenance


def apply_scale_lock(real_pass: Any, payload: dict[str, Any]) -> Any:
    override, provenance = scale_override_and_provenance(
        payload, real_pass.entry_thickness, real_pass.exit_thickness
    )
    return replace(real_pass, **override, **provenance)


def manifest_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale_lock_id": payload["lock_id"],
        "scale_lock_method": payload["method"],
        "scale_lock_payload_sha256": payload["payload_sha256"],
        "scale_lock_data1_passes": len(payload["entries"]),
    }
