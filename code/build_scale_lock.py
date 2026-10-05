"""Build or verify the frozen Data1-only pass-scale catalogue."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from scale_lock import LOCK_METHOD, LOCK_SCHEMA_VERSION, SCALE_FIELDS, seal_payload
from raw_archive import resolve_archive


DATA1_BATCHES = ("I", "IV", "VII")
EXPECTED_IDS = {
    "P01", "P02", "P03", "P04", "P12", "P13", "P14",
    "P21", "P22", "P23", "P24", "P25", "P26",
}


def load_suite(code_root: Path):
    engine = code_root / "engine_core"
    if str(engine) not in sys.path:
        sys.path.insert(0, str(engine))
    path = engine / "15_realdata_closed_loop_suite.py"
    spec = importlib.util.spec_from_file_location("scale_lock_suite", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pass_id(name: str) -> str:
    token = Path(name).stem.split("_", 1)[0].upper()
    if token not in EXPECTED_IDS:
        raise ValueError(f"Unexpected Data1 pass id in {name}: {token}")
    return token


def transition(name: str) -> tuple[float, float]:
    token = Path(name).stem.split("_", 2)[1]
    if "to" not in token:
        raise ValueError(f"Cannot parse transition from {name}")
    a, b = token.split("to", 1)
    return float(a), float(b)


def build_payload(batch_root: Path, code_root: Path) -> dict[str, Any]:
    suite = load_suite(code_root)
    archive_map, files_by_hash = resolve_archive(
        batch_root, code_root.parent / "data_map" / "production_batch_map.csv"
    )
    entries: list[dict[str, Any]] = []
    for batch_id in DATA1_BATCHES:
        batch_rows = archive_map.loc[archive_map["batch"].eq(batch_id)].copy()
        if set(batch_rows["source"].astype(str)) != {"Data1"}:
            raise AssertionError(f"Batch {batch_id} is not a Data1-only batch")
        with tempfile.TemporaryDirectory(prefix=f"timc_scale_{batch_id}_") as temp:
            staged = Path(temp)
            identity: dict[str, dict[str, Any]] = {}
            for _, map_row in batch_rows.sort_values("pass_id").iterrows():
                pid = str(map_row["pass_id"])
                entry_text, exit_text = str(map_row["transition"]).split("->")
                entry, exit_ = float(entry_text), float(exit_text)
                source = files_by_hash[str(map_row["sha256"])]
                staged_name = f"{entry:.3f}_to_{exit_:.3f}__{pid}.csv"
                shutil.copy2(source, staged / staged_name)
                identity[staged_name] = {
                    "pass_id": pid,
                    "source_sha256": str(map_row["sha256"]),
                    "rows": int(map_row["rows"]),
                }
            dataset = suite.load_real_process_data(str(staged))
            for df in dataset.frames:
                rp = suite.make_real_pass(df, allow_observed_scale_fit=True)
                staged_name = str(rp.pass_file)
                source_identity = identity[staged_name]
                pid = source_identity["pass_id"]
                if int(rp.rows) != source_identity["rows"]:
                    raise AssertionError(
                        f"{pid}: expected {source_identity['rows']} rows, loaded {rp.rows}"
                    )
                item: dict[str, Any] = {
                    "pass_id": pid,
                    "batch": batch_id,
                    "source_sha256": source_identity["source_sha256"],
                    "entry_thickness_mm": float(rp.entry_thickness),
                    "exit_thickness_mm": float(rp.exit_thickness),
                    "rows": int(rp.rows),
                }
                item.update({field: float(getattr(rp, field)) for field in SCALE_FIELDS})
                entries.append(item)
    entries.sort(key=lambda item: item["pass_id"])
    ids = {item["pass_id"] for item in entries}
    if ids != EXPECTED_IDS:
        raise AssertionError(f"Data1 scale-lock identity mismatch: {sorted(ids)}")
    payload = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "lock_id": "data1-gauge-nearest-v1",
        "method": LOCK_METHOD,
        "selection_inputs": ["entry_thickness_mm", "exit_thickness_mm"],
        "selection_uses_data2_response_values": False,
        "scale_formula": {
            "robust_base": "max(IQR/1.349, population standard deviation, floor)",
            "tension_scale": "robust_base(exit_tension_error, 0.025) * 3.0",
            "thickness_scale": "robust_base(exit_thickness_dev, 0.30) * 1.35",
            "flatness_scale": "robust_base(flatness_std, 0.15) * 1.50",
            "rollforce_scale": "robust_base(rollforce_actual, 25.0)",
            "radial_scale": "robust_base(radialforce_diff, 10.0)",
            "offcenter_scale": "robust_base(offcenter_diff, 0.02)",
        },
        "entries": entries,
    }
    return seal_payload(payload)


def normalized(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "data1_scale_lock.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parent
    candidate = build_payload(args.batch_root.resolve(), code_root)
    lock_path = args.lock_path.resolve()
    if args.write:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[scale-lock] wrote {lock_path}")
        print(f"[scale-lock] payload_sha256={candidate['payload_sha256']}")
        return 0
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    frozen = json.loads(lock_path.read_text(encoding="utf-8"))
    if normalized(candidate) != normalized(frozen):
        raise AssertionError(
            "Frozen Data1 scale lock differs from scales rebuilt from the supplied raw Data1 files"
        )
    print(f"[scale-lock] verified 13 Data1 passes; payload_sha256={candidate['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
