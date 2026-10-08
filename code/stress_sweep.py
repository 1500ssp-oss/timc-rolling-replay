from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from timc_paths import PACKAGE_ROOT, normalized_path_environment, package_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--extension-script", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def run_one(
    root: Path,
    batch_root: Path,
    extension_script: Path,
    item: dict,
    index: int,
    repeats: int,
) -> tuple[int, str, Path]:
    scenario_id = str(item["id"])
    out_dir = root / "run" / "06_stress" / f"scenario_{index:02d}_{scenario_id}"
    log_dir = root / "logs" / "stress"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "batch_sweep.py"),
        "--out-root",
        str(out_dir),
        "--run-root",
        str(root / "run"),
        "--batch-root",
        str(batch_root),
        "--extension-script",
        str(extension_script),
        "--config-json",
        str(package_path(item["config"])),
        "--dataset",
        "Data2",
        "--repeats",
        str(repeats),
        "--no-step-log",
    ]
    completed = subprocess.run(
        command, cwd=str(PACKAGE_ROOT), env=normalized_path_environment(),
        capture_output=True, text=True, check=False,
    )
    (log_dir / f"{index:02d}_{scenario_id}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (log_dir / f"{index:02d}_{scenario_id}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    return completed.returncode, scenario_id, out_dir


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = root / "run" / "06_stress" / "configs" / "scenario_manifest.json"
    items = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    results: list[tuple[int, str, Path]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                run_one,
                root,
                args.batch_root.resolve(),
                args.extension_script.resolve(),
                item,
                index,
                args.repeats,
            ): (index, item["id"])
            for index, item in enumerate(items, start=1)
        }
        for future in as_completed(futures):
            index, scenario_id = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"[stress] {len(results)}/{len(items)} {index:02d} {scenario_id} rc={result[0]}",
                flush=True,
            )

    failures = [item for item in results if item[0] != 0]
    frames = []
    for _, scenario_id, out_dir in sorted(results, key=lambda item: item[1]):
        path = out_dir / "stage_allocation_ablation_metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "stress_scenario_id", scenario_id)
            frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(
            root / "run" / "06_stress" / "stress_metrics_current.csv",
            index=False,
            encoding="utf-8-sig",
        )
    summary = {
        "scenarios": len(items),
        "completed": len(results),
        "failed": [item[1] for item in failures],
        "repeats": args.repeats,
        "workers": args.workers,
        "timebase": "segment_gap_v2",
    }
    (root / "run" / "06_stress" / "current_stress_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
