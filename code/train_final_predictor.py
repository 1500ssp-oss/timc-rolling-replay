from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_lopo_data import load_lopo_frames
from model_lopo import D0, EXOG, WINDOW, SeqModel, sequences, add_training_budget_argument, frame_source_hashes, fit_sampled_updates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    add_training_budget_argument(parser, 120)
    parser.add_argument("--kind", choices=["LSTM", "GRU", "Transformer"], default="GRU")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    frames = load_lopo_frames()
    parts = [sequences(frame, "exit_thickness_dev") for frame in frames]
    x = np.concatenate([item[0] for item in parts])
    y = np.concatenate([item[1] for item in parts])
    mu = x.mean((0, 1), keepdims=True)
    sd = x.std((0, 1), keepdims=True) + 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260710)
    model = SeqModel(args.kind, x.shape[-1]).to(device)
    tx = torch.tensor((x - mu) / sd, device=device)
    ty = torch.tensor(y, device=device)
    state, metadata, history = fit_sampled_updates(model, tx, ty, args.updates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": args.kind,
            "target": "exit_thickness_dev",
            "state_dict": state,
            "mu": mu.astype(np.float32),
            "sd": sd.astype(np.float32),
            "features": ["exit_thickness_dev", *EXOG],
            "window": WINDOW,
            "d0": D0,
            "seed": 20260710,
            "n_train": len(x),
            "training_source_sha256": frame_source_hashes(frames),
            **metadata,
        },
        args.out,
    )
    sidecar = {"kind":args.kind,"target":"exit_thickness_dev","seed":20260710,"n_train":len(x),
               "training_source_sha256":frame_source_hashes(frames),**metadata,
               "scope":"this new execution; not retrospective provenance of the released frozen model"}
    args.out.with_suffix(".training.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(history).to_csv(args.out.with_suffix(".training.csv"), index=False, encoding="utf-8-sig")
    print(json.dumps({"out": str(args.out), "device": str(device), "samples": len(x), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
