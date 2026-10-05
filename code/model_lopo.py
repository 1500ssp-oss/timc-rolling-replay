from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from segmented_archived_timebase import derive_segmented_archived_timebase


ROOT = Path(__file__).resolve().parent
D0 = 0.81298828125
TARGETS = ["exit_tension_error", "exit_thickness_dev", "flatness_std"]
EXOG = ["speed_avg", "rollforce_actual", "entry_tension_error", "entry_thickness_dev", "reduction_ratio", "radialforce_mean", "offcenter_diff", "effective_dt_s", "distance_step_m"]
WINDOW = 8


def add_training_budget_argument(parser, default: int) -> None:
    """The legacy --epochs option counted updates, never full-data epochs."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--updates", dest="updates", type=int, default=default,
                       help="Number of sampled-minibatch optimizer updates (not full-data epochs).")
    group.add_argument("--epochs", dest="updates", type=int, default=argparse.SUPPRESS,
                       help="Deprecated backward-compatible alias for --updates; not full-data epochs.")


def frame_source_hashes(frames: list[pd.DataFrame]) -> list[str | None]:
    """Record source identities without distributing production file names."""
    data_dir = Path(os.environ.get("ROLLING_DATA1_DIR", "data1"))
    hashes = []
    for frame in frames:
        names = frame.get("pass_file")
        path = data_dir / str(names.iloc[0]) if names is not None and len(names) else None
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest()
                      if path is not None and path.is_file() else None)
    return hashes


def fit_sampled_updates(model, x, y, updates: int):
    """Select the post-update state on a fixed training-only loss probe.

    This is NOT validation early stopping. The held-out LOPO pass is never
    used for state selection. Metadata describes future executions only;
    the provenance of the pre-existing released checkpoint is not inferred.
    """
    if updates < 1 or len(x) < 1:
        raise ValueError("updates and training sample count must both be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    probe_index = torch.linspace(0, len(x)-1, min(1024, len(x)), device=x.device).long()
    best_loss = float("inf")
    best_state = None
    best_update = None
    history = []
    for update in range(1, updates + 1):
        model.train()
        index = torch.randint(0, len(x), (min(256, len(x)),), device=x.device)
        loss = nn.functional.mse_loss(model(x[index]), y[index])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probe_loss = float(nn.functional.mse_loss(model(x[probe_index]), y[probe_index]).item())
        if not np.isfinite(probe_loss):
            raise FloatingPointError("non-finite post-update training-probe loss")
        history.append({"update": update, "minibatch_loss_before_update": float(loss.detach().item()),
                        "fixed_training_probe_loss_after_update": probe_loss})
        if probe_loss < best_loss:
            best_loss = probe_loss
            best_update = update
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    metadata = {"training_budget_unit": "sampled_minibatch_optimizer_update", "updates_requested": updates,
                "updates_completed": updates, "minibatch_size": min(256, len(x)), "sampling_with_replacement": True,
                "learning_rate": 2e-3, "weight_decay": 1e-4, "optimizer": "AdamW",
                "checkpoint_selection": "minimum post-update MSE on a fixed training-only probe; not validation early stopping",
                "probe_samples": len(probe_index), "selected_update": best_update,
                "best_training_loss": best_loss, "torch_version": str(torch.__version__), "device": str(x.device)}
    return best_state, metadata, history


def load_frames() -> list[pd.DataFrame]:
    sys.path.insert(0, str(ROOT / "engine_core"))
    from src.real_process_data import load_real_process_data
    data_dir = Path(os.environ.get("ROLLING_DATA1_DIR", "data1"))
    frames = load_real_process_data(str(data_dir)).frames
    result = []
    for frame in frames:
        f = frame.copy()
        timebase = derive_segmented_archived_timebase(f, D0, 0.40, 3.00)
        f["effective_dt_s"] = timebase.dt_control_s
        f["distance_step_m"] = timebase.observed_dlength_m
        f["segment_id"] = timebase.segment_id
        result.append(f)
    return result


def sequences(frame: pd.DataFrame, target: str):
    cols = [target, *EXOG]
    x, y, naive = [], [], []
    # Forward fill only: past rows may never be completed from future rows.
    arr = frame[cols].apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0).to_numpy(float)
    seg = frame["segment_id"].to_numpy(int)
    for end in range(WINDOW, len(frame) - 1):
        if seg[end - WINDOW] != seg[end] or seg[end] != seg[end + 1]:
            continue
        x.append(arr[end - WINDOW + 1 : end + 1])
        y.append(arr[end + 1, 0])
        naive.append(arr[end, 0])
    return np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(naive, np.float32)


class SeqModel(nn.Module):
    def __init__(self, kind: str, features: int):
        super().__init__(); self.kind = kind
        if kind == "LSTM": self.net = nn.LSTM(features, 32, batch_first=True); self.head = nn.Linear(32, 1)
        elif kind == "GRU": self.net = nn.GRU(features, 32, batch_first=True); self.head = nn.Linear(32, 1)
        else:
            layer = nn.TransformerEncoderLayer(d_model=features, nhead=1, dim_feedforward=32, batch_first=True)
            self.net = nn.TransformerEncoder(layer, num_layers=2); self.head = nn.Linear(features, 1)
    def forward(self, x):
        z = self.net(x)[0] if self.kind in {"LSTM", "GRU"} else self.net(x)
        return self.head(z[:, -1]).squeeze(-1)


def metrics(y, pred, naive):
    mae = float(np.mean(np.abs(y-pred))); denom=max(float(np.mean(np.abs(y-naive))),1e-8)
    return {"MAE":mae,"RMSE":float(np.sqrt(np.mean((y-pred)**2))),"MASE":mae/denom}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--out-root",required=True); add_training_budget_argument(p,60); args=p.parse_args()
    if args.updates < 1: p.error("--updates must be positive")
    out=Path(args.out_root); out.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames=load_frames(); rows=[]; training_rows=[]; source_hashes=frame_source_hashes(frames)
    for ti,target in enumerate(TARGETS):
      packed=[sequences(f,target) for f in frames]
      for hold in range(len(frames)):
        train=[packed[i] for i in range(len(frames)) if i!=hold]; test=packed[hold]
        X=np.concatenate([z[0] for z in train]); y=np.concatenate([z[1] for z in train])
        mu=X.mean((0,1),keepdims=True); sd=X.std((0,1),keepdims=True)+1e-6
        Xt,yt,nt=test; Xn=(X-mu)/sd; Xtn=(Xt-mu)/sd
        # Deterministic linear ARX.
        # Deterministic linear ARX on the last-input feature vector; the
        # target column appears exactly once (no duplicated regressor).
        A=np.c_[np.ones(len(Xn)),Xn[:,-1]]; coef=np.linalg.lstsq(A,y,rcond=None)[0]
        pred=np.c_[np.ones(len(Xtn)),Xtn[:,-1]]@coef
        for model,pr in [("Persistence",nt),("LinearARX",pred)]: rows.append({"target":target,"fold":hold+1,"model":model,"seed":0,**metrics(yt,pr,nt),"n_train":len(X),"n_test":len(Xt),"device":"cpu"})
        for kind in ["LSTM","GRU","Transformer"]:
          for seed in [101,202,303]:
            torch.manual_seed(seed); model=SeqModel(kind,X.shape[-1]).to(device)
            tx=torch.tensor(Xn,device=device); ty=torch.tensor(y,device=device)
            best_state,metadata,history=fit_sampled_updates(model,tx,ty,args.updates)
            with torch.no_grad(): pr=model(torch.tensor(Xtn,device=device)).detach().cpu().numpy()
            ckpt=out/"checkpoints"/f"{target}_fold{hold+1}_{kind}_seed{seed}.pt"; ckpt.parent.mkdir(exist_ok=True)
            torch.save({"model":kind,"target":target,"fold":hold+1,"seed":seed,"state_dict":best_state,
                        "mu":mu.astype(np.float32),"sd":sd.astype(np.float32),"features":[target,*EXOG],"window":WINDOW,
                        "n_train":len(X),"n_test":len(Xt),"held_out_source_sha256":source_hashes[hold],
                        "training_source_sha256":[v for i,v in enumerate(source_hashes) if i!=hold],**metadata},ckpt)
            training_rows.extend({"target":target,"fold":hold+1,"model":kind,"seed":seed,**item} for item in history)
            rows.append({"target":target,"fold":hold+1,"model":model.kind,"seed":seed,**metrics(yt,pr,nt),"n_train":len(X),"n_test":len(Xt),"device":str(device),"checkpoint":str(ckpt.relative_to(out)),**metadata})
        print(target,hold+1,flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/"lopo_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(training_rows).to_csv(out/"training_update_history.csv",index=False,encoding="utf-8-sig")
    summary=df.groupby(["target","model"],as_index=False)[["MAE","RMSE","MASE"]].mean(); summary.to_csv(out/"model_summary.csv",index=False,encoding="utf-8-sig")
    (out/"model_run.json").write_text(json.dumps({"d0":D0,"window":WINDOW,"device":str(device),
        "training_budget_unit":"sampled_minibatch_optimizer_update","updates":args.updates,
        "legacy_epochs_is_alias_for_updates":True,"seeds":[101,202,303],"source_sha256_in_fold_order":source_hashes,
        "torch_version":str(torch.__version__),"checkpoint_selection":"post-update fixed training probe, never held-out pass",
        "scope":"this new training execution; does not establish provenance of pre-existing released weights"},indent=2),encoding="utf-8")

if __name__=='__main__': main()
