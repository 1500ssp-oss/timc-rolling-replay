# Predictor training evidence and reproducibility boundary

The released `models/final_thickness_model.pt` is kept unchanged. Its payload
contains the GRU architecture state, feature normalization, target, window,
distance reference and a recorded training-loss scalar. It does **not** contain
the original training update count, seed, device, optimizer settings, data
membership hashes or training curves. The original per-fold/per-seed LOPO score
rows were not recovered. These unavailable facts are not inferred from script
defaults, nor replaced with newly generated historical records.

`results/predictor_lopo_one_se_summary.csv` retains the published 15 aggregate
rows. `config/model_selection.json` retains the frozen selection. The new
`results/predictor_training_provenance_audit.json` checks their aggregate
arithmetic and agreement with the checkpoint's architecture metadata; this is
not an independent reconstruction of historical training or proof against
underfitting. LOPO here means held-out passes, not held-out production batches.
Persistence MASE equals one by definition when its non-degenerate MAE is used
as the denominator. Neural seeds are averaged within each held-out pass before
the equal-pass mean and fold standard error are formed. The thickness GRU,
LSTM **and Transformer** are eligible at the published one-SE threshold;
eligibility does not establish a significant difference between them.

The master replay pipeline loads frozen weights and does not repeat historical
predictor fitting. Replaying the production archive therefore does not close
the missing historical training-provenance gap.

## Newly instrumented training (future execution, not historical recovery)

Both training scripts expose `--updates`: one unit is one optimizer update
using a sampled minibatch (with replacement), not a full-data epoch. The old
`--epochs` flag is accepted only as a documented backward-compatible alias.
The default budgets remain 60 updates per LOPO model/seed and 120 for a final
fit; these are **future script defaults**, not asserted budgets for the frozen
released model.

For future runs, checkpoint selection uses post-update MSE on a fixed,
training-only probe of up to 1024 samples. It never uses the held-out pass.
The selected state and the selection loss now refer to the same post-update
state; this differs from the old training-loss selection logic. It is not
validation early stopping. Each checkpoint/run records completed updates,
selected update, seed, sample counts, optimizer settings, feature normalization,
device/software version and available ordered source-file hashes. Per-update
loss histories are written separately.

Example commands, with authorized Data1 available:

```powershell
python code/model_lopo_data.py --out-root outputs/new_predictor_fit --updates 60
python code/predictor_training_audit.py --fold-scores outputs/new_predictor_fit/lopo_metrics.csv --out-root outputs/new_predictor_summary
python code/train_final_predictor.py --out outputs/new_predictor_fit/new_thickness_model.pt --updates 120
```

Set `ROLLING_DATA1_DIR` to the Data1 directory. These commands create a **new**
training experiment and do not authenticate or automatically replace the
frozen weights or published selection. The summary script rejects missing or
duplicate fold/seed rows and incompatible model/fold membership. Its declared
complexity order is Persistence, LinearARX, GRU, LSTM, Transformer.

The aggregate-only published audit can be rerun without production data:

```powershell
python code/predictor_training_audit.py
```

The audit exits successfully when the available aggregates agree, while its
JSON status still explicitly records incomplete historical provenance.
