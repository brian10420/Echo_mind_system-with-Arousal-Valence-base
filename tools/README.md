# Tools

Utility scripts for dataset preparation, training, and result analysis.
All commands should be run from the **repo root**.

---

## run_baselines.py

Trains all models sequentially on a given config and prints a comparison table at the end.

```bash
# Train all models on long_seq config
uv run tools/run_baselines.py --config configs/long_seq.yaml

# Train on base config
uv run tools/run_baselines.py --config configs/base.yaml

# Quick smoke-test (single fold, 3 epochs)
uv run tools/run_baselines.py --config configs/long_seq.yaml --fold 0 --epochs 3

# Run a subset of models
uv run tools/run_baselines.py --config configs/long_seq.yaml --models mamba_fusion mamba_v2_fusion

# Re-parse existing results without retraining
uv run tools/run_baselines.py --config configs/long_seq.yaml --parse-only
```

**Available models:** `late_fusion`, `cross_attention`, `mamba_fusion`, `mamba_dual_head`, `mamba_v2_fusion`

**Output** — after all runs complete, prints a ranked table:

```
============================================================
  BENCHMARK SUMMARY
============================================================
Model                     UA      ±       WA      ±   F1-Mac      ±       Time
------------------------------------------------------------
mamba_v2_fusion       0.7312  0.0198   0.7143  0.0210   0.7201  0.0195    42.3 min
mamba_fusion          0.7208  0.0244   0.7046  0.0220   0.7086  0.0216    38.1 min
...
------------------------------------------------------------
  Best UA: mamba_v2_fusion  →  0.7312 ± 0.0198
============================================================
```

Results are also saved individually per model at `outputs/<output_dir>/<model>/loso_results.txt`.

---

## iemocap_explorer/

Dataset exploration and EDA toolkit for IEMOCAP.
**Must be run before first training** — generates `iemocap_utterances.csv` which the dataloader depends on.

```bash
cd tools/iemocap_explorer && uv run python main.py
```

See [`iemocap_explorer/Readme.md`](iemocap_explorer/Readme.md) for full documentation.
