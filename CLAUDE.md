# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Project Overview

A privacy-centric multimodal emotion recognition system for psychological counseling support. Processes text + audio using a Mamba-Transformer hybrid to output an emotional probability matrix over the Valence-Arousal (V-A) space, which feeds downstream LLM-based empathetic response generation.

**Full system pipeline (research vision):**
`Text + Audio [+ ECG (Phase 2)] → Hybrid Encoder → V-A Probability Matrix → LLM Soft-Prompt → Empathetic Response`

**Current codebase implements Phase 1 only** — see Section 4.

## 2. AI Behavior Rules

- **No Yapping:** Output code immediately. Do not explain basic PyTorch or deep learning concepts unless asked.
- **Code Completeness:** Never use `# ... rest of code` or `pass` as placeholders. Provide the full copy-pasteable block.
- **Scientific Accuracy:** When writing signal processing (e.g. Butterworth filters) or attention mechanisms, verify matrix dimensions and tensor shapes explicitly before outputting.
- **Mixed Precision:** Always use `torch.amp.autocast` + `GradScaler` in training loops. Already implemented in `engine/trainer.py` — do not break this.
- **VRAM Awareness:** This targets a single RTX 4090 (24 GB). Flag any change that significantly increases activation memory (e.g. removing gradient checkpointing, large sequence lengths without chunking).

## 3. Commands

Package manager is `uv`. All execution uses `uv run`.

```bash
# Install dependencies
uv sync

# Verify Mamba v1 + CUDA environment before any model work
uv run python test.py

# Quick sanity check (single fold, 3 epochs)
uv run main.py --config configs/base.yaml --model mamba_fusion --fold 0 --epochs 3

# Full 5-fold LOSO (replace <model> with: late_fusion | cross_attention | mamba_fusion | mamba_dual_head)
uv run main.py --config configs/base.yaml --model <model>

# Ad-hoc overrides (no need to edit base.yaml)
uv run main.py --config configs/base.yaml --model mamba_fusion --epochs 50 --batch-size 16 --lr 2e-4 --fold 2

# Generate dataset CSV index — required before first training run
cd tools/iemocap_explorer && uv run python main.py
```

## 4. Current Implementation (Phase 1)

**What exists in this repo:**

```
Text + Audio → Frozen Encoders → Projection (768→256) → Temporal Modeling → Cross-Attention Fusion → Head(s)
```

**What does NOT exist yet (do not invent or call):**
- ECG pipeline, 1D-CNN projector, DREAMER dataset loader
- Llama 3.1 / DoRA / LoRA fine-tuning
- Physiology-Guided Attention (Audio/Text as Q/K, ECG as V anchor)
- Soft-prompt V-A injection into LLM
- Tensor parallelism / multi-GPU setup

### Data Pipeline (`data/`)
- **Tokenization happens in the collator, not the dataset.** `IEMOCAPDataset` returns raw waveforms + text strings. `MultimodalCollator` runs BERT tokenizer + Wav2Vec2 processor at batch time.
- `LOSOSplitter`: 5-fold Leave-One-Session-Out. Each fold = one IEMOCAP session held out (unique speaker pair → speaker-independent evaluation).
- **Label merging:** `hap` and `exc` are both mapped to class 1. There are 4 output classes: `ang=0, hap/exc=1, sad=2, neu=3`.
- **Prerequisite:** `tools/iemocap_explorer/outputs/iemocap_utterances.csv` must exist before training. Generate with `cd tools/iemocap_explorer && uv run python main.py`.

### Batch Dict Contract
Every model `forward(batch)` receives these exact keys from `MultimodalCollator`:
```python
batch = {
    "audio_input":          Tensor (B, num_samples),
    "audio_attention_mask": Tensor (B, num_samples),   # True = valid sample
    "text_input_ids":       Tensor (B, seq_len),
    "text_attention_mask":  Tensor (B, seq_len),       # True = valid token
    "labels":               Tensor (B,),               # int64, 0–3
    "valence":              Tensor (B,),               # float, mamba_dual_head only
    "arousal":              Tensor (B,),               # float, mamba_dual_head only
}
```
Returns `{"logits": (B, 4)}`. `mamba_dual_head` also returns `va_probs`, `va_targets`, `va_loss`.

### Mask Convention (critical)
Internal masks use **`True = valid`**. PyTorch's `src_key_padding_mask` uses **`True = IGNORE`**.
All models invert before passing to Transformer/attention: `pad_mask = ~valid_mask`.

### Model Registry (`models/__init__.py`)
`build_model()` inspects `__init__` signatures via `inspect.signature()` and auto-dispatches only the kwargs each model accepts. To add a new model: create the file, add one line to `MODEL_REGISTRY`. No if-statements or config changes needed.

| Registry key | Class | Architecture |
|---|---|---|
| `late_fusion` | `LateFusionBaseline` | Pool → concat → MLP. Baseline floor. |
| `cross_attention` | `CrossAttentionTransformer` | 6× Transformer + 2× cross-attn |
| `mamba_fusion` | `MambaFusion` | 6× BiMamba + 2× cross-attn. Best result (UA 72.08%). |
| `mamba_dual_head` | `MambaDualHead` | `mamba_fusion` + 9×9 V-A probability head |

### Mamba Implementation
- `BidirectionalMambaBlock`: two **separate** (not weight-shared) Mamba instances scan forward + backward. Merged via `Linear(2d→d)` + residual.
- Mamba parameter group gets **3× base LR** automatically via `model.get_param_groups()`. `Trainer` detects this method and uses it — implement it in any new Mamba model.

### V-A Dual Head (`mamba_dual_head.py`)
- `VAHead`: MLP → 81 logits → softmax → reshape `(B, 9, 9)`. Each cell = `P(V=v_i, A=a_j | input)`.
- `VASoftTargetGenerator`: converts ground-truth `(V, A)` point to 2D Gaussian on 9×9 grid (σ=0.5) for KL target.
- Loss: `(1 − 0.3) × CrossEntropy + 0.3 × KL-Divergence`.

### Training Engine (`engine/`)
- `Trainer` auto-detects `model.get_param_groups()` for differential LRs and `mamba_dual_head` by name for multi-task loss.
- Early stopping on **UA** (patience=7). Saves `best_model.pt` + `confusion_matrix.png` under `outputs/<model>/fold_<N>/`.
- Primary metric is **UA** (unweighted accuracy = mean per-class accuracy), not WA. Reports both.

## 5. Future Architecture Rules (for Phase 2 implementation)
When ECG and LLM components are added, follow these design decisions:
- **ECG projection:** 1D-CNN before any Mamba block. Raw ECG → CNN → Mamba.
- **Fusion:** Audio/Text as Query+Key, ECG as Value anchor ("Physiology-Guided Attention").
- **LLM interface:** Route the 9×9 V-A probability matrix as a soft-label prompt prefix — not hard-voted class labels.
- **Deterministic inference:** Eliminate Flash Attention non-determinism for medical-grade reproducibility.
