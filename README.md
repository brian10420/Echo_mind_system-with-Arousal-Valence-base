# Echo-Mind: Multimodal Emotion Recognition with Mamba-Transformer Hybrid

A privacy-centric multimodal emotion recognition system integrating text and audio modalities using a Mamba-Transformer hybrid architecture. Built for real-time psychological counseling support and mental health screening.

## Overview

This system implements a progressive model architecture for emotion recognition on the IEMOCAP dataset, comparing Transformer and Mamba (State Space Model) approaches for multimodal fusion. The core contribution is a **Bidirectional Mamba + Cross-Attention** hybrid that achieves superior accuracy with linear O(N) complexity.

### Key Results (IEMOCAP 4-class, 5-fold LOSO)

| Model | WA | UA | F1-Macro | Trainable Params |
|-------|----|----|----------|-----------------|
| Late Fusion Baseline | 67.15% | 68.41% | 67.46% | 461K |
| Cross-Attention Transformer | 68.81% | 70.46% | 69.28% | 13.2M |
| Mamba Unidirectional | 68.72% | 70.60% | 69.23% | ~10M |
| **Mamba Bidirectional** | **70.46%** | **72.08%** | **70.86%** | ~13M |
| Mamba Bidirectional + V-A Head | 70.09% | 71.48% | 70.43% | ~13.2M |

## Architecture

```
Audio → Wav2Vec2.0 (frozen) → proj 768→256 → 6× BiMamba → audio features ─┐
                                                                            ├→ 2× Cross-Attention → pool → MLP → emotion
Text  → BERT (frozen)       → proj 768→256 → 6× BiMamba → text features ──┘
```

The Bidirectional Mamba processes each modality's sequence in both forward and backward directions, then merges with a linear projection. Cross-attention handles inter-modal fusion (text ↔ audio interaction). An optional dual-head variant adds a 9×9 Valence-Arousal probability matrix output for downstream counseling applications.

## Project Structure

```
Echo_mind_system-with-Arousal-Valence-base/
├── configs/
│   └── base.yaml                    # All hyperparameters (mamba, va_head, training)
├── data/
│   ├── __init__.py                  # Clean imports
│   ├── dataset.py                   # IEMOCAP PyTorch Dataset (soundfile backend)
│   ├── collator.py                  # Dynamic padding & batch-level tokenization
│   └── splitter.py                  # LOSO 5-fold speaker-independent splits
├── models/
│   ├── __init__.py                  # Auto-dispatch registry (inspect-based)
│   ├── encoders.py                  # Frozen BERT + Wav2Vec2.0 wrappers
│   ├── baseline_late_fusion.py      # Model C: concat + MLP
│   ├── baseline_cross_attention.py  # Model E: 6L Transformer + 2L CrossAttn
│   ├── mamba_blocks.py              # MambaBlock + BidirectionalMambaBlock
│   ├── mamba_fusion.py              # Model G: 6L BiMamba + 2L CrossAttn
│   └── mamba_dual_head.py           # Model H: G + 9×9 V-A probability head
├── engine/
│   ├── __init__.py                  # Clean imports
│   ├── trainer.py                   # Training loop (multi-task loss, param group LR)
│   └── evaluator.py                 # WA, UA, F1, confusion matrix
├── tools/
│   └── iemocap_explorer/            # Dataset analysis & visualization toolkit
├── outputs/                         # Training outputs (per model/fold)
├── main.py                          # Entry point: LOSO orchestration
└── pyproject.toml                   # Dependencies (uv)
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on RTX 4090, 24GB VRAM)
- IEMOCAP dataset (request access from USC)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone <repo-url>
cd Echo_mind_system-with-Arousal-Valence-base

# Install dependencies
uv sync

# Add soundfile for audio loading (if not already installed)
uv add soundfile
```

### Dataset Preparation

1. Download IEMOCAP from [USC SAIL](https://sail.usc.edu/iemocap/)
2. Extract to `/data/Brian/Dataset/IEMOCAP_full_release/` (or update `configs/base.yaml`)
3. Run the explorer to generate the CSV index:

```bash
cd tools/iemocap_explorer
uv run python main.py
```

This creates `tools/iemocap_explorer/outputs/iemocap_utterances.csv` used by the training pipeline.

## Usage

### Quick Test (single fold, 3 epochs)

```bash
uv run main.py --config configs/base.yaml --model mamba_fusion --fold 0 --epochs 3
```

### Full 5-Fold LOSO Evaluation

```bash
# Late Fusion baseline
uv run main.py --config configs/base.yaml --model late_fusion

# Transformer baseline
uv run main.py --config configs/base.yaml --model cross_attention

# Mamba Bidirectional (best model)
uv run main.py --config configs/base.yaml --model mamba_fusion

# Mamba + V-A dual head
uv run main.py --config configs/base.yaml --model mamba_dual_head
```

### Command Line Overrides

```bash
uv run main.py --config configs/base.yaml \
    --model mamba_fusion \
    --epochs 50 \
    --batch-size 16 \
    --lr 2e-4 \
    --fold 2
```

## Model Details

### Late Fusion (Model C)

Simplest baseline. Each modality is independently encoded and mean-pooled to a single vector, then concatenated and classified with an MLP. No inter-modal interaction.

### Cross-Attention Transformer (Model E)

Each modality passes through 6 Transformer encoder layers (self-attention within modality), then 2 bidirectional cross-attention blocks allow text and audio to interact. Uses sinusoidal positional encoding and pre-norm architecture.

### Mamba Fusion (Model G)

Replaces TransformerEncoder with Bidirectional Mamba blocks. Each block runs two separate Mamba SSM instances (forward + backward scan) and merges with a linear projection. Cross-attention fusion is identical to Model E for fair comparison. Uses 3× learning rate for Mamba layers vs cross-attention layers.

### Mamba Dual Head (Model H)

Extends Model G with a second output head that predicts a 9×9 Valence-Arousal probability distribution using Gaussian soft targets (σ=0.5). Multi-task loss: (1-α)×CrossEntropy + α×KL-Divergence. The V-A probability matrix serves as input for downstream LLM-based empathetic response generation (Phase 2).

## Configuration

All hyperparameters are centralized in `configs/base.yaml`:

| Section | Key Parameters |
|---------|---------------|
| `dataset` | 4-class labels (ang, hap+exc, sad, neu), max 11s audio, 48 text tokens |
| `model` | hidden_dim=256, 8 attention heads, 6 layers per modality |
| `mamba` | d_state=16, d_conv=4, expand=2, bidirectional=true |
| `va_head` | 9×9 grid, σ=0.5 Gaussian, loss_weight=0.3 |
| `training` | batch=32, lr=1e-4, cosine schedule, early stopping patience=7 |

## Adding New Models

The registry uses `inspect.signature()` for automatic kwarg dispatch. To add a new model:

1. Create `models/my_model.py` with standard `__init__` signature
2. Add one line to `MODEL_REGISTRY` in `models/__init__.py`
3. Run with `--model my_model`

No if-statements or config changes needed. The registry automatically passes only the kwargs your model accepts.

## Evaluation Protocol

- **LOSO (Leave-One-Session-Out)**: 5-fold cross-validation, each fold holds out one IEMOCAP session (unique speaker pair). Ensures speaker-independent evaluation.
- **Primary metric**: Unweighted Accuracy (UA) — mean per-class accuracy, handles class imbalance.
- **Early stopping**: On UA with patience=7 epochs.
- **Outputs per fold**: Best model checkpoint, confusion matrix PNG, classification report.

## Hardware Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU VRAM | 12 GB | 24 GB (RTX 4090) |
| RAM | 16 GB | 64 GB |
| Storage | 10 GB (IEMOCAP + models) | — |

Training time per model (5-fold LOSO, RTX 4090):

| Model | Time per Epoch | Total (30 epochs × 5 folds) |
|-------|---------------|----------------------------|
| Late Fusion | ~17s | ~40 min |
| Transformer | ~21s | ~50 min |
| Mamba Bidirectional | ~20s | ~50 min |

## References

- Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752
- Dao, T. & Gu, A. (2024). "Transformers are SSMs." arXiv:2405.21060
- Busso, C. et al. (2008). "IEMOCAP: Interactive emotional dyadic motion capture database." JLRE.
- Yerragondu, N.R. et al. (2025). "Multifold Fusion Attention Variant for Emotion Recognition." CSASE.
- Li, X. et al. (2025). "Mamba-enhanced text-audio-video alignment network for emotion recognition." ADMA.

## License

See [LICENSE](LICENSE) for details.