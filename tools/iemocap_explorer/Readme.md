# IEMOCAP Explorer

Dataset exploration & evaluation toolkit for IEMOCAP, designed to inform model architecture decisions before training.

## Quick Start

```bash
cd /data/Brian/Echo_mind_system-with-Arousal-Valence-base/tools/iemocap_explorer
python main.py
```

## Options

```bash
python main.py --help
python main.py --no-audio          # Skip wav scanning (faster)
python main.py --no-plots          # Text report only
python main.py --verbose           # Debug logging
python main.py --iemocap-root /path/to/IEMOCAP_full_release
```

## Outputs

All saved to `./outputs/`:

| File | Description |
|------|-------------|
| `report.txt` | Full text statistics report |
| `iemocap_utterances.csv` | All parsed utterances as CSV (reusable for dataloader) |
| `01_raw_emotion_distribution.png` | All emotion categories bar chart |
| `02_four_class_distribution.png` | 4-class benchmark distribution |
| `03_six_class_per_session.png` | Per-session emotion balance (LOSO planning) |
| `04_audio_duration.png` | Audio length distribution + percentiles |
| `05_va_scatter.png` | Valence-Arousal 2D scatter (your V-A matrix target space) |
| `06_va_per_emotion.png` | V-A distribution per emotion class (separability) |
| `07_text_length.png` | Transcript word count distribution |
| `08_session_balance.png` | Utterances per session |
| `09_modality_completeness.png` | Audio + text availability check |

## Architecture

```
iemocap_explorer/
├── config.py        # Paths, label mappings, constants
├── parser.py        # Parse evaluation files, transcripts, audio metadata
├── statistics.py    # Compute all dataset statistics
├── visualizer.py    # Generate publication-quality plots
├── main.py          # Entry point
└── requirements.txt
```

## Design Notes

- **Parser output is reusable**: The `Utterance` dataclass and CSV export can directly feed your PyTorch Dataset class later
- **No heavy dependencies**: Uses `wave` module (stdlib) for audio metadata instead of librosa
- **Modular**: Import `parser.parse_iemocap()` anywhere in your project
