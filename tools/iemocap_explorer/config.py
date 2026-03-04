"""
IEMOCAP Explorer - Configuration
================================
Central config for paths, label mappings, and constants.
Update IEMOCAP_ROOT and PROJECT_ROOT to match your environment.
"""

from pathlib import Path

# ============================================================
# Path Configuration — UPDATE THESE FOR YOUR MACHINE
# ============================================================
IEMOCAP_ROOT = Path("/data/Brian/Dataset/IEMOCAP_full_release")
PROJECT_ROOT = Path("/data/Brian/Echo_mind_system-with-Arousal-Valence-base")
OUTPUT_DIR = PROJECT_ROOT / "tools" / "iemocap_explorer" / "outputs"

# ============================================================
# IEMOCAP Structure Constants
# ============================================================
NUM_SESSIONS = 5
SESSION_IDS = [f"Session{i}" for i in range(1, NUM_SESSIONS + 1)]

# ============================================================
# Emotion Label Mappings
# ============================================================

# All raw labels in IEMOCAP
ALL_EMOTIONS = [
    "ang", "hap", "sad", "neu", "fru", "exc",
    "fea", "sur", "dis", "oth", "xxx"
]

# Full name mapping
EMOTION_FULL_NAME = {
    "ang": "Angry",
    "hap": "Happy",
    "sad": "Sad",
    "neu": "Neutral",
    "fru": "Frustrated",
    "exc": "Excited",
    "fea": "Fearful",
    "sur": "Surprised",
    "dis": "Disgusted",
    "oth": "Other",
    "xxx": "No Agreement",
}

# Standard 4-class mapping (most common in literature)
# Happy + Excited merged → Happy
FOUR_CLASS_MAP = {
    "ang": "Angry",
    "hap": "Happy",
    "exc": "Happy",   # merged
    "sad": "Sad",
    "neu": "Neutral",
}

# Standard 6-class mapping
SIX_CLASS_MAP = {
    "ang": "Angry",
    "hap": "Happy",
    "exc": "Excited",
    "sad": "Sad",
    "neu": "Neutral",
    "fru": "Frustrated",
}

# ============================================================
# Visualization Settings
# ============================================================
PLOT_DPI = 150
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGSIZE_STANDARD = (12, 6)
FIGSIZE_LARGE = (16, 10)

# Color palette for emotions (consistent across all plots)
EMOTION_COLORS = {
    "Angry": "#E74C3C",
    "Happy": "#F39C12",
    "Sad": "#3498DB",
    "Neutral": "#95A5A6",
    "Frustrated": "#9B59B6",
    "Excited": "#E67E22",
    "Fearful": "#1ABC9C",
    "Surprised": "#2ECC71",
    "Disgusted": "#8B4513",
    "Other": "#BDC3C7",
    "No Agreement": "#7F8C8D",
}

# V-A quadrant colors
VA_QUADRANT_COLORS = {
    "High-A / Pos-V": "#E74C3C",   # Excited, Happy
    "High-A / Neg-V": "#F39C12",   # Angry, Fearful
    "Low-A / Pos-V":  "#2ECC71",   # Calm, Content
    "Low-A / Neg-V":  "#3498DB",   # Sad, Depressed
}
