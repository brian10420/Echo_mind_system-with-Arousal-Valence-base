"""
IEMOCAP Explorer - Visualizer
=============================
Generate publication-quality plots for dataset analysis.
All plots are saved to OUTPUT_DIR.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    OUTPUT_DIR, PLOT_DPI, FIGSIZE_STANDARD, FIGSIZE_LARGE,
    EMOTION_COLORS, FOUR_CLASS_MAP, SIX_CLASS_MAP,
)
from statistics import DatasetStats

logger = logging.getLogger(__name__)


def setup_plot_style():
    """Configure matplotlib for clean, publication-ready plots."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def generate_all_plots(stats: DatasetStats, output_dir: Path = OUTPUT_DIR):
    """Generate all visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    
    logger.info(f"Generating plots to {output_dir}")
    
    plot_emotion_distribution(stats, output_dir)
    plot_four_class_distribution(stats, output_dir)
    plot_six_class_per_session(stats, output_dir)
    plot_audio_duration_histogram(stats, output_dir)
    plot_va_scatter(stats, output_dir)
    plot_va_per_emotion(stats, output_dir)
    plot_text_length_histogram(stats, output_dir)
    plot_session_balance(stats, output_dir)
    plot_modality_completeness(stats, output_dir)
    
    logger.info(f"All plots saved to {output_dir}")


# ============================================================
# Individual Plot Functions
# ============================================================

def plot_emotion_distribution(stats: DatasetStats, output_dir: Path):
    """Bar chart of all raw emotion categories."""
    if not stats.raw_emotion_counts:
        return
    
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    # Sort by count
    sorted_items = sorted(stats.raw_emotion_counts.items(), key=lambda x: -x[1])
    labels = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    colors = [EMOTION_COLORS.get(label, "#BDC3C7") for label in labels]
    
    bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 20,
                str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_title("IEMOCAP Raw Emotion Distribution (All Categories)")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Number of Utterances")
    ax.set_ylim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    fig.savefig(output_dir / "01_raw_emotion_distribution.png", dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: 01_raw_emotion_distribution.png")


def plot_four_class_distribution(stats: DatasetStats, output_dir: Path):
    """4-class distribution with imbalance visualization."""
    if not stats.four_class_counts:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_STANDARD)
    
    labels = list(stats.four_class_counts.keys())
    counts = list(stats.four_class_counts.values())
    colors = [EMOTION_COLORS.get(l, "#BDC3C7") for l in labels]
    
    # Bar chart
    bars = ax1.bar(labels, counts, color=colors, edgecolor='white')
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_title("4-Class Distribution (Paper Standard)")
    ax1.set_ylabel("Count")
    ax1.set_ylim(0, max(counts) * 1.15)
    
    # Pie chart
    ax2.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax2.set_title("Class Proportions")
    
    plt.suptitle("IEMOCAP 4-Class Benchmark Setting (Happy+Excited merged)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "02_four_class_distribution.png", dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: 02_four_class_distribution.png")


def plot_six_class_per_session(stats: DatasetStats, output_dir: Path):
    """Stacked bar chart: 6-class emotion per session (for LOSO planning)."""
    if stats.df is None:
        return
    
    df = stats.df.dropna(subset=["emotion_6cls"])
    if len(df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    sessions = sorted(df["session"].unique())
    emotions_6 = ["Angry", "Happy", "Excited", "Sad", "Neutral", "Frustrated"]
    
    x = np.arange(len(sessions))
    width = 0.12
    
    for i, emo in enumerate(emotions_6):
        counts = [len(df[(df["session"] == s) & (df["emotion_6cls"] == emo)]) for s in sessions]
        color = EMOTION_COLORS.get(emo, "#BDC3C7")
        ax.bar(x + i * width, counts, width, label=emo, color=color, edgecolor='white')
    
    ax.set_xlabel("Session")
    ax.set_ylabel("Count")
    ax.set_title("6-Class Emotion Distribution per Session (LOSO Planning)")
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(sessions)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    fig.savefig(output_dir / "03_six_class_per_session.png", dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: 03_six_class_per_session.png")


def plot_audio_duration_histogram(stats: DatasetStats, output_dir: Path):
    """Audio duration distribution with key percentile markers."""
    if stats.audio_durations is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_STANDARD)
    
    durations = stats.audio_durations
    s = stats.audio_duration_stats
    
    # Histogram
    ax1.hist(durations, bins=80, color='#3498DB', alpha=0.7, edgecolor='white')
    ax1.axvline(s['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {s['mean']:.1f}s")
    ax1.axvline(s['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {s['median']:.1f}s")
    ax1.axvline(s['p95'], color='green', linestyle='--', linewidth=2, label=f"P95: {s['p95']:.1f}s")
    ax1.set_xlabel("Duration (seconds)")
    ax1.set_ylabel("Count")
    ax1.set_title("Audio Duration Distribution")
    ax1.legend()
    
    # Box plot per session
    if stats.df is not None:
        df_audio = stats.df[stats.df["has_audio"] & (stats.df["audio_duration"] > 0)]
        sessions = sorted(df_audio["session"].unique())
        session_data = [df_audio[df_audio["session"] == s]["audio_duration"].values for s in sessions]
        
        bp = ax2.boxplot(session_data, labels=sessions, patch_artist=True)
        colors_box = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB', '#9B59B6']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.set_xlabel("Session")
        ax2.set_ylabel("Duration (seconds)")
        ax2.set_title("Duration per Session")
    
    plt.suptitle(f"Audio Duration Analysis (Total: {s['total_hours']:.1f} hours)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "04_audio_duration.png", dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: 04_audio_duration.png")


def plot_va_scatter(stats: DatasetStats, output_dir: Path):
    """Valence-Arousal 2D scatter plot — critical for your V-A probability matrix design."""
    if stats.valence_values is None or stats.arousal_values is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color by raw emotion if available
    if stats.df is not None:
        df_va = stats.df[(stats.df["valence"] > 0) & (stats.df["arousal"] > 0)]
        
        for emo in df_va["emotion_full"].unique():
            mask = df_va["emotion_full"] == emo
            color = EMOTION_COLORS.get(emo, "#BDC3C7")
            ax.scatter(
                df_va[mask]["valence"], df_va[mask]["arousal"],
                c=color, label=emo, alpha=0.4, s=15, edgecolors='none'
            )
    else:
        ax.scatter(stats.valence_values, stats.arousal_values, 
                   alpha=0.3, s=10, c='#3498DB')
    
    # Draw quadrant lines at midpoint (2.5 on 1-5 scale)
    ax.axhline(y=2.5, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(x=2.5, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Quadrant labels
    ax.text(1.2, 4.7, "High Arousal\nNeg Valence\n(Angry, Fearful)", fontsize=9, alpha=0.6)
    ax.text(3.5, 4.7, "High Arousal\nPos Valence\n(Excited, Happy)", fontsize=9, alpha=0.6)
    ax.text(1.2, 1.2, "Low Arousal\nNeg Valence\n(Sad, Depressed)", fontsize=9, alpha=0.6)
    ax.text(3.5, 1.2, "Low Arousal\nPos Valence\n(Calm, Content)", fontsize=9, alpha=0.6)
    
    ax.set_xlabel("Valence (Negative ← → Positive)")
    ax.set_ylabel("Arousal (Calm ← → Excited)")
    ax.set_title("IEMOCAP Valence-Arousal Distribution\n(Your soft-label probability matrix target space)")
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    fig.savefig(output_dir / "05_va_scatter.png", dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: 05_va_scatter.png")


def plot_va_per_emotion(stats: DatasetStats, output_dir: Path):
    """V-A distribution per emotion class — shows overlap and separability."""
    if stats.df is None:
        return
    
    df = stats.df[(stats.df["valence"] > 0) & (stats.df["arousal"] > 0)]
    df_4cls = df.dropna(subset=["emotion_4cls"])
    
    if len(df_4cls) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    emotions_4 = ["Angry", "Happy", "Sad", "Neutral"]
    
    for ax, emo in zip(axes.flat, emotions_4):
        mask = df_4cls["emotion_4cls"] == emo
        color = EMOTION_COLORS.get(emo, "#BDC3C7")
        
        # All data in gray
        ax.scatter(df_4cls["valence"], df_4cls["arousal"], 
                   c='#CCCCCC', alpha=0.15, s=8, label="Other")
        # Target emotion highlighted
        ax.scatter(df_4cls[mask]["valence"], df_4cls[mask]["arousal"],
                   c=color, alpha=0.5, s=15, label=emo)
        
        ax.axhline(y=2.5, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=2.5, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Show mean point
        v_mean = df_4cls[mask]["valence"].mean()
        a_mean = df_4cls[mask]["arousal"].mean()
        ax.scatter([v_mean], [a_mean], c='black', marker='X', s=100, zorder=5)
        ax.annotate(f"μ=({v_mean:.2f}, {a_mean:.2f})", (v_mean, a_mean),
                    xytext=(10, 10), textcoords='offset points', fontsize=9)
        
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 5.2)
        ax.set_title(f"{emo} (n={mask.sum()})")
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle("V-A Distribution per 4-Class Emotion\n(Black X = centroid, shows class separability in V-A space)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "06_va_per_emotion.png", dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: 06_va_per_emotion.png")


def plot_text_length_histogram(stats: DatasetStats, output_dir: Path):
    """Text length distribution — for max_tokens planning."""
    if stats.text_lengths is None:
        return
    
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    ax.hist(stats.text_lengths, bins=60, color='#9B59B6', alpha=0.7, edgecolor='white')
    
    s = stats.text_length_stats
    ax.axvline(s['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {s['mean']:.0f}")
    ax.axvline(s['p95'], color='green', linestyle='--', linewidth=2, label=f"P95: {s['p95']:.0f}")
    
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Number of Utterances")
    ax.set_title("Transcript Length Distribution")
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / "07_text_length.png", dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: 07_text_length.png")


def plot_session_balance(stats: DatasetStats, output_dir: Path):
    """Session balance — important for LOSO cross-validation."""
    if not stats.session_counts:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sessions = sorted(stats.session_counts.keys())
    counts = [stats.session_counts[s] for s in sessions]
    colors = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB', '#9B59B6']
    
    bars = ax.bar(sessions, counts, color=colors, edgecolor='white')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    mean_count = np.mean(counts)
    ax.axhline(y=mean_count, color='black', linestyle='--', alpha=0.5, 
               label=f"Mean: {mean_count:.0f}")
    
    ax.set_xlabel("Session")
    ax.set_ylabel("Number of Utterances")
    ax.set_title("Utterances per Session (LOSO Fold Size)")
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / "08_session_balance.png", dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: 08_session_balance.png")


def plot_modality_completeness(stats: DatasetStats, output_dir: Path):
    """Venn-style bar showing modality availability."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ["Total\nUtterances", "Has Audio\n(.wav)", "Has Text\n(transcript)", "Has Both\n(audio+text)"]
    values = [
        stats.total_utterances,
        stats.total_with_audio,
        stats.total_with_text,
        stats.total_with_both,
    ]
    colors = ['#95A5A6', '#3498DB', '#9B59B6', '#2ECC71']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='white')
    for bar, val in zip(bars, values):
        pct = val / stats.total_utterances * 100 if stats.total_utterances > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 20,
                f"{val}\n({pct:.0f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel("Count")
    ax.set_title("Modality Completeness Check\n(Ensures all samples have required inputs for training)")
    ax.set_ylim(0, max(values) * 1.2)
    
    plt.tight_layout()
    fig.savefig(output_dir / "09_modality_completeness.png", dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: 09_modality_completeness.png")
