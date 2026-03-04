"""
IEMOCAP Explorer - Statistics
=============================
Compute comprehensive dataset statistics for experiment planning.
"""

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FOUR_CLASS_MAP, SIX_CLASS_MAP, EMOTION_FULL_NAME
from parser import Utterance

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Container for all computed statistics."""
    total_utterances: int = 0
    total_with_audio: int = 0
    total_with_text: int = 0
    total_with_both: int = 0
    
    # Emotion distributions
    raw_emotion_counts: dict = None
    four_class_counts: dict = None
    six_class_counts: dict = None
    
    # Per-session breakdown
    session_counts: dict = None
    session_emotion_counts: dict = None
    
    # Audio stats
    audio_durations: np.ndarray = None
    audio_duration_stats: dict = None
    
    # V-A-D stats
    valence_values: np.ndarray = None
    arousal_values: np.ndarray = None
    dominance_values: np.ndarray = None
    vad_stats: dict = None
    
    # Dialog type breakdown
    dialog_type_counts: dict = None
    
    # Speaker breakdown
    speaker_counts: dict = None
    
    # Text length stats
    text_lengths: np.ndarray = None
    text_length_stats: dict = None
    
    # DataFrame for detailed analysis
    df: pd.DataFrame = None


def compute_statistics(utterances: list[Utterance]) -> DatasetStats:
    """Compute all statistics from parsed utterances."""
    stats = DatasetStats()
    
    if not utterances:
        logger.warning("No utterances to compute statistics for.")
        return stats
    
    # ============================================================
    # Build DataFrame
    # ============================================================
    records = []
    for u in utterances:
        records.append({
            "utterance_id": u.utterance_id,
            "session": u.session,
            "dialog_id": u.dialog_id,
            "speaker": u.speaker,
            "dialog_type": u.dialog_type,
            "emotion_raw": u.emotion,
            "emotion_full": u.emotion_full,
            "valence": u.valence,
            "arousal": u.arousal,
            "dominance": u.dominance,
            "transcript": u.transcript,
            "text_length": len(u.transcript.split()) if u.transcript else 0,
            "wav_path": u.wav_path,
            "audio_duration": u.audio_duration,
            "audio_sr": u.audio_sr,
            "has_audio": u.wav_path is not None,
            "has_text": len(u.transcript) > 0,
            "start_time": u.start_time,
            "end_time": u.end_time,
            "annotation_duration": u.duration,
        })
    
    df = pd.DataFrame(records)
    stats.df = df
    
    # ============================================================
    # Basic Counts
    # ============================================================
    stats.total_utterances = len(df)
    stats.total_with_audio = df["has_audio"].sum()
    stats.total_with_text = df["has_text"].sum()
    stats.total_with_both = (df["has_audio"] & df["has_text"]).sum()
    
    # ============================================================
    # Emotion Distributions
    # ============================================================
    stats.raw_emotion_counts = dict(df["emotion_full"].value_counts())
    
    # 4-class mapping
    df["emotion_4cls"] = df["emotion_raw"].map(FOUR_CLASS_MAP)
    df_4cls = df.dropna(subset=["emotion_4cls"])
    stats.four_class_counts = dict(df_4cls["emotion_4cls"].value_counts())
    
    # 6-class mapping
    df["emotion_6cls"] = df["emotion_raw"].map(SIX_CLASS_MAP)
    df_6cls = df.dropna(subset=["emotion_6cls"])
    stats.six_class_counts = dict(df_6cls["emotion_6cls"].value_counts())
    
    # ============================================================
    # Per-Session Breakdown
    # ============================================================
    stats.session_counts = dict(df["session"].value_counts().sort_index())
    
    session_emotion = {}
    for session in df["session"].unique():
        session_df = df[df["session"] == session]
        session_emotion[session] = dict(session_df["emotion_full"].value_counts())
    stats.session_emotion_counts = session_emotion
    
    # ============================================================
    # Audio Duration Stats
    # ============================================================
    audio_df = df[df["has_audio"] & (df["audio_duration"] > 0)]
    if len(audio_df) > 0:
        durations = audio_df["audio_duration"].values
        stats.audio_durations = durations
        stats.audio_duration_stats = {
            "count": len(durations),
            "mean": float(np.mean(durations)),
            "std": float(np.std(durations)),
            "min": float(np.min(durations)),
            "max": float(np.max(durations)),
            "median": float(np.median(durations)),
            "p25": float(np.percentile(durations, 25)),
            "p75": float(np.percentile(durations, 75)),
            "p95": float(np.percentile(durations, 95)),
            "total_hours": float(np.sum(durations) / 3600),
        }
    
    # ============================================================
    # V-A-D Stats
    # ============================================================
    # Filter out zero values (likely missing)
    vad_df = df[(df["valence"] > 0) & (df["arousal"] > 0)]
    if len(vad_df) > 0:
        stats.valence_values = vad_df["valence"].values
        stats.arousal_values = vad_df["arousal"].values
        stats.dominance_values = vad_df["dominance"].values
        
        stats.vad_stats = {
            "valence": _compute_array_stats(stats.valence_values),
            "arousal": _compute_array_stats(stats.arousal_values),
            "dominance": _compute_array_stats(stats.dominance_values),
            "v_a_correlation": float(np.corrcoef(stats.valence_values, stats.arousal_values)[0, 1]),
        }
    
    # ============================================================
    # Dialog Type & Speaker
    # ============================================================
    stats.dialog_type_counts = dict(df["dialog_type"].value_counts())
    stats.speaker_counts = dict(df["speaker"].value_counts())
    
    # ============================================================
    # Text Length Stats
    # ============================================================
    text_df = df[df["has_text"]]
    if len(text_df) > 0:
        text_lens = text_df["text_length"].values
        stats.text_lengths = text_lens
        stats.text_length_stats = {
            "count": len(text_lens),
            "mean": float(np.mean(text_lens)),
            "std": float(np.std(text_lens)),
            "min": float(np.min(text_lens)),
            "max": float(np.max(text_lens)),
            "median": float(np.median(text_lens)),
            "p95": float(np.percentile(text_lens, 95)),
        }
    
    return stats


def _compute_array_stats(arr: np.ndarray) -> dict:
    """Compute summary stats for a numpy array."""
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def print_report(stats: DatasetStats) -> str:
    """Generate and print a comprehensive text report."""
    lines = []
    sep = "=" * 70
    
    lines.append(sep)
    lines.append("IEMOCAP DATASET EXPLORATION REPORT")
    lines.append(sep)
    
    # --- Overview ---
    lines.append("\n1. OVERVIEW")
    lines.append(f"   Total utterances:         {stats.total_utterances}")
    lines.append(f"   With audio (wav):         {stats.total_with_audio}")
    lines.append(f"   With transcript:          {stats.total_with_text}")
    lines.append(f"   With both (audio+text):   {stats.total_with_both}")
    
    # --- Raw Emotion Distribution ---
    lines.append(f"\n2. RAW EMOTION DISTRIBUTION (all {stats.total_utterances} utterances)")
    if stats.raw_emotion_counts:
        for emo, count in sorted(stats.raw_emotion_counts.items(), key=lambda x: -x[1]):
            pct = count / stats.total_utterances * 100
            bar = "█" * int(pct / 2)
            lines.append(f"   {emo:15s}  {count:5d}  ({pct:5.1f}%)  {bar}")
    
    # --- 4-Class Distribution ---
    lines.append(f"\n3. STANDARD 4-CLASS DISTRIBUTION (for paper benchmarks)")
    if stats.four_class_counts:
        total_4cls = sum(stats.four_class_counts.values())
        for emo, count in sorted(stats.four_class_counts.items(), key=lambda x: -x[1]):
            pct = count / total_4cls * 100
            bar = "█" * int(pct / 2)
            lines.append(f"   {emo:15s}  {count:5d}  ({pct:5.1f}%)  {bar}")
        lines.append(f"   {'TOTAL':15s}  {total_4cls:5d}")
        
        # Imbalance ratio
        max_cls = max(stats.four_class_counts.values())
        min_cls = min(stats.four_class_counts.values())
        lines.append(f"   Imbalance ratio (max/min): {max_cls/min_cls:.2f}x")
    
    # --- 6-Class Distribution ---
    lines.append(f"\n4. STANDARD 6-CLASS DISTRIBUTION")
    if stats.six_class_counts:
        total_6cls = sum(stats.six_class_counts.values())
        for emo, count in sorted(stats.six_class_counts.items(), key=lambda x: -x[1]):
            pct = count / total_6cls * 100
            lines.append(f"   {emo:15s}  {count:5d}  ({pct:5.1f}%)")
    
    # --- Per-Session ---
    lines.append(f"\n5. PER-SESSION BREAKDOWN (for LOSO cross-validation)")
    if stats.session_counts:
        for session, count in sorted(stats.session_counts.items()):
            lines.append(f"   {session}: {count} utterances")
    
    # --- Audio Duration ---
    lines.append(f"\n6. AUDIO DURATION STATISTICS")
    if stats.audio_duration_stats:
        s = stats.audio_duration_stats
        lines.append(f"   Count:         {s['count']}")
        lines.append(f"   Total hours:   {s['total_hours']:.2f} h")
        lines.append(f"   Mean:          {s['mean']:.2f} s")
        lines.append(f"   Std:           {s['std']:.2f} s")
        lines.append(f"   Min:           {s['min']:.2f} s")
        lines.append(f"   Median:        {s['median']:.2f} s")
        lines.append(f"   P75:           {s['p75']:.2f} s")
        lines.append(f"   P95:           {s['p95']:.2f} s")
        lines.append(f"   Max:           {s['max']:.2f} s")
        lines.append(f"\n   → Design hint: Set max_audio_length to ~{s['p95']:.0f}s (P95) for training")
    
    # --- V-A-D ---
    lines.append(f"\n7. VALENCE-AROUSAL-DOMINANCE STATISTICS (scale: 1-5)")
    if stats.vad_stats:
        for dim in ["valence", "arousal", "dominance"]:
            d = stats.vad_stats[dim]
            lines.append(f"   {dim.capitalize():12s}  mean={d['mean']:.3f}  std={d['std']:.3f}  "
                        f"range=[{d['min']:.2f}, {d['max']:.2f}]")
        lines.append(f"   V-A correlation:  {stats.vad_stats['v_a_correlation']:.3f}")
        lines.append(f"\n   → Design hint: V-A are {'weakly' if abs(stats.vad_stats['v_a_correlation']) < 0.3 else 'moderately' if abs(stats.vad_stats['v_a_correlation']) < 0.6 else 'strongly'} correlated")
    
    # --- Dialog Type ---
    lines.append(f"\n8. DIALOG TYPE BREAKDOWN")
    if stats.dialog_type_counts:
        for dtype, count in stats.dialog_type_counts.items():
            pct = count / stats.total_utterances * 100
            lines.append(f"   {dtype:15s}  {count:5d}  ({pct:.1f}%)")
    
    # --- Speaker ---
    lines.append(f"\n9. SPEAKER GENDER BREAKDOWN")
    if stats.speaker_counts:
        for speaker, count in stats.speaker_counts.items():
            label = "Female" if speaker == "F" else "Male" if speaker == "M" else speaker
            lines.append(f"   {label:15s}  {count:5d}")
    
    # --- Text Length ---
    lines.append(f"\n10. TEXT LENGTH STATISTICS (word count)")
    if stats.text_length_stats:
        s = stats.text_length_stats
        lines.append(f"   Mean:    {s['mean']:.1f} words")
        lines.append(f"   Median:  {s['median']:.0f} words")
        lines.append(f"   P95:     {s['p95']:.0f} words")
        lines.append(f"   Max:     {s['max']:.0f} words")
        lines.append(f"\n   → Design hint: Set max_text_tokens to ~{int(s['p95'] * 1.5)} (P95 × 1.5 for subword expansion)")
    
    lines.append(f"\n{sep}")
    lines.append("END OF REPORT")
    lines.append(sep)
    
    report = "\n".join(lines)
    print(report)
    return report
