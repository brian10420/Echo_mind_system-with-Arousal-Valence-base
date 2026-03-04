"""
IEMOCAP Explorer - Parser
=========================
Parse IEMOCAP evaluation files, transcripts, and audio metadata.

Evaluation file format (per utterance block):
    [START_TIME - END_TIME] UTTERANCE_ID EMOTION [V, A, D]
    
    Example:
    [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
"""

import re
import wave
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import IEMOCAP_ROOT, SESSION_IDS, EMOTION_FULL_NAME

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Utterance:
    """Single IEMOCAP utterance with all available metadata."""
    utterance_id: str
    session: str           # e.g., "Session1"
    dialog_id: str         # e.g., "Ses01F_impro01"
    speaker: str           # "F" or "M"
    dialog_type: str       # "impro" or "script"
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Emotion labels
    emotion: str = "xxx"          # raw label (e.g., "ang", "hap")
    emotion_full: str = "Unknown" # full name (e.g., "Angry")
    
    # Dimensional values (from evaluators, scale 1-5)
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    
    # Text
    transcript: str = ""
    
    # Audio
    wav_path: Optional[str] = None
    audio_duration: float = 0.0   # in seconds
    audio_sr: int = 0             # sample rate
    
    # Evaluator details
    num_categorical_evaluators: int = 0
    evaluator_labels: list = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Duration from annotation timestamps."""
        return self.end_time - self.start_time


# ============================================================
# Evaluation File Parser
# ============================================================

def parse_evaluation_file(eval_path: Path, session: str) -> list[Utterance]:
    """
    Parse a single IEMOCAP evaluation file.
    
    Each utterance block starts with a summary line:
    [START - END] UTT_ID EMOTION [V, A, D]
    
    Followed by individual evaluator lines (C-E1, A-E1, etc.)
    """
    utterances = []
    
    # Pattern for the summary line
    # e.g., [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
    summary_pattern = re.compile(
        r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+'
        r'(Ses\w+)\s+'
        r'(\w+)\s+'
        r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]'
    )
    
    # Pattern for categorical evaluator lines
    # e.g., C-E1: neu;  or  C-E2: hap; fru;
    cat_eval_pattern = re.compile(r'C-E\d+:\s*(.+)')
    
    try:
        with open(eval_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except FileNotFoundError:
        logger.warning(f"Evaluation file not found: {eval_path}")
        return []
    
    # Split by blank lines to get utterance blocks
    lines = content.split('\n')
    
    current_utt = None
    eval_labels = []
    
    for line in lines:
        line = line.strip()
        if not line:
            # Save accumulated evaluator labels
            if current_utt is not None and eval_labels:
                current_utt.evaluator_labels = eval_labels
                current_utt.num_categorical_evaluators = len(eval_labels)
            eval_labels = []
            continue
            
        # Try matching summary line
        match = summary_pattern.search(line)
        if match:
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            utt_id = match.group(3)
            emotion = match.group(4)
            valence = float(match.group(5))
            arousal = float(match.group(6))
            dominance = float(match.group(7))
            
            # Extract metadata from utterance ID
            dialog_id = _extract_dialog_id(utt_id)
            speaker = _extract_speaker(utt_id)
            dialog_type = _extract_dialog_type(utt_id)
            
            utt = Utterance(
                utterance_id=utt_id,
                session=session,
                dialog_id=dialog_id,
                speaker=speaker,
                dialog_type=dialog_type,
                start_time=start_time,
                end_time=end_time,
                emotion=emotion,
                emotion_full=EMOTION_FULL_NAME.get(emotion, emotion),
                valence=valence,
                arousal=arousal,
                dominance=dominance,
            )
            utterances.append(utt)
            current_utt = utt
            continue
        
        # Try matching categorical evaluator line
        cat_match = cat_eval_pattern.search(line)
        if cat_match and current_utt is not None:
            raw_labels = cat_match.group(1).strip().rstrip(';')
            labels = [l.strip() for l in raw_labels.split(';') if l.strip()]
            eval_labels.append(labels)
    
    # Don't forget last block
    if current_utt is not None and eval_labels:
        current_utt.evaluator_labels = eval_labels
        current_utt.num_categorical_evaluators = len(eval_labels)
    
    return utterances


def _extract_dialog_id(utt_id: str) -> str:
    """Extract dialog ID from utterance ID. e.g., Ses01F_impro01_F000 -> Ses01F_impro01"""
    parts = utt_id.rsplit('_', 1)
    return parts[0] if len(parts) > 1 else utt_id


def _extract_speaker(utt_id: str) -> str:
    """Extract speaker from utterance ID. e.g., Ses01F_impro01_F000 -> F"""
    parts = utt_id.rsplit('_', 1)
    if len(parts) > 1 and len(parts[1]) > 0:
        return parts[1][0]  # First char: F or M
    return "Unknown"


def _extract_dialog_type(utt_id: str) -> str:
    """Extract dialog type. e.g., Ses01F_impro01 -> impro, Ses01F_script01 -> script"""
    if "impro" in utt_id:
        return "improvised"
    elif "script" in utt_id:
        return "scripted"
    return "unknown"


# ============================================================
# Transcript Parser
# ============================================================

def parse_transcripts(session_path: Path) -> dict[str, str]:
    """
    Parse all transcript files in a session.
    Returns dict: utterance_id -> transcript text
    """
    transcripts = {}
    trans_dir = session_path / "dialog" / "transcriptions"
    
    if not trans_dir.exists():
        logger.warning(f"Transcriptions dir not found: {trans_dir}")
        return transcripts
    
    for trans_file in sorted(trans_dir.glob("*.txt")):
        try:
            with open(trans_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Format: Ses01F_impro01_F000 [start-end]: text text text
                    match = re.match(r'(\S+)\s+\[[\d.]+-[\d.]+\]:\s*(.*)', line)
                    if match:
                        utt_id = match.group(1)
                        text = match.group(2).strip()
                        transcripts[utt_id] = text
        except Exception as e:
            logger.warning(f"Error parsing transcript {trans_file}: {e}")
    
    return transcripts


# ============================================================
# Audio Metadata Parser (fast — no full loading)
# ============================================================

def get_audio_metadata(wav_path: Path) -> tuple[float, int]:
    """
    Get audio duration and sample rate using wave module (fast, no librosa needed).
    Returns (duration_seconds, sample_rate)
    """
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / sr if sr > 0 else 0.0
            return duration, sr
    except Exception as e:
        logger.warning(f"Error reading audio {wav_path}: {e}")
        return 0.0, 0


def attach_audio_metadata(utterances: list[Utterance], session_path: Path) -> None:
    """
    Find wav files for each utterance and attach duration + sample rate.
    Modifies utterances in-place.
    """
    wav_dir = session_path / "sentences" / "wav"
    
    if not wav_dir.exists():
        logger.warning(f"Wav directory not found: {wav_dir}")
        return
    
    # Build a lookup: utterance_id -> wav_path
    wav_lookup = {}
    for wav_file in wav_dir.rglob("*.wav"):
        utt_id = wav_file.stem  # filename without extension
        wav_lookup[utt_id] = wav_file
    
    for utt in utterances:
        wav_path = wav_lookup.get(utt.utterance_id)
        if wav_path and wav_path.exists():
            utt.wav_path = str(wav_path)
            utt.audio_duration, utt.audio_sr = get_audio_metadata(wav_path)
        else:
            logger.debug(f"No wav found for {utt.utterance_id}")


# ============================================================
# Master Parser — Parse Entire IEMOCAP
# ============================================================

def parse_iemocap(
    root: Path = IEMOCAP_ROOT,
    include_audio: bool = True,
    include_transcripts: bool = True,
) -> list[Utterance]:
    """
    Parse the entire IEMOCAP dataset.
    
    Args:
        root: Path to IEMOCAP_full_release/
        include_audio: Whether to scan wav files for duration/sr
        include_transcripts: Whether to parse transcript files
    
    Returns:
        List of all Utterance objects across all 5 sessions
    """
    all_utterances = []
    
    for session in SESSION_IDS:
        session_path = root / session
        if not session_path.exists():
            logger.warning(f"Session directory not found: {session_path}")
            continue
        
        logger.info(f"Parsing {session}...")
        
        # 1. Parse evaluation files
        eval_dir = session_path / "dialog" / "EmoEvaluation"
        if not eval_dir.exists():
            # Try alternative naming
            eval_dir = session_path / "dialog" / "Evaluation"
            if not eval_dir.exists():
                # Search for any Evaluation* directory
                eval_candidates = list((session_path / "dialog").glob("Eval*"))
                if eval_candidates:
                    eval_dir = eval_candidates[0]
                else:
                    logger.warning(f"No evaluation dir found for {session}")
                    continue
        
        session_utterances = []
        for eval_file in sorted(eval_dir.glob("*.txt")):
            utts = parse_evaluation_file(eval_file, session)
            session_utterances.extend(utts)
        
        logger.info(f"  Found {len(session_utterances)} utterances from evaluations")
        
        # 2. Attach transcripts
        if include_transcripts:
            transcripts = parse_transcripts(session_path)
            for utt in session_utterances:
                utt.transcript = transcripts.get(utt.utterance_id, "")
            n_with_text = sum(1 for u in session_utterances if u.transcript)
            logger.info(f"  Matched {n_with_text} transcripts")
        
        # 3. Attach audio metadata
        if include_audio:
            attach_audio_metadata(session_utterances, session_path)
            n_with_audio = sum(1 for u in session_utterances if u.wav_path)
            logger.info(f"  Matched {n_with_audio} wav files")
        
        all_utterances.extend(session_utterances)
    
    logger.info(f"Total utterances parsed: {len(all_utterances)}")
    return all_utterances
