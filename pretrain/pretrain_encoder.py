"""
Pre-train RawAudioEncoder on LibriSpeech (mel-spectrogram reconstruction)
=========================================================================
Teaches the Conv1d stack to produce acoustically meaningful features
before fine-tuning on IEMOCAP's 12h of emotional speech.

Objective: Reconstruct log-mel spectrogram at the same frame rate as the
encoder's output (hop_length = frame_stride = 64 samples → 250 fps).
The encoder learns phonetics, prosody, and rhythm without emotion labels.

Dataset: LibriSpeech train-clean-100 (~100h, ~28k utterances, auto-download)
         Validation: LibriSpeech dev-clean (~5h, ~2.7k utterances)

Usage:
    # Download LibriSpeech and pre-train (first run downloads ~6GB)
    uv run pretrain/pretrain_encoder.py --data-dir ./data --epochs 20

    # Resume from a previous run
    uv run pretrain/pretrain_encoder.py --data-dir ./data --epochs 20 \\
        --resume pretrain/checkpoints/raw_audio_encoder_latest.pt

    # Use custom stride (must match configs/long_seq.yaml stride_config)
    uv run pretrain/pretrain_encoder.py --data-dir ./data --stride 32

Output:
    pretrain/checkpoints/raw_audio_encoder_best.pt    ← best val loss
    pretrain/checkpoints/raw_audio_encoder_latest.pt  ← latest epoch

Loading in main pipeline:
    Set configs/long_seq.yaml audio_encoder.checkpoint to the .pt path above.
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import soundfile as sf
import torchaudio
import torchaudio.transforms as T

# ── Backend fix ───────────────────────────────────────────────────────────────
# New torchaudio (>=2.1) defaults to torchcodec which requires system FFmpeg.
# Patch torchaudio.load to use soundfile (already a project dependency) so no
# system FFmpeg installation is needed.
def _load_with_soundfile(filepath, frame_offset=0, num_frames=-1, **kwargs):
    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    # soundfile returns (N, C); torchaudio convention is (C, N)
    t = torch.from_numpy(data).T.contiguous()
    if num_frames > 0:
        t = t[:, frame_offset: frame_offset + num_frames]
    elif frame_offset > 0:
        t = t[:, frame_offset:]
    return t, sr

torchaudio.load = _load_with_soundfile
# ─────────────────────────────────────────────────────────────────────────────

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.encoders_raw import RawAudioEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LibriSpeechDataset(Dataset):
    """Thin wrapper over torchaudio.datasets.LIBRISPEECH.

    Returns fixed-length audio chunks (or shorter at end of file).
    Chunks avoid padding overhead: 90%+ of batches are fully valid.

    Args:
        root: Directory to download/load LibriSpeech into
        url: Split to use ("train-clean-100", "dev-clean", etc.)
        download: Whether to download if not found
        target_sr: Resample all audio to this sample rate
        max_sec: Truncate utterances longer than this
    """

    def __init__(
        self,
        root: str,
        url: str = "train-clean-100",
        download: bool = True,
        target_sr: int = 16000,
        max_sec: float = 15.0,
    ):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=url, download=download
        )
        self.target_sr = target_sr
        self.max_samples = int(max_sec * target_sr)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        waveform, sr, *_ = self.dataset[idx]   # waveform: (C, N)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        waveform = waveform.squeeze(0)   # (N,)

        # Truncate
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]

        # Normalize amplitude
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        return waveform


def collate_fn(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic padding to longest waveform in batch."""
    lengths = [w.shape[0] for w in batch]
    max_len = max(lengths)
    B = len(batch)

    audio_padded = torch.zeros(B, max_len)
    attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (wav, length) in enumerate(zip(batch, lengths)):
        audio_padded[i, :length] = wav
        attention_mask[i, :length] = True

    return audio_padded, attention_mask


# ─────────────────────────────────────────────────────────────────────────────
# Mel decoder (pre-training head only — discarded after pre-training)
# ─────────────────────────────────────────────────────────────────────────────

class MelDecoder(nn.Module):
    """Linear projection from encoder features to log-mel targets.

    This head is ONLY used during pre-training and is discarded afterward.
    The encoder learns to produce features that can reconstruct mel spectrograms,
    capturing phonetics and prosody without needing emotion labels.
    """

    def __init__(self, input_dim: int = 768, n_mels: int = 128):
        super().__init__()
        self.proj = nn.Linear(input_dim, n_mels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, input_dim) → (B, T, n_mels)"""
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_mel_transform(frame_stride: int, sample_rate: int = 16000, n_mels: int = 80):
    """Build mel-spectrogram transform that matches encoder frame rate.

    hop_length = frame_stride ensures mel frames align exactly with
    encoder output frames, so MSE loss can be computed without resampling.

    n_fft=1024 → 513 frequency bins, safely supports 80 mel filters with no
    zero-weight filterbanks. 80 mel bins is the industry standard for speech
    (HuBERT, Wav2Vec2-BERT, Whisper all use 80 mel bins).
    """
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,           # 513 freq bins → no zero-weight mel filterbanks
        hop_length=frame_stride,
        n_mels=n_mels,        # 80: industry standard for speech
        f_min=60.0,           # above first FFT bin (16000/1024 ≈ 15.6 Hz) to avoid boundary
        f_max=sample_rate / 2,
        power=2.0,
    )


def compute_loss(
    encoder: RawAudioEncoder,
    decoder: MelDecoder,
    mel_transform: T.MelSpectrogram,
    audio: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Forward pass + mel reconstruction loss.

    Both encoder and mel transform produce T = N / frame_stride frames.
    Small misalignments (±1 frame from conv padding) are handled by slicing.
    """
    # Encoder: (B, N) → (B, T_enc, 768)
    _, features = encoder(audio, mask)             # (B, T_enc, output_dim)

    # Mel target: (B, N) → (B, n_mels, T_mel) → (B, T_mel, n_mels)
    with torch.no_grad():
        mel = mel_transform(audio)                 # (B, n_mels, T_mel)
        log_mel = (mel + 1e-6).log()               # log-mel is smoother target
        log_mel = log_mel.transpose(1, 2)          # (B, T_mel, n_mels)

    # Align: slice to min T to handle ±1 frame from conv padding
    T = min(features.shape[1], log_mel.shape[1])
    features = features[:, :T, :]
    log_mel = log_mel[:, :T, :]

    # Decoder: (B, T, 768) → (B, T, n_mels)
    predicted = decoder(features)

    # Frame-level mask for loss: only supervise valid (non-padded) frames
    frame_mask = encoder._compute_frame_mask(mask, T)    # (B, T)
    frame_mask = frame_mask[:, :T].unsqueeze(-1).float()  # (B, T, 1)

    # MSE loss over valid frames only
    loss = ((predicted - log_mel) ** 2 * frame_mask).sum() / frame_mask.sum().clamp(min=1)
    return loss


def run_epoch(
    encoder: RawAudioEncoder,
    decoder: MelDecoder,
    mel_transform: T.MelSpectrogram,
    loader: DataLoader,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    train: bool = True,
) -> float:
    encoder.train(train)
    decoder.train(train)
    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for audio, mask in loader:
            audio = audio.to(device)
            mask = mask.to(device)

            if train:
                optimizer.zero_grad()

            with autocast("cuda", enabled=device.type == "cuda"):
                loss = compute_loss(encoder, decoder, mel_transform, audio, mask, device)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-train RawAudioEncoder on LibriSpeech")
    parser.add_argument("--data-dir",  type=str, default="./data",
                        help="Root dir for LibriSpeech download/cache")
    parser.add_argument("--epochs",    type=int, default=20,
                        help="Number of pre-training epochs")
    parser.add_argument("--batch-size",type=int, default=32)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--stride",    type=int, default=64,
                        help="Total frame stride (must be 16, 32, or 64). "
                             "Maps to stride_config: 64→[4,4,4], 32→[4,4,2], 16→[4,2,2]")
    parser.add_argument("--n-mels",    type=int, default=80)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--resume",    type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--out-dir",   type=str,
                        default="pretrain/checkpoints",
                        help="Directory to save checkpoints")
    args = parser.parse_args()

    # ── stride_config from total stride ──────────────────────────────────
    stride_map = {
        64: [4, 4, 4],
        32: [4, 4, 2],
        16: [4, 2, 2],
    }
    if args.stride not in stride_map:
        raise ValueError(f"--stride must be one of {list(stride_map.keys())}, got {args.stride}")
    stride_config = stride_map[args.stride]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("RawAudioEncoder Pre-training on LibriSpeech")
    logger.info("=" * 60)
    logger.info(f"Device:        {device}")
    logger.info(f"Stride:        {args.stride}× → {stride_config}")
    logger.info(f"Frame rate:    {16000 // args.stride} fps @ 16kHz")
    logger.info(f"Epochs:        {args.epochs}")
    logger.info(f"Batch size:    {args.batch_size}")
    logger.info(f"LR:            {args.lr}")
    logger.info(f"Output dir:    {out_dir}")

    # ── Build encoder + decoder ───────────────────────────────────────────
    encoder = RawAudioEncoder(
        output_dim=768,
        stride_config=stride_config,
        channels=[64, 256, 512],
        dropout=0.1,
        freeze=False,
    ).to(device)

    decoder = MelDecoder(input_dim=768, n_mels=args.n_mels).to(device)
    mel_transform = build_mel_transform(args.stride, n_mels=args.n_mels).to(device)

    total_params = (
        sum(p.numel() for p in encoder.parameters()) +
        sum(p.numel() for p in decoder.parameters())
    )
    logger.info(f"Total params:  {total_params:,} (encoder + decoder)")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scaler = GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from epoch {ckpt['epoch']} (best_val_loss={best_val_loss:.4f})")

    # ── Datasets ──────────────────────────────────────────────────────────
    logger.info(f"\nLoading LibriSpeech from '{args.data_dir}' (downloading if needed)...")
    logger.info("train-clean-100 is ~6GB on first download. Please wait...\n")

    train_dataset = LibriSpeechDataset(args.data_dir, url="train-clean-100", download=True)
    val_dataset   = LibriSpeechDataset(args.data_dir, url="dev-clean",       download=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=False,
    )

    logger.info(f"Train: {len(train_dataset):,} utterances ({len(train_loader)} batches)")
    logger.info(f"Val:   {len(val_dataset):,} utterances ({len(val_loader)} batches)\n")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t_start = time.time()

        train_loss = run_epoch(
            encoder, decoder, mel_transform,
            train_loader, optimizer, scaler, device, train=True
        )
        val_loss = run_epoch(
            encoder, decoder, mel_transform,
            val_loader, optimizer, scaler, device, train=False
        )
        scheduler.step()

        t_elapsed = time.time() - t_start
        lr = optimizer.param_groups[0]["lr"]
        is_best = val_loss < best_val_loss

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {t_elapsed:.0f}s"
            + (" ★ best" if is_best else "")
        )

        # ── Save checkpoint ───────────────────────────────────────────────
        ckpt = {
            "epoch":                epoch,
            "encoder_state_dict":   encoder.state_dict(),
            "decoder_state_dict":   decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss":        min(val_loss, best_val_loss),
            "stride_config":        stride_config,
            "frame_stride":         encoder.frame_stride,
        }

        # Always save latest (for resuming)
        torch.save(ckpt, out_dir / "raw_audio_encoder_latest.pt")

        # Save best separately (for loading in main pipeline)
        if is_best:
            best_val_loss = val_loss
            torch.save(ckpt, out_dir / "raw_audio_encoder_best.pt")

    logger.info(
        f"\nPre-training complete. Best val loss: {best_val_loss:.4f}\n"
        f"Encoder weights: {out_dir / 'raw_audio_encoder_best.pt'}\n"
        f"\nNext step: set in configs/long_seq.yaml:\n"
        f"  audio_encoder:\n"
        f"    checkpoint: \"{out_dir / 'raw_audio_encoder_best.pt'}\"\n"
        f"    freeze: true   # freeze after pre-training for fair comparison"
    )


if __name__ == "__main__":
    main()
