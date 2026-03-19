"""
RawAudioEncoder — Trainable Conv1d stack for raw waveform encoding
==================================================================
Replaces Wav2Vec2.0 when you need longer sequences entering Mamba.

Wav2Vec2.0 downsamples 320× (fixed) → T ≈ 550 for 11s   (50 fps)
RawAudioEncoder downsamples  64× (default) → T ≈ 2750 for 11s  (250 fps)

This gives ~5× longer sequences, putting Mamba's O(N) advantage
into effect and previewing the ECG pipeline's sequence lengths.

Architecture (default: strides=[4,4,4], channels=[64,256,512]):
    (B, N) → unsqueeze → (B, 1, N)
        Conv1d(1→64,   k=8, s=4) + GroupNorm + GELU      → (B, 64,  N/4)
        Conv1d(64→256, k=4, s=4) + GroupNorm + GELU      → (B, 256, N/16)
        Conv1d(256→512,k=4, s=4) + GroupNorm + GELU      → (B, 512, N/64)
    transpose → (B, T, 512)
        Linear(512 → 768)                                 → (B, T, 768)
    frame_stride = 4×4×4 = 64,  T ≈ N/64,  250 fps @ 16kHz

Interface is IDENTICAL to AudioEncoder:
    forward(audio_input, attention_mask) → (pooled, sequence)
    pooled:   (B, output_dim)
    sequence: (B, T, output_dim)

frame_stride attribute is read by all models' _compute_audio_frame_mask()
to correctly derive the valid-frame count from the sample-level mask.

Pre-training:
    See pretrain/pretrain_encoder.py — trains on LibriSpeech (100h)
    using mel-spectrogram reconstruction before fine-tuning on IEMOCAP.
    Load via checkpoint= argument or configs/long_seq.yaml.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RawAudioEncoder(nn.Module):
    """Trainable lightweight Conv1d audio feature extractor.

    Args:
        output_dim: Output feature dimension. Use 768 to match Wav2Vec2 interface
                    so downstream projection layers need no changes.
        stride_config: Stride per conv layer. frame_stride = product of all.
                       Default [4,4,4] → 64× → T≈2750 for 11s audio.
                       Use [4,4,2] → 32× for even longer sequences.
        channels: Channel widths per conv layer. Must match len(stride_config).
        dropout: Dropout rate between conv layers (0 = disabled).
        freeze: If True, freeze all weights after init (e.g. after pre-training).
        checkpoint: Path to pre-trained weights saved by pretrain_encoder.py.
                    If None or file not found, uses random initialization.
    """

    def __init__(
        self,
        output_dim: int = 768,
        stride_config: list[int] = None,
        channels: list[int] = None,
        dropout: float = 0.1,
        freeze: bool = False,
        checkpoint: str = None,
    ):
        super().__init__()

        if stride_config is None:
            stride_config = [4, 4, 4]
        if channels is None:
            channels = [64, 256, 512]

        if len(stride_config) != len(channels):
            raise ValueError(
                f"stride_config (len={len(stride_config)}) and "
                f"channels (len={len(channels)}) must have the same length."
            )

        self.output_dim = output_dim
        self.frame_stride = 1
        for s in stride_config:
            self.frame_stride *= s

        # ── Conv1d feature extractor ──────────────────────────────────────
        # First kernel=8 captures ~0.5ms of context at full sample rate.
        # Subsequent kernel=4 refines within the already-strided representation.
        kernel_sizes = [8] + [4] * (len(channels) - 1)

        conv_layers = []
        in_ch = 1
        for i, (out_ch, k, s) in enumerate(zip(channels, kernel_sizes, stride_config)):
            conv_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2)
            )
            # GroupNorm(32 groups): batch-size independent, stable for small batches.
            # Requires out_ch divisible by num_groups:
            #   64/32=2 ✓,  256/32=8 ✓,  512/32=16 ✓
            num_groups = min(out_ch, 32)
            conv_layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))
            conv_layers.append(nn.GELU())
            if dropout > 0 and i < len(channels) - 1:
                conv_layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.conv_stack = nn.Sequential(*conv_layers)

        # ── Final projection to output_dim ────────────────────────────────
        self.proj = nn.Linear(channels[-1], output_dim)

        trainable = sum(p.numel() for p in self.parameters())
        logger.info(
            f"RawAudioEncoder: strides={stride_config} → frame_stride={self.frame_stride}× | "
            f"channels={channels}→{output_dim}d | "
            f"{trainable:,} trainable params | "
            f"{16000 // self.frame_stride} fps @ 16kHz"
        )

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            logger.info("RawAudioEncoder: FROZEN")

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

    def forward(
        self,
        audio_input: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_input:    (B, num_samples) raw waveform, amplitude in [-1, 1]
            attention_mask: (B, num_samples) True = valid sample

        Returns:
            pooled:   (B, output_dim) masked mean-pooled representation
            sequence: (B, T, output_dim)  T ≈ num_samples / frame_stride
        """
        # (B, N) → (B, 1, N) for Conv1d
        x = audio_input.unsqueeze(1)

        # Conv stack: (B, 1, N) → (B, channels[-1], T)
        x = self.conv_stack(x)

        # (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)

        # (B, T, C) → (B, T, output_dim)
        sequence = self.proj(x)

        # Frame-level mask from sample-level mask
        frame_mask = self._compute_frame_mask(attention_mask, sequence.shape[1])

        # Masked mean pooling
        mask = frame_mask.unsqueeze(-1).float()    # (B, T, 1)
        pooled = (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return pooled, sequence

    def _compute_frame_mask(
        self, sample_mask: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """Derive frame-level validity mask from sample-level mask.

        Args:
            sample_mask: (B, num_samples) True = valid
            num_frames: Actual T from conv output (sequence.shape[1])

        Returns:
            (B, T) True = valid frame
        """
        valid_samples = sample_mask.sum(dim=1)                   # (B,)
        valid_frames = (valid_samples / self.frame_stride).long().clamp(min=1, max=num_frames)
        frame_indices = torch.arange(num_frames, device=sample_mask.device).unsqueeze(0)
        return frame_indices < valid_frames.unsqueeze(1)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load encoder weights saved by pretrain_encoder.py.

        Supports both full checkpoint dicts (with 'encoder_state_dict' key)
        and bare state_dicts.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(
                f"RawAudioEncoder: checkpoint not found at '{checkpoint_path}'. "
                f"Continuing with random initialization."
            )
            return

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if "encoder_state_dict" in ckpt:
            self.load_state_dict(ckpt["encoder_state_dict"])
            epoch = ckpt.get("epoch", "?")
            loss = ckpt.get("best_val_loss", "?")
            logger.info(
                f"RawAudioEncoder: loaded pre-trained weights from '{checkpoint_path}' "
                f"(pretrain epoch={epoch}, val_loss={loss:.4f})"
                if isinstance(loss, float)
                else f"RawAudioEncoder: loaded pre-trained weights from '{checkpoint_path}' "
                     f"(pretrain epoch={epoch})"
            )
        else:
            self.load_state_dict(ckpt)
            logger.info(f"RawAudioEncoder: loaded weights from '{checkpoint_path}'")

    def train(self, mode: bool = True):
        return super().train(mode)
