"""
Mamba V2 + Cross-Attention Hybrid (Model G-v2) — Bidirectional
===============================================================
Architecture identical to MambaFusion (Model G) but replaces Mamba v1
blocks with Mamba2 (SSD) blocks.

Audio -> Wav2Vec2.0 (frozen) -> proj 768->d -> 6x BiMamba2Block -> audio features
Text  -> BERT (frozen)       -> proj 768->d -> 6x BiMamba2Block -> text features
                                    |
                        2x Bidirectional Cross-Attention
                        (text <-> audio interact)
                                    |
                              masked mean pool
                                    |
                         concat -> MLP -> 4 classes

Differences vs MambaFusion (v1):
    1. Mamba2 kernel: multi-head SSM via SSD algorithm
    2. headdim parameter: nheads = (d_model * expand) // headdim
    3. d_state=128 default (vs 16 in v1) — richer SSM dynamics
    4. Uses mamba_v2_config dict from configs/base.yaml [mamba_v2] section

Learning rate groups are identical to v1:
    Mamba2 layers → 3x base_lr
    Cross-attention → base_lr
    Other (proj, classifier) → base_lr
"""

import logging

import torch
import torch.nn as nn

from models.encoders import TextEncoder, AudioEncoder
from models.mamba_v2_blocks import MambaV2TemporalEncoder
from models.baseline_cross_attention import CrossAttentionBlock

logger = logging.getLogger(__name__)


class MambaV2Fusion(nn.Module):
    """Mamba2 + Cross-Attention hybrid for multimodal emotion recognition.

    Args:
        text_encoder: Frozen BERT encoder
        audio_encoder: Frozen Wav2Vec2.0 encoder
        num_classes: Number of emotion classes
        hidden_dim: Model dimension (d_model)
        dropout: Dropout rate
        num_heads: Number of attention heads (cross-attention only)
        num_layers: Number of Mamba2 layers per modality
        num_cross_layers: Number of cross-attention blocks
        dim_feedforward: FFN hidden dim in cross-attention (default 4x hidden_dim)
        mamba_v2_config: Dict with Mamba2-specific params:
            d_state (int, default 128): SSM state dimension
            d_conv (int, default 4): Local convolution width
            expand (int, default 2): Inner expansion factor
            headdim (int, default 64): Per-head dimension
            bidirectional (bool, default True): Use forward + backward scan
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        audio_encoder: AudioEncoder,
        num_classes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_heads: int = 8,
        num_layers: int = 6,
        num_cross_layers: int = 2,
        dim_feedforward: int = None,
        mamba_v2_config: dict = None,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.hidden_dim = hidden_dim

        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4

        # Mamba2 config with v2 defaults (larger d_state, new headdim)
        if mamba_v2_config is None:
            mamba_v2_config = {}
        d_state = mamba_v2_config.get("d_state", 128)       # v2 default: 128 (vs 16 in v1)
        d_conv = mamba_v2_config.get("d_conv", 4)
        expand = mamba_v2_config.get("expand", 2)
        headdim = mamba_v2_config.get("headdim", 64)        # v2 new param
        bidirectional = mamba_v2_config.get("bidirectional", True)

        # ── Projection layers: encoder_dim (768) → hidden_dim (256) ──
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_encoder.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_encoder.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Per-modality Mamba2 temporal encoders ──
        self.audio_mamba = MambaV2TemporalEncoder(
            d_model=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.text_mamba = MambaV2TemporalEncoder(
            d_model=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # ── Cross-attention fusion (reused from baseline, unchanged) ──
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_cross_layers)
        ])

        # ── Classification head ──
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()
        self._log_architecture(num_layers, d_state, d_conv, expand, headdim, num_cross_layers, bidirectional)

    def _init_weights(self):
        """Initialize trainable weights (skip frozen encoders and Mamba2 internals)."""
        for name, module in self.named_modules():
            if name.startswith(('text_encoder.model', 'audio_encoder.model')):
                continue
            # Skip Mamba2 internal params (initialized by mamba-ssm library)
            if '.mamba2_fwd.' in name or '.mamba2_bwd.' in name or '.mamba2.' in name:
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    continue
            if isinstance(module, nn.Linear) and 'mamba2' not in name:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _log_architecture(self, num_layers, d_state, d_conv, expand, headdim, num_cross, bidirectional):
        inner_dim = self.hidden_dim * expand
        nheads = inner_dim // headdim
        mode = "Bidirectional" if bidirectional else "Unidirectional"
        audio_info = self.audio_mamba.get_param_count()
        text_info = self.text_mamba.get_param_count()

        logger.info(
            f"MambaV2Fusion: {mode} {num_layers}L × 2mod Mamba2 "
            f"(d_state={d_state}, d_conv={d_conv}, expand={expand}, "
            f"headdim={headdim}, nheads={nheads}) "
            f"+ {num_cross}L CrossAttn | "
            f"Audio: {audio_info['total']:,} ({audio_info['per_layer']:,}/layer) | "
            f"Text: {text_info['total']:,} ({text_info['per_layer']:,}/layer)"
        )

    def get_param_groups(self, base_lr: float = 1e-4) -> list:
        """Return parameter groups with different learning rates.

        Mamba2 layers: 3x base_lr  (SSM dynamics need stronger gradients)
        Cross-attention: base_lr   (sensitive, keep conservative)
        Other: base_lr
        """
        mamba_params = []
        cross_attn_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'mamba' in name and 'cross' not in name:
                mamba_params.append(param)
            elif 'cross_attention' in name:
                cross_attn_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if mamba_params:
            param_groups.append({"params": mamba_params, "lr": base_lr * 3.0, "name": "mamba2"})
        if cross_attn_params:
            param_groups.append({"params": cross_attn_params, "lr": base_lr, "name": "cross_attention"})
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr, "name": "other"})

        for g in param_groups:
            n = sum(p.numel() for p in g["params"])
            logger.info(f"  Param group '{g['name']}': {n:,} params, lr={g['lr']:.1e}")

        return param_groups

    def _compute_audio_frame_mask(self, sample_mask: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Compute frame-level mask from sample-level mask. Wav2Vec2 downsamples ~320x."""
        valid_samples = sample_mask.sum(dim=1)
        valid_frames = (valid_samples / self.audio_encoder.frame_stride).long().clamp(min=1, max=num_frames)
        frame_indices = torch.arange(num_frames, device=sample_mask.device).unsqueeze(0)
        return frame_indices < valid_frames.unsqueeze(1)

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: MultimodalCollator batch dict

        Returns:
            {"logits": (B, num_classes)}
        """
        # 1. Frozen encoder feature extraction
        _, text_seq = self.text_encoder(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
        )
        text_mask = batch["text_attention_mask"].bool()          # (B, S) True=valid

        _, audio_seq = self.audio_encoder(
            audio_input=batch["audio_input"],
            attention_mask=batch["audio_attention_mask"],
        )
        audio_mask = self._compute_audio_frame_mask(
            batch["audio_attention_mask"], audio_seq.shape[1]
        )                                                        # (B, T) True=valid

        # 2. Project: 768 → hidden_dim
        text_features = self.text_proj(text_seq)                # (B, S, hidden_dim)
        audio_features = self.audio_proj(audio_seq)             # (B, T, hidden_dim)

        # 3. Bidirectional Mamba2 temporal encoding
        text_features = self.text_mamba(text_features)
        audio_features = self.audio_mamba(audio_features)

        # 4. Cross-attention fusion (mask convention: True=IGNORE for PyTorch)
        text_pad_mask = ~text_mask
        audio_pad_mask = ~audio_mask

        for cross_block in self.cross_attention_blocks:
            text_features, audio_features = cross_block(
                text_features, audio_features,
                text_pad_mask, audio_pad_mask,
            )

        # 5. Masked mean pooling → concat → classify
        text_mask_exp = text_mask.unsqueeze(-1).float()
        text_pooled = (
            (text_features * text_mask_exp).sum(dim=1)
            / text_mask_exp.sum(dim=1).clamp(min=1e-9)
        )

        audio_mask_exp = audio_mask.unsqueeze(-1).float()
        audio_pooled = (
            (audio_features * audio_mask_exp).sum(dim=1)
            / audio_mask_exp.sum(dim=1).clamp(min=1e-9)
        )

        fused = torch.cat([text_pooled, audio_pooled], dim=-1)  # (B, 2×hidden_dim)
        logits = self.classifier(fused)                         # (B, num_classes)

        return {"logits": logits}

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
