"""
Mamba V2 Dual-Head Model — Classification + V-A Probability Matrix
===================================================================
Architecture identical to MambaDualHead (Model H) but replaces Mamba v1
blocks with Mamba2 (SSD) blocks.

Audio -> Wav2Vec2.0 (frozen) -> proj 768->d -> 6x BiMamba2Block -> audio features
Text  -> BERT (frozen)       -> proj 768->d -> 6x BiMamba2Block -> text features
                                    |
                        2x Bidirectional Cross-Attention
                                    |
                              masked mean pool
                                    |
                         concat (B, 2×hidden_dim)
                        /                       \\
              Head 1: Classifier          Head 2: V-A Head
              MLP → 4 classes             MLP → 9×9 grid
              CrossEntropy                KL-Divergence
                        \\                       /
                Total Loss = (1-α)×CE + α×KL

Differences vs MambaDualHead (v1):
    1. Uses MambaV2TemporalEncoder (SSD algorithm, parallel scan)
    2. mamba_v2_config: d_state=128, headdim=64 (vs d_state=16 in v1)
    3. ~2.7× faster training at T≈2750 vs cross_attention baseline
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import TextEncoder, AudioEncoder
from models.mamba_v2_blocks import MambaV2TemporalEncoder
from models.baseline_cross_attention import CrossAttentionBlock
from models.mamba_dual_head import VAHead, VASoftTargetGenerator

logger = logging.getLogger(__name__)


class MambaV2DualHead(nn.Module):
    """Mamba2 + Cross-Attention with dual output heads.

    Head 1: 4-class emotion classification (CrossEntropy)
    Head 2: 9×9 V-A probability matrix (KL-Divergence)

    Args:
        text_encoder: Frozen BERT encoder
        audio_encoder: Frozen Wav2Vec2.0 or RawAudioEncoder
        num_classes: Number of emotion classes
        hidden_dim: Model dimension (d_model)
        dropout: Dropout rate
        num_heads: Number of attention heads (cross-attention only)
        num_layers: Number of Mamba2 layers per modality
        num_cross_layers: Number of cross-attention blocks
        dim_feedforward: FFN hidden dim (default 4× hidden_dim)
        mamba_v2_config: Dict with Mamba2-specific params:
            d_state (int, default 128): SSM state dimension
            d_conv (int, default 4): Local convolution width
            expand (int, default 2): Inner expansion factor
            headdim (int, default 64): Per-head dimension
            bidirectional (bool, default True): Bidirectional scan
        va_config: V-A head params (grid_size, v_range, a_range, sigma, loss_weight)
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
        va_config: dict = None,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.hidden_dim = hidden_dim

        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4

        # Mamba v2 config
        if mamba_v2_config is None:
            mamba_v2_config = {}
        d_state = mamba_v2_config.get("d_state", 128)
        d_conv = mamba_v2_config.get("d_conv", 4)
        expand = mamba_v2_config.get("expand", 2)
        headdim = mamba_v2_config.get("headdim", 64)
        bidirectional = mamba_v2_config.get("bidirectional", True)

        # V-A config
        if va_config is None:
            va_config = {}
        self.va_grid_size = va_config.get("grid_size", 9)
        self.va_loss_weight = va_config.get("loss_weight", 0.3)
        v_range = tuple(va_config.get("v_range", [1.0, 5.0]))
        a_range = tuple(va_config.get("a_range", [1.0, 5.0]))
        sigma = va_config.get("sigma", 0.5)

        self.va_target_gen = VASoftTargetGenerator(
            grid_size=self.va_grid_size,
            v_range=v_range,
            a_range=a_range,
            sigma=sigma,
        )

        # ── Projection layers ──
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

        # ── Mamba2 temporal encoders ──
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

        # ── Cross-attention fusion ──
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_cross_layers)
        ])

        # ── Head 1: Classification ──
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # ── Head 2: V-A Probability Matrix ──
        self.va_head = VAHead(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            grid_size=self.va_grid_size,
            dropout=dropout,
        )

        self._init_weights()

        inner_dim = hidden_dim * expand
        nheads = inner_dim // headdim
        mode = "Bidirectional" if bidirectional else "Unidirectional"
        logger.info(
            f"MambaV2DualHead: {mode} {num_layers}L × 2mod Mamba2 "
            f"(d_state={d_state}, d_conv={d_conv}, expand={expand}, "
            f"headdim={headdim}, nheads={nheads}) + {num_cross_layers}L CrossAttn | "
            f"V-A grid: {self.va_grid_size}×{self.va_grid_size} | "
            f"Loss: (1-{self.va_loss_weight})×CE + {self.va_loss_weight}×KL"
        )

    def _init_weights(self):
        for name, module in self.named_modules():
            if name.startswith(('text_encoder.model', 'audio_encoder.model')):
                continue
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

    def get_param_groups(self, base_lr: float = 1e-4) -> list:
        """Mamba2 layers → 3× base_lr, cross-attention → base_lr, other → base_lr."""
        mamba_params, cross_attn_params, other_params = [], [], []

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
        valid_samples = sample_mask.sum(dim=1)
        valid_frames = (valid_samples / self.audio_encoder.frame_stride).long().clamp(min=1, max=num_frames)
        frame_indices = torch.arange(num_frames, device=sample_mask.device).unsqueeze(0)
        return frame_indices < valid_frames.unsqueeze(1)

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: MultimodalCollator batch dict

        Returns:
            logits:     (B, num_classes)
            va_probs:   (B, grid_size, grid_size)
            va_targets: (B, grid_size, grid_size)
            va_loss:    scalar
        """
        # 1. Frozen encoder feature extraction
        _, text_seq = self.text_encoder(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
        )
        text_mask = batch["text_attention_mask"].bool()

        _, audio_seq = self.audio_encoder(
            audio_input=batch["audio_input"],
            attention_mask=batch["audio_attention_mask"],
        )
        audio_mask = self._compute_audio_frame_mask(
            batch["audio_attention_mask"], audio_seq.shape[1]
        )

        # 2. Project → Mamba2 → Cross-Attention
        text_features = self.text_proj(text_seq)
        audio_features = self.audio_proj(audio_seq)

        text_features = self.text_mamba(text_features)
        audio_features = self.audio_mamba(audio_features)

        text_pad_mask = ~text_mask
        audio_pad_mask = ~audio_mask

        for cross_block in self.cross_attention_blocks:
            text_features, audio_features = cross_block(
                text_features, audio_features,
                text_pad_mask, audio_pad_mask,
            )

        # 3. Masked mean pooling → concat
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

        # 4. Dual heads
        logits = self.classifier(fused)
        va_probs = self.va_head(fused)

        va_targets = self.va_target_gen.generate(
            batch["valence"], batch["arousal"]
        )

        va_probs_flat = va_probs.view(-1, self.va_grid_size ** 2)
        va_targets_flat = va_targets.view(-1, self.va_grid_size ** 2)

        va_loss = F.kl_div(
            torch.log(va_probs_flat + 1e-9),
            va_targets_flat,
            reduction='batchmean',
        )

        return {
            "logits": logits,
            "va_probs": va_probs,
            "va_targets": va_targets,
            "va_loss": va_loss,
        }

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
