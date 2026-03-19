"""
Mamba + Cross-Attention Hybrid (Model G) — Bidirectional
========================================================
Architecture:
    Audio -> Wav2Vec2.0 (frozen) -> proj 768->d -> 6x BiMambaBlock -> audio features
    Text  -> BERT (frozen)       -> proj 768->d -> 6x BiMambaBlock -> text features
                                       |
                           2x Bidirectional Cross-Attention
                           (text <-> audio interact with each other)
                                       |
                                 masked mean pool
                                       |
                              concat -> MLP -> 4 classes

Improvements over v1 (unidirectional):
    1. Bidirectional Mamba -- forward + backward scan captures full context
    2. Separate param groups for different learning rates
    3. Mamba layers get higher LR (3x), cross-attention gets base LR
"""

import logging

import torch
import torch.nn as nn

from models.encoders import TextEncoder, AudioEncoder
from models.mamba_blocks import MambaTemporalEncoder
from models.baseline_cross_attention import CrossAttentionBlock

logger = logging.getLogger(__name__)


class MambaFusion(nn.Module):
    """Mamba + Cross-Attention hybrid for multimodal emotion recognition.
    
    Args:
        text_encoder: Frozen BERT encoder
        audio_encoder: Frozen Wav2Vec2.0 encoder
        num_classes: Number of emotion classes
        hidden_dim: Model dimension (d_model)
        dropout: Dropout rate
        num_heads: Number of attention heads (for cross-attention only)
        num_layers: Number of Mamba layers per modality
        num_cross_layers: Number of cross-attention blocks
        dim_feedforward: FFN hidden dim in cross-attention (default 4x hidden_dim)
        mamba_config: Dict with Mamba-specific params (d_state, d_conv, expand, bidirectional)
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
        mamba_config: dict = None,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.hidden_dim = hidden_dim
        
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4
        
        # Mamba config
        if mamba_config is None:
            mamba_config = {}
        d_state = mamba_config.get("d_state", 16)
        d_conv = mamba_config.get("d_conv", 4)
        expand = mamba_config.get("expand", 2)
        bidirectional = mamba_config.get("bidirectional", True)
        
        # -- Projection layers: encoder_dim (768) -> hidden_dim (256) --
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
        
        # -- Per-modality Mamba temporal encoders --
        self.audio_mamba = MambaTemporalEncoder(
            d_model=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        
        self.text_mamba = MambaTemporalEncoder(
            d_model=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        
        # -- Cross-attention fusion layers --
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_cross_layers)
        ])
        
        # -- Classification head --
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self._init_weights()
        self._log_architecture(num_layers, d_state, d_conv, expand, num_cross_layers, bidirectional)
    
    def _init_weights(self):
        """Initialize trainable weights (skip frozen encoders and Mamba internals)."""
        for name, module in self.named_modules():
            # Skip frozen encoder submodules
            if name.startswith(('text_encoder.model', 'audio_encoder.model')):
                continue
            # Skip Mamba internal params (initialized by mamba-ssm library)
            if '.mamba_fwd.' in name or '.mamba_bwd.' in name or '.mamba.' in name:
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    continue
            if isinstance(module, nn.Linear) and 'mamba' not in name:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _log_architecture(self, num_layers, d_state, d_conv, expand, num_cross, bidirectional):
        """Log architecture details."""
        audio_info = self.audio_mamba.get_param_count()
        text_info = self.text_mamba.get_param_count()
        mode = "Bidirectional" if bidirectional else "Unidirectional"
        
        logger.info(
            f"MambaFusion architecture: "
            f"{mode} {num_layers}L x 2mod Mamba "
            f"(d_state={d_state}, d_conv={d_conv}, expand={expand}) "
            f"+ {num_cross}L CrossAttn | "
            f"Audio Mamba: {audio_info['total']:,} ({audio_info['per_layer']:,}/layer) | "
            f"Text Mamba: {text_info['total']:,} ({text_info['per_layer']:,}/layer)"
        )
    
    def get_param_groups(self, base_lr: float = 1e-4) -> list:
        """Return parameter groups with different learning rates.
        
        Mamba layers get 3x the base LR because:
        - They have fewer params and need stronger gradients
        - SSM dynamics benefit from faster initial learning
        - Cross-attention layers are more sensitive and need lower LR
        
        Args:
            base_lr: Base learning rate (from config)
            
        Returns:
            List of param group dicts for optimizer
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
            param_groups.append({"params": mamba_params, "lr": base_lr * 3.0, "name": "mamba"})
        if cross_attn_params:
            param_groups.append({"params": cross_attn_params, "lr": base_lr, "name": "cross_attention"})
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr, "name": "other"})
        
        for g in param_groups:
            n_params = sum(p.numel() for p in g["params"])
            logger.info(f"  Param group '{g['name']}': {n_params:,} params, lr={g['lr']:.1e}")
        
        return param_groups
    
    def _compute_audio_frame_mask(self, sample_mask, num_frames):
        """Compute frame-level mask from sample-level mask (vectorized)."""
        valid_samples = sample_mask.sum(dim=1)
        valid_frames = (valid_samples / self.audio_encoder.frame_stride).long().clamp(min=1, max=num_frames)
        frame_indices = torch.arange(num_frames, device=sample_mask.device).unsqueeze(0)
        frame_mask = frame_indices < valid_frames.unsqueeze(1)
        return frame_mask
    
    def forward(self, batch: dict) -> dict:
        """Forward pass: encode -> project -> mamba -> cross-attn -> classify."""
        
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
        
        # 2. Project: 768 -> hidden_dim
        text_features = self.text_proj(text_seq)
        audio_features = self.audio_proj(audio_seq)
        
        # 3. Bidirectional Mamba temporal encoding
        text_features = self.text_mamba(text_features)
        audio_features = self.audio_mamba(audio_features)
        
        # 4. Cross-attention fusion
        text_pad_mask = ~text_mask
        audio_pad_mask = ~audio_mask
        
        for cross_block in self.cross_attention_blocks:
            text_features, audio_features = cross_block(
                text_features, audio_features,
                text_pad_mask, audio_pad_mask,
            )
        
        # 5. Masked mean pooling -> concat -> classify
        text_mask_expanded = text_mask.unsqueeze(-1).float()
        text_pooled = (
            (text_features * text_mask_expanded).sum(dim=1)
            / text_mask_expanded.sum(dim=1).clamp(min=1e-9)
        )
        
        audio_mask_expanded = audio_mask.unsqueeze(-1).float()
        audio_pooled = (
            (audio_features * audio_mask_expanded).sum(dim=1)
            / audio_mask_expanded.sum(dim=1).clamp(min=1e-9)
        )
        
        fused = torch.cat([text_pooled, audio_pooled], dim=-1)
        logits = self.classifier(fused)
        
        return {"logits": logits}
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())