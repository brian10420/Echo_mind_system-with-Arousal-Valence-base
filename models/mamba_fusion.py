"""
Mamba + Cross-Attention Hybrid (Model G)
========================================
Architecture:
    Audio → Wav2Vec2.0 (frozen) → proj 768→d → 6× MambaBlock → audio features
    Text  → BERT (frozen)       → proj 768→d → 6× MambaBlock → text features
                                       ↓
                           2× Bidirectional Cross-Attention
                           (text ↔ audio interact with each other)
                                       ↓
                                 masked mean pool
                                       ↓
                              concat → MLP → 4 classes

Compared to CrossAttentionTransformer:
    - ONLY difference: TransformerEncoder → MambaTemporalEncoder
    - Same projections, same cross-attention, same classifier
    - This makes the comparison scientifically clean

Advantages over Transformer:
    1. O(N) linear complexity for within-modality temporal modeling
    2. ~33% fewer trainable parameters at same d_model
    3. No positional encoding needed (SSM has inherent position awareness)
    4. Better long-sequence handling (critical for 550-frame audio)
"""

import torch
import torch.nn as nn

from models.encoders import TextEncoder, AudioEncoder
from models.mamba_blocks import MambaTemporalEncoder
from models.baseline_cross_attention import CrossAttentionBlock


class MambaFusion(nn.Module):
    """Mamba + Cross-Attention hybrid for multimodal emotion recognition.
    
    Architecture:
        1. Frozen encoders extract features (shared with all models)
        2. Linear projection: 768 → hidden_dim
        3. Per-modality Mamba temporal encoder (SSM within each modality)
        4. Cross-attention fusion (interaction between modalities)
        5. Masked mean pooling → concat → MLP classifier
    
    Args:
        text_encoder: Frozen BERT encoder
        audio_encoder: Frozen Wav2Vec2.0 encoder
        num_classes: Number of emotion classes
        hidden_dim: Model dimension (d_model)
        dropout: Dropout rate
        num_heads: Number of attention heads (for cross-attention only)
        num_layers: Number of Mamba layers per modality
        num_cross_layers: Number of cross-attention blocks
        dim_feedforward: FFN hidden dim in cross-attention (default 4× hidden_dim)
        mamba_config: Dict with Mamba-specific params (d_state, d_conv, expand)
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
        
        # Default Mamba config
        if mamba_config is None:
            mamba_config = {}
        d_state = mamba_config.get("d_state", 16)
        d_conv = mamba_config.get("d_conv", 4)
        expand = mamba_config.get("expand", 2)
        
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
        
        # ── Per-modality Mamba temporal encoders ──
        # NOTE: No positional encoding needed — Mamba's SSM has
        # inherent position awareness through its recurrent state
        self.audio_mamba = MambaTemporalEncoder(
            d_model=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        self.text_mamba = MambaTemporalEncoder(
            d_model=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        # ── Cross-attention fusion layers ──
        # (identical to CrossAttentionTransformer — fair comparison)
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
        self._log_architecture(num_layers, d_state, d_conv, expand, num_cross_layers)
    
    def _init_weights(self):
        """Initialize trainable weights (skip frozen encoders and Mamba internals)."""
        for name, module in self.named_modules():
            # Skip frozen encoder submodules
            if name.startswith(('text_encoder.model', 'audio_encoder.model')):
                continue
            # Skip Mamba internal parameters (initialized by mamba-ssm library)
            if 'mamba.mamba' in name or '.mamba.A_log' in name or '.mamba.D' in name:
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _log_architecture(self, num_layers, d_state, d_conv, expand, num_cross):
        """Log architecture details."""
        import logging
        logger = logging.getLogger(__name__)
        
        audio_info = self.audio_mamba.get_param_count()
        text_info = self.text_mamba.get_param_count()
        
        logger.info(
            f"MambaFusion architecture: "
            f"{num_layers}L×2mod Mamba (d_state={d_state}, d_conv={d_conv}, expand={expand}) "
            f"+ {num_cross}L CrossAttn | "
            f"Audio Mamba: {audio_info['total']:,} params ({audio_info['per_layer']:,}/layer) | "
            f"Text Mamba: {text_info['total']:,} params ({text_info['per_layer']:,}/layer)"
        )
    
    def _compute_audio_frame_mask(
        self, sample_mask: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """Compute frame-level mask from sample-level mask (vectorized).
        
        Args:
            sample_mask: (B, num_samples) True = valid sample
            num_frames: Number of output frames from Wav2Vec2.0
            
        Returns:
            (B, num_frames) True = valid frame
        """
        valid_samples = sample_mask.sum(dim=1)  # (B,)
        valid_frames = (valid_samples / 320).long().clamp(min=1, max=num_frames)
        
        # Vectorized mask — no loop
        frame_indices = torch.arange(num_frames, device=sample_mask.device).unsqueeze(0)
        frame_mask = frame_indices < valid_frames.unsqueeze(1)
        
        return frame_mask
    
    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: Dict from MultimodalCollator with keys:
                - audio_input: (B, num_samples)
                - audio_attention_mask: (B, num_samples)
                - text_input_ids: (B, S)
                - text_attention_mask: (B, S)
                - labels: (B,)
                
        Returns:
            Dict with:
                - logits: (B, num_classes) classification logits
        """
        # ══════════════════════════════════════════════════════
        # 1. Extract features from frozen encoders
        # ══════════════════════════════════════════════════════
        
        # Text: (B, S, 768) full sequence
        _, text_seq = self.text_encoder(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
        )
        text_mask = batch["text_attention_mask"].bool()  # (B, S) True = valid
        
        # Audio: (B, T, 768) frame-level features
        _, audio_seq = self.audio_encoder(
            audio_input=batch["audio_input"],
            attention_mask=batch["audio_attention_mask"],
        )
        audio_mask = self._compute_audio_frame_mask(
            batch["audio_attention_mask"], audio_seq.shape[1]
        )  # (B, T) True = valid
        
        # ══════════════════════════════════════════════════════
        # 2. Project: 768 → hidden_dim
        # ══════════════════════════════════════════════════════
        
        text_features = self.text_proj(text_seq)      # (B, S, hidden_dim)
        audio_features = self.audio_proj(audio_seq)    # (B, T, hidden_dim)
        
        # ══════════════════════════════════════════════════════
        # 3. Per-modality Mamba temporal encoding
        # ══════════════════════════════════════════════════════
        # NOTE: Mamba does NOT need positional encoding or padding masks.
        # It processes sequences causally through its recurrent state.
        # Padding is handled during pooling (step 6).
        
        text_features = self.text_mamba(text_features)    # (B, S, hidden_dim)
        audio_features = self.audio_mamba(audio_features)  # (B, T, hidden_dim)
        
        # ══════════════════════════════════════════════════════
        # 4. Cross-attention fusion
        # ══════════════════════════════════════════════════════
        # PyTorch cross-attention: key_padding_mask True = IGNORE
        # Our mask: True = VALID → invert
        
        text_pad_mask = ~text_mask    # (B, S) True = ignore
        audio_pad_mask = ~audio_mask  # (B, T) True = ignore
        
        for cross_block in self.cross_attention_blocks:
            text_features, audio_features = cross_block(
                text_features, audio_features,
                text_pad_mask, audio_pad_mask,
            )
        
        # ══════════════════════════════════════════════════════
        # 5. Masked mean pooling → concat → classify
        # ══════════════════════════════════════════════════════
        
        # Text pooling
        text_mask_expanded = text_mask.unsqueeze(-1).float()  # (B, S, 1)
        text_pooled = (
            (text_features * text_mask_expanded).sum(dim=1)
            / text_mask_expanded.sum(dim=1).clamp(min=1e-9)
        )  # (B, hidden_dim)
        
        # Audio pooling
        audio_mask_expanded = audio_mask.unsqueeze(-1).float()  # (B, T, 1)
        audio_pooled = (
            (audio_features * audio_mask_expanded).sum(dim=1)
            / audio_mask_expanded.sum(dim=1).clamp(min=1e-9)
        )  # (B, hidden_dim)
        
        # Concat and classify
        fused = torch.cat([text_pooled, audio_pooled], dim=-1)  # (B, 2 × hidden_dim)
        logits = self.classifier(fused)  # (B, num_classes)
        
        return {"logits": logits}
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters (excludes frozen encoders)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count all parameters including frozen encoders."""
        return sum(p.numel() for p in self.parameters())