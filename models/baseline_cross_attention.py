"""
Cross-Attention Transformer Baseline (Model E)
===============================================
Architecture:
    Audio → Wav2Vec2.0 (frozen) → proj 768→d → 6× TransformerEncoder → audio features
    Text  → BERT (frozen)       → proj 768→d → 6× TransformerEncoder → text features
                                       ↓
                           2× Bidirectional Cross-Attention
                           (text ↔ audio interact with each other)
                                       ↓
                                 masked mean pool
                                       ↓
                              concat → MLP → 4 classes

This is the Transformer baseline that reviewers expect. Same frozen encoders
as late fusion, but with learnable self-attention + cross-modal attention.
The ONLY difference vs the Mamba model will be: TransformerEncoder → MambaBlock.
"""

import math

import torch
import torch.nn as nn

from models.encoders import TextEncoder, AudioEncoder


# ──────────────────────────────────────────────────────────────
# Sinusoidal Positional Encoding
# ──────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (not learned).
    
    Supports variable sequence lengths — precomputes up to max_len positions.
    Same as the original Transformer paper (Vaswani et al., 2017).
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length to precompute
        dropout: Dropout rate applied after adding PE
    """
    
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute PE matrix: (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model) input features
        Returns:
            (B, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────
# Cross-Attention Block (Bidirectional)
# ──────────────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention: text ↔ audio.
    
    Each block performs:
    1. text attends to audio (text as Q, audio as K/V) + residual + norm + FFN
    2. audio attends to updated text (audio as Q, text as K/V) + residual + norm + FFN
    
    Args:
        d_model: Feature dimension
        num_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Text attends to Audio
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.text_norm2 = nn.LayerNorm(d_model)
        
        # Audio attends to Text
        self.audio_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.audio_norm1 = nn.LayerNorm(d_model)
        self.audio_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.audio_norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        text_features: torch.Tensor,    # (B, S, d)
        audio_features: torch.Tensor,   # (B, T, d)
        text_pad_mask: torch.Tensor,    # (B, S) True = IGNORE (PyTorch convention)
        audio_pad_mask: torch.Tensor,   # (B, T) True = IGNORE (PyTorch convention)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_out: (B, S, d) updated text features
            audio_out: (B, T, d) updated audio features
        """
        # 1. Text attends to Audio: Q=text, K=V=audio
        text_attended, _ = self.text_cross_attn(
            query=text_features,
            key=audio_features,
            value=audio_features,
            key_padding_mask=audio_pad_mask,  # mask audio padding
        )
        text_out = self.text_norm1(text_features + text_attended)  # residual + norm
        text_out = self.text_norm2(text_out + self.text_ffn(text_out))  # FFN + residual + norm
        
        # 2. Audio attends to (updated) Text: Q=audio, K=V=text_out
        audio_attended, _ = self.audio_cross_attn(
            query=audio_features,
            key=text_out,
            value=text_out,
            key_padding_mask=text_pad_mask,  # mask text padding
        )
        audio_out = self.audio_norm1(audio_features + audio_attended)  # residual + norm
        audio_out = self.audio_norm2(audio_out + self.audio_ffn(audio_out))  # FFN + residual + norm
        
        return text_out, audio_out


# ──────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────

class CrossAttentionTransformer(nn.Module):
    """Cross-Attention Transformer for multimodal emotion recognition.
    
    Architecture:
        1. Frozen encoders extract features (shared with all models)
        2. Linear projection: 768 → hidden_dim
        3. Per-modality Transformer encoder (self-attention within modality)
        4. Cross-attention fusion (interaction between modalities)
        5. Masked mean pooling → concat → MLP classifier
    
    Args:
        text_encoder: Frozen BERT encoder
        audio_encoder: Frozen Wav2Vec2.0 encoder
        num_classes: Number of emotion classes
        hidden_dim: Model dimension (d_model) for all Transformer layers
        dropout: Dropout rate
        num_heads: Number of attention heads
        num_layers: Number of Transformer encoder layers per modality
        num_cross_layers: Number of cross-attention blocks
        dim_feedforward: FFN hidden dimension (default 4× hidden_dim)
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
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.hidden_dim = hidden_dim
        
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4  # standard Transformer ratio
        
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
        
        # ── Positional encoding ──
        self.audio_pe = SinusoidalPositionalEncoding(hidden_dim, max_len=4096, dropout=dropout)
        self.text_pe = SinusoidalPositionalEncoding(hidden_dim, max_len=512, dropout=dropout)
        
        # ── Per-modality Transformer encoders (self-attention) ──
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable training)
        )
        self.audio_transformer = nn.TransformerEncoder(
            audio_encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        
        text_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.text_transformer = nn.TransformerEncoder(
            text_encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        
        # ── Cross-attention fusion layers ──
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
        # Input: concat of text_pooled + audio_pooled = 2 × hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize trainable weights with Xavier uniform."""
        for name, module in self.named_modules():
            # Skip frozen encoder submodules
            if name.startswith(('text_encoder.model', 'audio_encoder.model')):
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _compute_audio_frame_mask(
        self, sample_mask: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """Compute frame-level mask from sample-level mask (vectorized).
        
        Wav2Vec2.0 downsamples ~320×, so valid_frames ≈ valid_samples / 320.
        
        Args:
            sample_mask: (B, num_samples) True = valid sample
            num_frames: Number of output frames from Wav2Vec2.0
            
        Returns:
            (B, num_frames) True = valid frame
        """
        valid_samples = sample_mask.sum(dim=1)  # (B,)
        valid_frames = (valid_samples / self.audio_encoder.frame_stride).long().clamp(min=1, max=num_frames)

        # Vectorized mask creation — no Python loop
        frame_indices = torch.arange(num_frames, device=sample_mask.device).unsqueeze(0)  # (1, T)
        frame_mask = frame_indices < valid_frames.unsqueeze(1)  # (B, T)
        
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
        
        text_features = self.text_proj(text_seq)    # (B, S, hidden_dim)
        audio_features = self.audio_proj(audio_seq)  # (B, T, hidden_dim)
        
        # ══════════════════════════════════════════════════════
        # 3. Add positional encoding
        # ══════════════════════════════════════════════════════
        
        text_features = self.text_pe(text_features)
        audio_features = self.audio_pe(audio_features)
        
        # ══════════════════════════════════════════════════════
        # 4. Per-modality self-attention (Transformer encoder)
        # ══════════════════════════════════════════════════════
        
        # PyTorch convention: src_key_padding_mask True = IGNORE
        # Our convention: mask True = VALID
        # So we INVERT: ~mask
        text_pad_mask = ~text_mask    # (B, S) True = ignore
        audio_pad_mask = ~audio_mask  # (B, T) True = ignore
        
        text_features = self.text_transformer(
            text_features,
            src_key_padding_mask=text_pad_mask,
        )  # (B, S, hidden_dim)
        
        audio_features = self.audio_transformer(
            audio_features,
            src_key_padding_mask=audio_pad_mask,
        )  # (B, T, hidden_dim)
        
        # ══════════════════════════════════════════════════════
        # 5. Cross-attention fusion
        # ══════════════════════════════════════════════════════
        
        for cross_block in self.cross_attention_blocks:
            text_features, audio_features = cross_block(
                text_features, audio_features,
                text_pad_mask, audio_pad_mask,
            )
        
        # ══════════════════════════════════════════════════════
        # 6. Masked mean pooling → concat → classify
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