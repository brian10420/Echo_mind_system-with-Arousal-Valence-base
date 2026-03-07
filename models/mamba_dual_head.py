"""
Mamba Dual-Head Model (Model H) — Classification + V-A Probability Matrix
==========================================================================
Architecture:
    Audio → Wav2Vec2.0 (frozen) → proj → 6× BiMamba → audio features
    Text  → BERT (frozen)       → proj → 6× BiMamba → text features
                                    ↓
                        2× Cross-Attention Fusion
                                    ↓
                          masked mean pool → concat (B, 512)
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            Head 1: Classification           Head 2: V-A Probability
            MLP → 4 classes                  MLP → 9×9 grid (81 bins)
            Loss: CrossEntropy               Loss: KL-Divergence
                    ↓                               ↓
                    └───────────────┬───────────────┘
                    Total Loss = (1-α) × CE + α × KL

This is the full contribution model from the research proposal.
Head 2 outputs the "Probabilistic Distribution Embedding" — a soft
probability distribution over the Valence-Arousal space that preserves
emotional ambiguity and feeds into Phase 2 (LLM prompt prefix).
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import TextEncoder, AudioEncoder
from models.mamba_blocks import MambaTemporalEncoder
from models.baseline_cross_attention import CrossAttentionBlock

logger = logging.getLogger(__name__)


class VAHead(nn.Module):
    """Valence-Arousal probability distribution head.
    
    Outputs a (grid_size × grid_size) probability distribution over V-A space.
    Each cell represents P(V=v_i, A=a_j | input).
    
    Args:
        input_dim: Input feature dimension (2 × hidden_dim from concat)
        hidden_dim: Hidden layer dimension
        grid_size: Grid resolution (9 → 9×9 = 81 bins)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        grid_size: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.grid_size = grid_size
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, grid_size * grid_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) fused features
        Returns:
            (B, grid_size, grid_size) probability distribution over V-A space
            Each row sums to... well, the entire grid sums to 1.0 (softmax over all cells)
        """
        logits = self.mlp(x)                          # (B, grid_size²)
        probs = F.softmax(logits, dim=-1)             # (B, grid_size²)
        probs = probs.view(-1, self.grid_size, self.grid_size)  # (B, G, G)
        return probs


class VASoftTargetGenerator:
    """Generates soft V-A target distributions from continuous V-A values.
    
    Converts a point (V=2.1, A=3.8) into a 2D Gaussian probability
    distribution over the 9×9 grid, preserving emotional ambiguity.
    
    Args:
        grid_size: Grid resolution (9)
        v_range: (min, max) valence range
        a_range: (min, max) arousal range
        sigma: Gaussian standard deviation (controls spread)
    """
    
    def __init__(
        self,
        grid_size: int = 9,
        v_range: tuple = (1.0, 5.0),
        a_range: tuple = (1.0, 5.0),
        sigma: float = 0.5,
    ):
        self.grid_size = grid_size
        self.sigma = sigma
        
        # Precompute grid centers
        self.v_centers = torch.linspace(v_range[0], v_range[1], grid_size)  # (G,)
        self.a_centers = torch.linspace(a_range[0], a_range[1], grid_size)  # (G,)
        
        # Create meshgrid for vectorized computation
        # v_grid: (G, G), a_grid: (G, G)
        self.v_grid, self.a_grid = torch.meshgrid(
            self.v_centers, self.a_centers, indexing='ij'
        )
    
    def generate(
        self, valence: torch.Tensor, arousal: torch.Tensor
    ) -> torch.Tensor:
        """Generate soft target distributions for a batch.
        
        Args:
            valence: (B,) continuous valence values
            arousal: (B,) continuous arousal values
            
        Returns:
            (B, grid_size, grid_size) soft probability distributions
        """
        B = valence.shape[0]
        device = valence.device
        
        # Move grids to same device
        v_grid = self.v_grid.to(device)  # (G, G)
        a_grid = self.a_grid.to(device)  # (G, G)
        
        # Expand for batch: (B, 1, 1) and (1, G, G)
        v = valence.view(B, 1, 1)   # (B, 1, 1)
        a = arousal.view(B, 1, 1)   # (B, 1, 1)
        
        v_g = v_grid.unsqueeze(0)    # (1, G, G)
        a_g = a_grid.unsqueeze(0)    # (1, G, G)
        
        # 2D Gaussian: exp(-((v - v_center)² + (a - a_center)²) / (2σ²))
        dist_sq = (v - v_g) ** 2 + (a - a_g) ** 2  # (B, G, G)
        gaussian = torch.exp(-dist_sq / (2 * self.sigma ** 2))  # (B, G, G)
        
        # Normalize to sum to 1
        gaussian = gaussian / gaussian.sum(dim=(1, 2), keepdim=True).clamp(min=1e-9)
        
        return gaussian  # (B, G, G)


class MambaDualHead(nn.Module):
    """Mamba + Cross-Attention with dual output heads.
    
    Head 1: 4-class emotion classification (CrossEntropy)
    Head 2: 9×9 V-A probability matrix (KL-Divergence)
    
    The shared backbone (Mamba + CrossAttn) is identical to MambaFusion.
    Only the output stage differs.
    
    Args:
        text_encoder: Frozen BERT encoder
        audio_encoder: Frozen Wav2Vec2.0 encoder
        num_classes: Number of emotion classes
        hidden_dim: Model dimension (d_model)
        dropout: Dropout rate
        num_heads: Number of attention heads (for cross-attention)
        num_layers: Number of Mamba layers per modality
        num_cross_layers: Number of cross-attention blocks
        dim_feedforward: FFN hidden dim (default 4× hidden_dim)
        mamba_config: Mamba params (d_state, d_conv, expand, bidirectional)
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
        mamba_config: dict = None,
        va_config: dict = None,
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
        
        # V-A config
        if va_config is None:
            va_config = {}
        self.va_grid_size = va_config.get("grid_size", 9)
        self.va_loss_weight = va_config.get("loss_weight", 0.3)
        v_range = tuple(va_config.get("v_range", [1.0, 5.0]))
        a_range = tuple(va_config.get("a_range", [1.0, 5.0]))
        sigma = va_config.get("sigma", 0.5)
        
        # V-A soft target generator (not a nn.Module, just a utility)
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
        
        # ── Mamba temporal encoders ──
        self.audio_mamba = MambaTemporalEncoder(
            d_model=hidden_dim, num_layers=num_layers,
            d_state=d_state, d_conv=d_conv, expand=expand,
            dropout=dropout, bidirectional=bidirectional,
        )
        self.text_mamba = MambaTemporalEncoder(
            d_model=hidden_dim, num_layers=num_layers,
            d_state=d_state, d_conv=d_conv, expand=expand,
            dropout=dropout, bidirectional=bidirectional,
        )
        
        # ── Cross-attention fusion ──
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=hidden_dim, num_heads=num_heads,
                dim_feedforward=dim_feedforward, dropout=dropout,
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
        
        logger.info(
            f"MambaDualHead: {'Bi' if bidirectional else 'Uni'}directional "
            f"{num_layers}L Mamba + {num_cross_layers}L CrossAttn | "
            f"V-A grid: {self.va_grid_size}×{self.va_grid_size} | "
            f"Loss: (1-{self.va_loss_weight})×CE + {self.va_loss_weight}×KL"
        )
    
    def _init_weights(self):
        """Initialize trainable weights."""
        for name, module in self.named_modules():
            if name.startswith(('text_encoder.model', 'audio_encoder.model')):
                continue
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
    
    def get_param_groups(self, base_lr: float = 1e-4) -> list:
        """Return parameter groups with different learning rates."""
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
        valid_frames = (valid_samples / 320).long().clamp(min=1, max=num_frames)
        frame_indices = torch.arange(num_frames, device=sample_mask.device).unsqueeze(0)
        return frame_indices < valid_frames.unsqueeze(1)
    
    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: Dict from MultimodalCollator
            
        Returns:
            Dict with:
                - logits: (B, num_classes) classification logits
                - va_probs: (B, grid_size, grid_size) V-A probability matrix
                - va_targets: (B, grid_size, grid_size) soft target distributions
                - va_loss: scalar KL-divergence loss
        """
        # ══════════════════════════════════════════════════════
        # 1. Frozen encoder feature extraction
        # ══════════════════════════════════════════════════════
        
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
        
        # ══════════════════════════════════════════════════════
        # 2. Project → Mamba → Cross-Attention
        # ══════════════════════════════════════════════════════
        
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
        
        # ══════════════════════════════════════════════════════
        # 3. Masked mean pooling → concat
        # ══════════════════════════════════════════════════════
        
        text_mask_exp = text_mask.unsqueeze(-1).float()
        text_pooled = (text_features * text_mask_exp).sum(dim=1) / text_mask_exp.sum(dim=1).clamp(min=1e-9)
        
        audio_mask_exp = audio_mask.unsqueeze(-1).float()
        audio_pooled = (audio_features * audio_mask_exp).sum(dim=1) / audio_mask_exp.sum(dim=1).clamp(min=1e-9)
        
        fused = torch.cat([text_pooled, audio_pooled], dim=-1)  # (B, 2×hidden_dim)
        
        # ══════════════════════════════════════════════════════
        # 4. Dual heads
        # ══════════════════════════════════════════════════════
        
        # Head 1: Classification
        logits = self.classifier(fused)  # (B, num_classes)
        
        # Head 2: V-A probability matrix
        va_probs = self.va_head(fused)  # (B, G, G)
        
        # Generate soft targets from ground-truth V-A values
        va_targets = self.va_target_gen.generate(
            batch["valence"], batch["arousal"]
        )  # (B, G, G)
        
        # KL-divergence loss: KL(target || predicted)
        # Add small epsilon to avoid log(0)
        va_probs_flat = va_probs.view(-1, self.va_grid_size * self.va_grid_size)  # (B, 81)
        va_targets_flat = va_targets.view(-1, self.va_grid_size * self.va_grid_size)  # (B, 81)
        
        # KL(P || Q) = sum(P * log(P / Q))
        va_loss = F.kl_div(
            torch.log(va_probs_flat + 1e-9),  # log(Q) — model prediction
            va_targets_flat,                    # P — target distribution
            reduction='batchmean',
        )
        
        return {
            "logits": logits,
            "va_probs": va_probs,
            "va_targets": va_targets,
            "va_loss": va_loss,
        }
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())