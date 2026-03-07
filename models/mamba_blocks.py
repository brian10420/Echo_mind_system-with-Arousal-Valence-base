"""
Mamba Blocks — SSM-based temporal modeling layers
=================================================
Includes both unidirectional and bidirectional variants.

Bidirectional Mamba processes sequences in both directions:
    Forward:  [x1, x2, ..., xN] → Mamba → forward features
    Backward: [xN, ..., x2, x1] → Mamba → flip back
    Output:   Linear(concat(forward, backward)) → d_model

This is critical for emotion recognition because:
- A sigh at the end changes the meaning of "I'm fine" at the start
- Prosodic contours (rising/falling pitch) need full context
- Unlike language modeling (causal), audio emotion is non-causal

Uses the official mamba-ssm library (Gu & Dao, 2023).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning("mamba-ssm not installed. Install with: pip install mamba-ssm>=1.0.0")


# ──────────────────────────────────────────────────────────────
# Unidirectional Mamba Block (kept for ablation)
# ──────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Single unidirectional Mamba block: LayerNorm → Mamba → residual."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm required. Install: pip install mamba-ssm>=1.0.0")
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d) → (B, L, d)"""
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


# ──────────────────────────────────────────────────────────────
# Bidirectional Mamba Block
# ──────────────────────────────────────────────────────────────

class BidirectionalMambaBlock(nn.Module):
    """Bidirectional Mamba: forward scan + backward scan + learned merge.
    
    Architecture:
        x ──┐
            │ → LayerNorm ──┬── Mamba_fwd(x)               → fwd features
            │               └── Mamba_bwd(flip(x)) → flip  → bwd features
            │                          │
            │               concat(fwd, bwd) → Linear(2d → d) → Dropout
            │                          │
            └──── residual add ────────┘
    
    Two separate Mamba instances (not weight-shared) because forward
    and backward contexts have different statistical properties.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm required. Install: pip install mamba-ssm>=1.0.0")
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.merge = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d) → (B, L, d)"""
        residual = x
        x = self.norm(x)
        
        # Forward scan: left → right
        fwd = self.mamba_fwd(x)
        
        # Backward scan: right → left, then flip back to original order
        bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        
        # Learned merge: concat → project → residual
        merged = self.merge(torch.cat([fwd, bwd], dim=-1))
        merged = self.dropout(merged)
        
        return residual + merged


# ──────────────────────────────────────────────────────────────
# Temporal Encoder (stacks of Mamba blocks)
# ──────────────────────────────────────────────────────────────

class MambaTemporalEncoder(nn.Module):
    """Stack of MambaBlocks — drop-in replacement for nn.TransformerEncoder.
    
    Supports both unidirectional and bidirectional modes via config.
    
    Args:
        d_model: Feature dimension
        num_layers: Number of stacked blocks
        d_state: SSM state dimension
        d_conv: Local convolution width
        expand: Inner expansion factor
        dropout: Dropout rate
        bidirectional: If True, use BidirectionalMambaBlock
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        
        block_cls = BidirectionalMambaBlock if bidirectional else MambaBlock
        
        self.layers = nn.ModuleList([
            block_cls(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) → (B, L, d_model). No mask needed — Mamba handles it."""
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
    
    def get_param_count(self) -> dict:
        """Detailed parameter breakdown."""
        total = sum(p.numel() for p in self.parameters())
        per_layer = sum(p.numel() for p in self.layers[0].parameters()) if self.layers else 0
        return {
            "total": total,
            "per_layer": per_layer,
            "num_layers": len(self.layers),
            "bidirectional": self.bidirectional,
        }