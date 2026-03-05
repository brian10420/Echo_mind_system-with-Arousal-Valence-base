"""
Mamba Blocks — SSM-based temporal modeling layers
=================================================
Drop-in replacement for nn.TransformerEncoderLayer.

Each MambaBlock performs:
    x → LayerNorm → Mamba SSM → residual → output

Uses the official mamba-ssm library (Gu & Dao, 2023).
Mamba processes sequences in O(N) linear time vs Transformer's O(N²).

Key parameters:
    d_model:  Model dimension (256)
    d_state:  SSM state dimension (16) — controls how much history is compressed
    d_conv:   Local convolution width (4) — short-range feature extraction
    expand:   Expansion factor (2) — inner dimension = d_model × expand = 512
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Try importing mamba-ssm, fall back gracefully
# ──────────────────────────────────────────────────────────────

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    logger.info("mamba-ssm library loaded successfully")
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning(
        "mamba-ssm not installed. Install with: pip install mamba-ssm>=1.0.0. "
        "MambaBlock will not be available."
    )


class MambaBlock(nn.Module):
    """Single Mamba block with pre-norm and residual connection.
    
    Architecture:
        x ──┐
            │ → LayerNorm → Mamba SSM → Dropout
            │                              │
            └──── residual add ────────────┘
                        │
                      output
    
    This mirrors the structure of a TransformerEncoderLayer
    but replaces self-attention with selective state space modeling.
    
    Args:
        d_model: Input/output feature dimension
        d_state: SSM state expansion factor (N in the paper)
        d_conv: Width of the local convolution
        expand: Inner dimension expansion factor
        dropout: Dropout rate after SSM output
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
            raise ImportError(
                "mamba-ssm library is required for MambaBlock. "
                "Install with: pip install mamba-ssm>=1.0.0"
            )
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) input sequence
            
        Returns:
            (B, L, d_model) output sequence with same shape
        """
        # Pre-norm + Mamba + residual
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


class MambaTemporalEncoder(nn.Module):
    """Stack of MambaBlocks for temporal sequence modeling.
    
    Drop-in replacement for nn.TransformerEncoder.
    Processes a sequence of features through N stacked MambaBlocks,
    with a final LayerNorm for output stability.
    
    Args:
        d_model: Feature dimension
        num_layers: Number of stacked MambaBlocks
        d_state: SSM state dimension
        d_conv: Local convolution width
        expand: Inner expansion factor
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final norm (matches TransformerEncoder convention)
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) input sequence
            
        Returns:
            (B, L, d_model) output sequence
            
        Note:
            Unlike TransformerEncoder, MambaTemporalEncoder does NOT
            need a padding mask. Mamba processes sequences causally and
            the padding is handled by masking during pooling (downstream).
            This is one of Mamba's advantages — no quadratic mask computation.
        """
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        return x
    
    def get_param_count(self) -> dict:
        """Detailed parameter breakdown for logging."""
        total = sum(p.numel() for p in self.parameters())
        per_layer = sum(p.numel() for p in self.layers[0].parameters()) if self.layers else 0
        return {
            "total": total,
            "per_layer": per_layer,
            "num_layers": len(self.layers),
        }