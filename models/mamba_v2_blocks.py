"""
Mamba V2 Blocks — SSD-based temporal modeling layers
=====================================================
Uses Mamba2 (mamba_ssm >= 2.0.0) which implements State Space Duality (SSD).

Key differences from Mamba v1:
    - Multi-head SSM: nheads = (d_model * expand) // headdim
      Constraint: (d_model * expand) must be divisible by headdim
    - Larger state dimension: d_state=128 default (vs 16 in v1)
      The SSD algorithm supports larger states with better efficiency
    - Parallel training via SSD (similar to multi-head attention's parallel form)
    - No causal conv bias by default: cleaner for bidirectional use

Bidirectional design is identical to v1:
    Forward:  [x1, x2, ..., xN] → Mamba2 → forward features
    Backward: [xN, ..., x2, x1] → Mamba2 → flip back
    Output:   Linear(concat(forward, backward)) → d_model

Two separate Mamba2 instances (not weight-shared): forward and backward contexts
have different statistical properties and benefit from independent parameters.

Reference: Dao & Gu (2024). "Transformers are SSMs: Generalized Models and
Efficient Algorithms Through Structured State Space Duality." arXiv:2405.21060
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba2
    MAMBA2_AVAILABLE = True
    logger.debug("Mamba2 (mamba_ssm >= 2.0.0) available.")
except ImportError:
    MAMBA2_AVAILABLE = False
    logger.warning(
        "Mamba2 not found. Upgrade mamba-ssm: pip install mamba-ssm>=2.0.0"
    )


def _check_headdim(d_model: int, expand: int, headdim: int):
    """Validate that (d_model * expand) is divisible by headdim."""
    inner = d_model * expand
    if inner % headdim != 0:
        raise ValueError(
            f"Mamba2 constraint violated: (d_model={d_model} * expand={expand})={inner} "
            f"must be divisible by headdim={headdim}. "
            f"Valid headdim values for inner_dim={inner}: "
            f"{[h for h in [16, 32, 64, 128, 256] if inner % h == 0]}"
        )
    nheads = inner // headdim
    logger.debug(f"Mamba2: d_model={d_model}, expand={expand}, headdim={headdim} → nheads={nheads}")


# ──────────────────────────────────────────────────────────────
# Unidirectional Mamba2 Block (for ablation)
# ──────────────────────────────────────────────────────────────

class MambaV2Block(nn.Module):
    """Single unidirectional Mamba2 block: LayerNorm → Mamba2 → residual.

    Args:
        d_model: Feature dimension
        d_state: SSM state dimension (128 recommended for Mamba2, vs 16 in v1)
        d_conv: Local convolution width
        expand: Inner expansion factor; inner_dim = d_model * expand
        headdim: Per-head dimension; nheads = (d_model * expand) / headdim
        dropout: Post-SSM dropout
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not MAMBA2_AVAILABLE:
            raise ImportError("mamba-ssm >= 2.0.0 required. Run: pip install mamba-ssm>=2.0.0")
        _check_headdim(d_model, expand, headdim)

        self.norm = nn.LayerNorm(d_model)
        self.mamba2 = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) → (B, L, d_model)"""
        residual = x
        x = self.norm(x)
        x = self.mamba2(x)
        x = self.dropout(x)
        return residual + x


# ──────────────────────────────────────────────────────────────
# Bidirectional Mamba2 Block
# ──────────────────────────────────────────────────────────────

class BidirectionalMambaV2Block(nn.Module):
    """Bidirectional Mamba2: forward scan + backward scan + learned merge.

    Architecture:
        x ──┐
            │ → LayerNorm ──┬── Mamba2_fwd(x)                → fwd features
            │               └── Mamba2_bwd(flip(x)) → flip   → bwd features
            │                           │
            │               concat(fwd, bwd) → Linear(2d → d) → Dropout
            │                           │
            └──── residual add ─────────┘

    Args:
        d_model: Feature dimension
        d_state: SSM state dimension (128 recommended for Mamba2)
        d_conv: Local convolution width
        expand: Inner expansion factor
        headdim: Per-head dimension for multi-head SSM
        dropout: Post-merge dropout
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not MAMBA2_AVAILABLE:
            raise ImportError("mamba-ssm >= 2.0.0 required. Run: pip install mamba-ssm>=2.0.0")
        _check_headdim(d_model, expand, headdim)

        self.norm = nn.LayerNorm(d_model)
        self.mamba2_fwd = Mamba2(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand, headdim=headdim,
        )
        self.mamba2_bwd = Mamba2(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand, headdim=headdim,
        )
        self.merge = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) → (B, L, d_model)"""
        residual = x
        x = self.norm(x)

        # Forward scan: left → right
        fwd = self.mamba2_fwd(x)

        # Backward scan: right → left, then flip back to original order
        bwd = self.mamba2_bwd(x.flip(dims=[1])).flip(dims=[1])

        # Learned merge: concat → linear → dropout → residual
        merged = self.merge(torch.cat([fwd, bwd], dim=-1))
        merged = self.dropout(merged)

        return residual + merged


# ──────────────────────────────────────────────────────────────
# Temporal Encoder (stacks of Mamba2 blocks)
# ──────────────────────────────────────────────────────────────

class MambaV2TemporalEncoder(nn.Module):
    """Stack of Mamba2 blocks — drop-in replacement for MambaTemporalEncoder.

    Same interface as MambaTemporalEncoder but uses Mamba2 internals.
    Adds headdim as a new required parameter vs v1.

    Args:
        d_model: Feature dimension
        num_layers: Number of stacked blocks
        d_state: SSM state dimension (128 recommended for Mamba2)
        d_conv: Local convolution width
        expand: Inner expansion factor
        headdim: Per-head dimension; nheads = (d_model * expand) / headdim
        dropout: Dropout rate
        bidirectional: If True, use BidirectionalMambaV2Block
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        block_cls = BidirectionalMambaV2Block if bidirectional else MambaV2Block

        self.layers = nn.ModuleList([
            block_cls(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) → (B, L, d_model)"""
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

    def get_param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        per_layer = sum(p.numel() for p in self.layers[0].parameters()) if self.layers else 0
        return {
            "total": total,
            "per_layer": per_layer,
            "num_layers": len(self.layers),
            "bidirectional": self.bidirectional,
        }
