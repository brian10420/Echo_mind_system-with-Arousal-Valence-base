"""
Model Registry — Maps config names to model classes
====================================================
Usage:
    model = build_model(cfg, text_encoder, audio_encoder)
"""

import logging

from models.encoders import TextEncoder, AudioEncoder
from models.baseline_late_fusion import LateFusionBaseline
from models.baseline_cross_attention import CrossAttentionTransformer
from models.mamba_fusion import MambaFusion

logger = logging.getLogger(__name__)

# Registry: model_name → class
MODEL_REGISTRY = {
    "late_fusion": LateFusionBaseline,
    "cross_attention": CrossAttentionTransformer,
    "mamba_fusion": MambaFusion,
    # Future:
    # "mamba_dual_head": MambaDualHead,
}


def build_encoders(cfg: dict) -> tuple[TextEncoder, AudioEncoder]:
    """Build frozen text and audio encoders from config."""
    text_encoder = TextEncoder(
        model_id=cfg["text_encoder"]["model_id"],
        freeze=cfg["text_encoder"]["freeze"],
        pooling=cfg["text_encoder"]["pooling"],
        output_dim=cfg["text_encoder"]["output_dim"],
    )
    
    audio_encoder = AudioEncoder(
        model_id=cfg["audio_encoder"]["model_id"],
        freeze=cfg["audio_encoder"]["freeze"],
        output_dim=cfg["audio_encoder"]["output_dim"],
    )
    
    return text_encoder, audio_encoder


def build_model(
    cfg: dict,
    text_encoder: TextEncoder,
    audio_encoder: AudioEncoder,
) -> LateFusionBaseline:
    """Build model from config.
    
    Args:
        cfg: Full config dict
        text_encoder: Frozen text encoder
        audio_encoder: Frozen audio encoder
        
    Returns:
        Model instance
    """
    model_name = cfg["model"]["name"]
    
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    model_cls = MODEL_REGISTRY[model_name]
    
    # Common kwargs shared by all models
    kwargs = {
        "text_encoder": text_encoder,
        "audio_encoder": audio_encoder,
        "num_classes": cfg["dataset"]["num_classes"],
        "hidden_dim": cfg["model"]["hidden_dim"],
        "dropout": cfg["model"]["dropout"],
    }
    
    # Model-specific kwargs for models with temporal + cross-attention layers
    if model_name in ("cross_attention", "mamba_fusion", "mamba_dual_head"):
        kwargs["num_heads"] = cfg["model"]["num_heads"]
        kwargs["num_layers"] = cfg["model"]["num_layers"]
    
    # Mamba-specific config
    if model_name in ("mamba_fusion", "mamba_dual_head"):
        kwargs["mamba_config"] = cfg.get("mamba", {})
    
    # Dual-head V-A config
    if model_name == "mamba_dual_head":
        kwargs["va_config"] = cfg.get("va_head", {})
    
    model = model_cls(**kwargs)
    
    trainable = model.get_trainable_params()
    total = model.get_total_params()
    logger.info(
        f"Model '{model_name}' built: "
        f"{trainable:,} trainable / {total:,} total params "
        f"({trainable/total*100:.1f}% trainable)"
    )
    
    return model


__all__ = ["build_encoders", "build_model", "MODEL_REGISTRY"]