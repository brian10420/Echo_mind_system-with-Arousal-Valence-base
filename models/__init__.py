"""
Model Registry — Maps config names to model classes
====================================================
Usage:
    model = build_model(cfg, text_encoder, audio_encoder)
"""

import logging

from models.encoders import TextEncoder, AudioEncoder
from models.baseline_late_fusion import LateFusionBaseline

logger = logging.getLogger(__name__)

# Registry: model_name → class
MODEL_REGISTRY = {
    "late_fusion": LateFusionBaseline,
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


def build_model(cfg, text_encoder, audio_encoder):
    """Build model from config."""
    model_name = cfg["model"]["name"]
    
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    model_cls = MODEL_REGISTRY[model_name]
    
    kwargs = {
        "text_encoder": text_encoder,
        "audio_encoder": audio_encoder,
        "num_classes": cfg["dataset"]["num_classes"],
        "hidden_dim": cfg["model"]["hidden_dim"],
        "dropout": cfg["model"]["dropout"],
    }
    
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