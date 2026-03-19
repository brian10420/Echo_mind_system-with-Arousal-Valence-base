"""
Model Registry — Maps config names to model classes
====================================================
Adding a new model:
    1. Create your model class with the standard interface
    2. Add one line to MODEL_REGISTRY below
    That's it. build_model() auto-dispatches kwargs based on __init__ signature.

Usage:
    text_enc, audio_enc = build_encoders(cfg)
    model = build_model(cfg, text_enc, audio_enc)
"""

import inspect
import logging
from typing import Union

import torch.nn as nn

from models.encoders import TextEncoder, AudioEncoder
from models.baseline_late_fusion import LateFusionBaseline
from models.baseline_cross_attention import CrossAttentionTransformer
from models.mamba_fusion import MambaFusion
from models.mamba_dual_head import MambaDualHead
from models.mamba_v2_fusion import MambaV2Fusion
from models.mamba_v2_dual_head import MambaV2DualHead

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Registry: model name (str) → model class
# To add a new model, just add one line here.
# ──────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "late_fusion": LateFusionBaseline,
    "cross_attention": CrossAttentionTransformer,
    "mamba_fusion": MambaFusion,
    "mamba_dual_head": MambaDualHead,
    "mamba_v2_fusion": MambaV2Fusion,
    "mamba_v2_dual_head": MambaV2DualHead,
}

# ──────────────────────────────────────────────────────────────
# Config key → constructor kwarg mapping
# Maps each possible kwarg name to where it lives in the config.
# build_model() only passes kwargs that the model actually accepts.
# ──────────────────────────────────────────────────────────────

_KWARG_SOURCES: dict[str, tuple[str, ...]] = {
    # kwarg name         → (config section, key)
    "text_encoder":       (),                          # passed directly
    "audio_encoder":      (),                          # passed directly
    "num_classes":        ("dataset", "num_classes"),
    "hidden_dim":         ("model", "hidden_dim"),
    "dropout":            ("model", "dropout"),
    "num_heads":          ("model", "num_heads"),
    "num_layers":         ("model", "num_layers"),
    "dim_feedforward":    ("model", "dim_feedforward"),
    "num_cross_layers":   ("model", "num_cross_layers"),
    "mamba_config":       ("mamba",),
    "mamba_v2_config":    ("mamba_v2",),
    "va_config":          ("va_head",),
}


def _resolve_kwarg(key: str, cfg: dict, direct_kwargs: dict):
    """Resolve a single kwarg value from config or direct kwargs."""
    # Direct kwargs (text_encoder, audio_encoder) take priority
    if key in direct_kwargs:
        return direct_kwargs[key]
    
    path = _KWARG_SOURCES.get(key)
    if path is None or len(path) == 0:
        return None
    
    # Walk the config path
    value = cfg
    for segment in path:
        if isinstance(value, dict) and segment in value:
            value = value[segment]
        else:
            return None  # not found in config
    
    return value


def build_encoders(cfg: dict) -> tuple[TextEncoder, AudioEncoder]:
    """Build text and audio encoders from config.

    Supports two audio encoder backends, selected by cfg["audio_encoder"]["type"]:
        "wav2vec2"  (default) — frozen Wav2Vec2-base-960h, 320× downsample, T≈550
        "raw_conv"            — trainable Conv1d stack, configurable downsample, T≈2750

    Args:
        cfg: Full config dict (parsed from YAML)

    Returns:
        (text_encoder, audio_encoder) tuple
    """
    text_encoder = TextEncoder(
        model_id=cfg["text_encoder"]["model_id"],
        freeze=cfg["text_encoder"]["freeze"],
        pooling=cfg["text_encoder"]["pooling"],
        output_dim=cfg["text_encoder"]["output_dim"],
    )

    encoder_type = cfg["audio_encoder"].get("type", "wav2vec2")

    if encoder_type == "raw_conv":
        from models.encoders_raw import RawAudioEncoder
        audio_encoder = RawAudioEncoder(
            output_dim=cfg["audio_encoder"]["output_dim"],
            stride_config=cfg["audio_encoder"].get("stride_config", [4, 4, 4]),
            channels=cfg["audio_encoder"].get("channels", [64, 256, 512]),
            dropout=cfg["audio_encoder"].get("dropout", 0.1),
            freeze=cfg["audio_encoder"].get("freeze", False),
            checkpoint=cfg["audio_encoder"].get("checkpoint", None),
        )
        logger.info(
            f"AudioEncoder: raw_conv (frame_stride={audio_encoder.frame_stride}×, "
            f"trainable, freeze={cfg['audio_encoder'].get('freeze', False)})"
        )
    else:
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
) -> nn.Module:
    """Build model from config using automatic kwarg dispatch.
    
    Inspects the model class __init__ signature and only passes
    kwargs that the class actually accepts. No if-statements needed.
    
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
    
    # Inspect what kwargs the model class accepts
    sig = inspect.signature(model_cls.__init__)
    accepted_params = set(sig.parameters.keys()) - {"self"}
    
    # Direct kwargs (not from config)
    direct = {
        "text_encoder": text_encoder,
        "audio_encoder": audio_encoder,
    }
    
    # Build kwargs dict: only include what the model accepts
    kwargs = {}
    for param_name in accepted_params:
        value = _resolve_kwarg(param_name, cfg, direct)
        if value is not None:
            kwargs[param_name] = value
    
    # Build model
    model = model_cls(**kwargs)
    
    trainable = model.get_trainable_params()
    total = model.get_total_params()
    logger.info(
        f"Model '{model_name}' built: "
        f"{trainable:,} trainable / {total:,} total params "
        f"({trainable / total * 100:.1f}% trainable)"
    )
    
    return model


__all__ = ["build_encoders", "build_model", "MODEL_REGISTRY"]