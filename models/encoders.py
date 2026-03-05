"""
Frozen Encoders — BERT + Wav2Vec2.0 wrappers
=============================================
These encoders are shared across ALL models (baseline, Transformer, Mamba).
They are frozen (no gradient) to save VRAM and ensure fair comparison.
Only the downstream layers (fusion, classification) are trained.
"""

import logging

import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2Model

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Frozen BERT encoder for text features.
    
    Args:
        model_id: HuggingFace model ID (e.g., "bert-base-uncased")
        freeze: Whether to freeze all parameters
        pooling: "cls" for [CLS] token, "mean" for mean pooling
        output_dim: Expected output dimension (768 for bert-base)
    """
    
    def __init__(
        self,
        model_id: str = "bert-base-uncased",
        freeze: bool = True,
        pooling: str = "cls",
        output_dim: int = 768,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        self.pooling = pooling
        self.output_dim = output_dim
        
        if freeze:
            self._freeze()
            logger.info(f"TextEncoder: {model_id} loaded and FROZEN")
        else:
            logger.info(f"TextEncoder: {model_id} loaded (trainable)")
    
    def _freeze(self):
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, seq_len) token IDs
            attention_mask: (B, seq_len) attention mask
            
        Returns:
            pooled: (B, output_dim) pooled text representation
            sequence: (B, seq_len, output_dim) full sequence output
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        sequence = outputs.last_hidden_state  # (B, seq_len, 768)
        
        if self.pooling == "cls":
            pooled = sequence[:, 0, :]  # (B, 768)
        elif self.pooling == "mean":
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
            pooled = (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return pooled, sequence
    
    def train(self, mode: bool = True):
        """Override to keep encoder in eval mode when frozen."""
        super().train(mode)
        self.model.eval()
        return self


class AudioEncoder(nn.Module):
    """Frozen Wav2Vec2.0 encoder for audio features.
    
    Args:
        model_id: HuggingFace model ID
        freeze: Whether to freeze all parameters
        output_dim: Expected output dimension (768 for wav2vec2-base)
    """
    
    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-base-960h",
        freeze: bool = True,
        output_dim: int = 768,
    ):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_id)
        self.output_dim = output_dim
        
        if freeze:
            self._freeze()
            logger.info(f"AudioEncoder: {model_id} loaded and FROZEN")
        else:
            logger.info(f"AudioEncoder: {model_id} loaded (trainable)")
    
    def _freeze(self):
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def forward(
        self,
        audio_input: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_input: (B, num_samples) raw waveform
            attention_mask: (B, num_samples) mask (True=valid)
            
        Returns:
            pooled: (B, output_dim) mean-pooled audio representation
            sequence: (B, T, output_dim) frame-level features
                      T ≈ num_samples / 320 (Wav2Vec2.0 downsampling factor)
        """
        with torch.no_grad():
            outputs = self.model(
                input_values=audio_input,
                attention_mask=attention_mask.long(),
            )
        
        sequence = outputs.last_hidden_state  # (B, T, 768)
        
        # Compute frame-level attention mask (vectorized)
        frame_mask = self._compute_frame_mask(attention_mask, sequence.shape[1])
        
        # Masked mean pooling
        mask = frame_mask.unsqueeze(-1).float()  # (B, T, 1)
        pooled = (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, 768)
        
        return pooled, sequence
    
    def _compute_frame_mask(
        self, sample_mask: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """Compute frame-level mask from sample-level mask (vectorized).
        
        Wav2Vec2.0 downsamples ~320x, so:
            valid_frames ≈ valid_samples / 320
        
        Args:
            sample_mask: (B, num_samples) True = valid
            num_frames: Number of output frames T
            
        Returns:
            (B, T) True = valid frame
        """
        valid_samples = sample_mask.sum(dim=1)  # (B,)
        valid_frames = (valid_samples / 320).long().clamp(min=1, max=num_frames)
        
        # Vectorized: broadcast arange (1, T) < valid_frames (B, 1) → (B, T)
        frame_indices = torch.arange(num_frames, device=sample_mask.device).unsqueeze(0)
        frame_mask = frame_indices < valid_frames.unsqueeze(1)
        
        return frame_mask
    
    def train(self, mode: bool = True):
        """Override to keep encoder in eval mode when frozen."""
        super().train(mode)
        self.model.eval()
        return self