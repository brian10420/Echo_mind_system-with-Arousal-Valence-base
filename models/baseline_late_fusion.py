"""
Late Fusion Baseline (Model C)
==============================
Simplest multimodal model:
  Audio → Wav2Vec2.0 (frozen) → mean pool → 768-dim
  Text  → BERT (frozen)       → [CLS]     → 768-dim
  Concat → MLP → 4 classes

This establishes the performance floor for multimodal approaches.
"""

import torch
import torch.nn as nn

from models.encoders import TextEncoder, AudioEncoder


class LateFusionBaseline(nn.Module):
    """Late fusion baseline: independent encoding + concat + MLP.
    
    Args:
        text_encoder: Frozen BERT encoder
        audio_encoder: Frozen Wav2Vec2.0 encoder
        num_classes: Number of emotion classes
        hidden_dim: Hidden layer dimension in MLP
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        text_encoder: TextEncoder,
        audio_encoder: AudioEncoder,
        num_classes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        
        # Input dim = text_dim + audio_dim
        input_dim = text_encoder.output_dim + audio_encoder.output_dim  # 768 + 768 = 1536
        
        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: Dict from MultimodalCollator with keys:
                - audio_input, audio_attention_mask
                - text_input_ids, text_attention_mask
                - labels
                
        Returns:
            Dict with:
                - logits: (B, num_classes) classification logits
                - loss: scalar loss (if labels provided)
        """
        # --- Encode ---
        text_pooled, _ = self.text_encoder(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
        )
        
        audio_pooled, _ = self.audio_encoder(
            audio_input=batch["audio_input"],
            attention_mask=batch["audio_attention_mask"],
        )
        
        # --- Fuse (simple concatenation) ---
        fused = torch.cat([text_pooled, audio_pooled], dim=-1)  # (B, 1536)
        
        # --- Classify ---
        logits = self.classifier(fused)  # (B, num_classes)
        
        return {"logits": logits}
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters (excludes frozen encoders)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count all parameters including frozen encoders."""
        return sum(p.numel() for p in self.parameters())
