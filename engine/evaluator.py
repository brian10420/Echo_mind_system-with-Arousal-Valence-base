"""
Evaluator — IEMOCAP Standard Metrics
=====================================
Computes all metrics expected in MER papers:
- WA (Weighted Accuracy): standard accuracy
- UA (Unweighted Accuracy): mean per-class accuracy (handles imbalance)
- F1 Macro / Weighted
- Per-class precision, recall, F1
- Confusion matrix
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

CLASS_NAMES_4 = ["Angry", "Happy", "Sad", "Neutral"]
CLASS_NAMES_6 = ["Angry", "Happy", "Excited", "Sad", "Neutral", "Frustrated"]


@dataclass
class EvalResult:
    """Container for evaluation results."""
    wa: float = 0.0           # Weighted Accuracy
    ua: float = 0.0           # Unweighted Accuracy (balanced)
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    per_class: dict = field(default_factory=dict)
    confusion: np.ndarray = None
    loss: float = 0.0
    
    def summary(self, prefix: str = "") -> str:
        """One-line summary for logging."""
        return (
            f"{prefix}WA={self.wa:.4f} | UA={self.ua:.4f} | "
            f"F1m={self.f1_macro:.4f} | F1w={self.f1_weighted:.4f} | "
            f"Loss={self.loss:.4f}"
        )


class Evaluator:
    """Computes standard IEMOCAP evaluation metrics.
    
    Args:
        num_classes: Number of emotion classes
        class_names: List of class names for reports
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        class_names: list[str] = None,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or (
            CLASS_NAMES_4 if num_classes == 4 else CLASS_NAMES_6
        )
    
    def compute(
        self,
        all_preds: list[int],
        all_labels: list[int],
        all_losses: list[float] = None,
    ) -> EvalResult:
        """Compute all metrics from predictions and labels.
        
        Args:
            all_preds: List of predicted class indices
            all_labels: List of ground truth class indices
            all_losses: Optional list of per-batch losses
            
        Returns:
            EvalResult with all metrics
        """
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        
        result = EvalResult(
            wa=accuracy_score(labels, preds),
            ua=balanced_accuracy_score(labels, preds),
            f1_macro=f1_score(labels, preds, average='macro', zero_division=0),
            f1_weighted=f1_score(labels, preds, average='weighted', zero_division=0),
            confusion=confusion_matrix(labels, preds, labels=list(range(self.num_classes))),
            loss=np.mean(all_losses) if all_losses else 0.0,
        )
        
        # Per-class metrics
        report = classification_report(
            labels, preds,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        result.per_class = {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in self.class_names
            if name in report
        }
        
        return result
    
    def print_report(
        self,
        all_preds: list[int],
        all_labels: list[int],
    ) -> str:
        """Print full classification report."""
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        
        report = classification_report(
            labels, preds,
            target_names=self.class_names,
            zero_division=0,
        )
        print(report)
        return report
    
    def save_confusion_matrix(
        self,
        result: EvalResult,
        save_path: Path,
        title: str = "Confusion Matrix",
    ):
        """Save confusion matrix as PNG."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = result.confusion
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(
            xticks=np.arange(self.num_classes),
            yticks=np.arange(self.num_classes),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ylabel='True Label',
            xlabel='Predicted Label',
            title=title,
        )
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(
                    j, i,
                    f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=9,
                )
        
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Confusion matrix saved to {save_path}")
