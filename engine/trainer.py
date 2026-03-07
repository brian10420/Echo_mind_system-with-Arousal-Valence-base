"""
Trainer — Shared training loop for all models
==============================================
Handles: training loop, validation, AMP, gradient clipping,
learning rate scheduling, early stopping, checkpointing.

Supports:
- Per-module learning rates via model.get_param_groups()
- Multi-task loss for dual-head models (CE + KL)
"""

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from engine.evaluator import Evaluator, EvalResult

logger = logging.getLogger(__name__)


class Trainer:
    """Universal trainer for all Echo-Mind models.
    
    Args:
        model: Any model that accepts batch dict and returns {"logits": tensor}
        cfg: Full config dict
        device: torch device
        fold_idx: Current LOSO fold index (for logging/saving)
    """
    
    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: torch.device,
        fold_idx: int = 0,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.fold_idx = fold_idx
        
        train_cfg = cfg["training"]
        
        # Loss function with class weights
        class_weights = torch.tensor(
            train_cfg.get("class_weights", [1.0, 1.0, 1.0, 1.0]),
            dtype=torch.float,
        ).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Multi-task loss weight (for dual-head models)
        # total_loss = (1 - va_weight) * CE + va_weight * KL
        va_cfg = cfg.get("va_head", {})
        self.va_loss_weight = va_cfg.get("loss_weight", 0.3)
        self.use_multitask = cfg["model"]["name"] == "mamba_dual_head"
        
        if self.use_multitask:
            logger.info(
                f"Multi-task loss: {1 - self.va_loss_weight:.1f} × CE + "
                f"{self.va_loss_weight:.1f} × KL"
            )
        
        # Optimizer — supports per-module learning rates
        base_lr = train_cfg["learning_rate"]
        
        if hasattr(model, 'get_param_groups'):
            param_groups = model.get_param_groups(base_lr=base_lr)
            logger.info(f"Using per-module learning rates ({len(param_groups)} groups)")
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=train_cfg["weight_decay"],
            )
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=base_lr,
                weight_decay=train_cfg["weight_decay"],
            )
        
        # Scheduler
        self.epochs = train_cfg["epochs"]
        self.scheduler = self._build_scheduler(train_cfg)
        
        # AMP
        self.use_amp = cfg["hardware"].get("mixed_precision", True)
        self.scaler = GradScaler("cuda") if self.use_amp else None
        
        # Gradient clipping
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        
        # Early stopping
        self.patience = train_cfg.get("early_stopping_patience", 7)
        self.best_ua = 0.0
        self.patience_counter = 0
        self.best_state = None
        
        # Evaluator
        self.evaluator = Evaluator(
            num_classes=cfg["dataset"]["num_classes"]
        )
        
        # Output directory
        self.output_dir = Path(cfg["paths"]["output_dir"]) / f"fold_{fold_idx}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _build_scheduler(self, train_cfg: dict):
        """Build learning rate scheduler."""
        scheduler_type = train_cfg.get("scheduler", "cosine")
        
        if scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-7)
        elif scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR
            return StepLR(self.optimizer, step_size=10, gamma=0.5)
        else:
            return None
    
    def _compute_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        """Compute loss — supports both single-task and multi-task.
        
        Single-task (late_fusion, cross_attention, mamba_fusion):
            loss = CrossEntropy(logits, labels)
            
        Multi-task (mamba_dual_head):
            loss = (1-α) × CrossEntropy + α × KL-divergence
        """
        ce_loss = self.criterion(outputs["logits"], batch["labels"])
        
        if self.use_multitask and "va_loss" in outputs:
            va_loss = outputs["va_loss"]
            total_loss = (1 - self.va_loss_weight) * ce_loss + self.va_loss_weight * va_loss
            return total_loss
        
        return ce_loss
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast("cuda"):
                    outputs = self.model(batch)
                    loss = self._compute_loss(outputs, batch)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.gradient_clip,
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                loss = self._compute_loss(outputs, batch)
                loss.backward()
                
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.gradient_clip,
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> EvalResult:
        """Run evaluation on a dataloader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_losses = []
        
        for batch in dataloader:
            batch = self._to_device(batch)
            
            if self.use_amp:
                with autocast("cuda"):
                    outputs = self.model(batch)
                    loss = self._compute_loss(outputs, batch)
            else:
                outputs = self.model(batch)
                loss = self._compute_loss(outputs, batch)
            
            preds = outputs["logits"].argmax(dim=-1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_losses.append(loss.item())
        
        result = self.evaluator.compute(all_preds, all_labels, all_losses)
        return result
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> EvalResult:
        """Full training loop with early stopping."""
        logger.info(f"Starting training: {self.epochs} epochs, fold {self.fold_idx}")
        best_result = None
        
        for epoch in range(1, self.epochs + 1):
            t_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_result = self.evaluate(val_loader)
            
            t_elapsed = time.time() - t_start
            lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Fold {self.fold_idx + 1} | Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"{val_result.summary('Val: ')} | "
                f"LR: {lr:.2e} | Time: {t_elapsed:.1f}s"
            )
            
            # Early stopping on UA
            if val_result.ua > self.best_ua:
                self.best_ua = val_result.ua
                self.patience_counter = 0
                best_result = val_result
                
                # Save best model state
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                
                if self.cfg["evaluation"].get("save_best_model", True):
                    ckpt_path = self.output_dir / "best_model.pt"
                    torch.save(self.best_state, ckpt_path)
                    logger.info(f"  ★ New best UA={self.best_ua:.4f}, saved to {ckpt_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(
                        f"  Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break
        
        # Load best model for final evaluation
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        # Save confusion matrix
        if best_result is not None and self.cfg["evaluation"].get("save_confusion_matrix", True):
            self.evaluator.save_confusion_matrix(
                best_result,
                self.output_dir / "confusion_matrix.png",
                title=f"Fold {self.fold_idx + 1} — {self.cfg['model']['name']}",
            )
        
        return best_result
    
    def _to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        device_batch = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                device_batch[key] = val.to(self.device)
            else:
                device_batch[key] = val
        return device_batch