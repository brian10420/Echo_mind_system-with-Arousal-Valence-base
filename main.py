#!/usr/bin/env python3
"""
Echo-Mind System — Main Entry Point
====================================
Multimodal Emotion Recognition with Mamba-Transformer Hybrid

Usage:
    # Full 5-fold LOSO evaluation
    python main.py --config configs/base.yaml
    
    # Single fold for quick testing
    python main.py --config configs/base.yaml --fold 0
    
    # Override model from command line
    python main.py --config configs/base.yaml --model late_fusion
"""

import os
import sys
import yaml
import logging
import argparse
import time
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import IEMOCAPDataset
from data.collator import MultimodalCollator
from data.splitter import LOSOSplitter
from models import build_encoders, build_model
from engine.trainer import Trainer
from engine.evaluator import Evaluator, EvalResult


def setup_logging(output_dir: str):
    """Configure logging to both console and file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(output_dir) / "training.log"),
        ],
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def run_fold(
    cfg: dict,
    train_df,
    test_df,
    fold_idx: int,
    text_encoder,
    audio_encoder,
    collator,
    device: torch.device,
) -> EvalResult:
    """Run training and evaluation for a single LOSO fold.
    
    Args:
        cfg: Config dict
        train_df: Training DataFrame
        test_df: Test DataFrame
        fold_idx: Fold index (0-based)
        text_encoder: Shared frozen text encoder
        audio_encoder: Shared frozen audio encoder
        collator: Shared collator
        device: Torch device
        
    Returns:
        EvalResult for this fold
    """
    logger = logging.getLogger(__name__)
    
    # --- Build datasets ---
    train_dataset = IEMOCAPDataset(
        df=train_df,
        max_audio_sec=cfg["dataset"]["max_audio_sec"],
        sample_rate=cfg["dataset"]["audio_sample_rate"],
    )
    test_dataset = IEMOCAPDataset(
        df=test_df,
        max_audio_sec=cfg["dataset"]["max_audio_sec"],
        sample_rate=cfg["dataset"]["audio_sample_rate"],
    )
    
    # --- Build dataloaders ---
    hw_cfg = cfg["hardware"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=hw_cfg["num_workers"],
        pin_memory=hw_cfg["pin_memory"],
        collate_fn=collator,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=hw_cfg["num_workers"],
        pin_memory=hw_cfg["pin_memory"],
        collate_fn=collator,
        drop_last=False,
    )
    
    logger.info(
        f"Fold {fold_idx + 1}: "
        f"train={len(train_dataset)} ({len(train_loader)} batches), "
        f"test={len(test_dataset)} ({len(test_loader)} batches)"
    )
    
    # --- Build model (fresh for each fold) ---
    model = build_model(cfg, text_encoder, audio_encoder)
    
    # --- Train ---
    trainer = Trainer(
        model=model,
        cfg=cfg,
        device=device,
        fold_idx=fold_idx,
    )
    
    best_result = trainer.fit(train_loader, test_loader)
    
    # --- Final evaluation with best model ---
    final_result = trainer.evaluate(test_loader)
    
    # Print per-class report
    logger.info(f"\nFold {fold_idx + 1} — Final Classification Report:")
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = trainer._to_device(batch)
            outputs = model(batch)
            all_preds.extend(outputs["logits"].argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    
    trainer.evaluator.print_report(all_preds, all_labels)
    
    return final_result


def main():
    parser = argparse.ArgumentParser(description="Echo-Mind Multimodal Emotion Recognition")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run single fold (0-4). If None, run all 5 folds.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name from config")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    
    args = parser.parse_args()
    
    # --- Load config ---
    cfg = load_config(args.config)
    
    # --- Apply overrides ---
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    
    # --- Setup ---
    output_dir = Path(cfg["paths"]["output_dir"]) / cfg["model"]["name"]
    cfg["paths"]["output_dir"] = str(output_dir)
    
    setup_logging(str(output_dir))
    logger = logging.getLogger(__name__)
    
    set_seed(cfg["hardware"]["seed"])
    device = torch.device(cfg["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 70)
    logger.info("Echo-Mind System — Multimodal Emotion Recognition")
    logger.info("=" * 70)
    logger.info(f"Model:    {cfg['model']['name']}")
    logger.info(f"Device:   {device}")
    logger.info(f"Epochs:   {cfg['training']['epochs']}")
    logger.info(f"Batch:    {cfg['training']['batch_size']}")
    logger.info(f"LR:       {cfg['training']['learning_rate']}")
    logger.info(f"Output:   {output_dir}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU:      {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # --- Build shared components (loaded once, used by all folds) ---
    logger.info("\nLoading pretrained encoders (this may take a moment)...")
    t_start = time.time()
    
    text_encoder, audio_encoder = build_encoders(cfg)
    text_encoder = text_encoder.to(device)
    audio_encoder = audio_encoder.to(device)
    
    logger.info(f"Encoders loaded in {time.time() - t_start:.1f}s")
    
    # --- Build collator ---
    collator = MultimodalCollator(
        text_model_id=cfg["text_encoder"]["model_id"],
        audio_model_id=cfg["audio_encoder"]["model_id"],
        max_text_tokens=cfg["dataset"]["max_text_tokens"],
        audio_sample_rate=cfg["dataset"]["audio_sample_rate"],
    )
    
    # --- Build LOSO splitter ---
    csv_path = (
        Path(cfg["paths"]["iemocap_root"]).parent.parent
        / "Echo_mind_system-with-Arousal-Valence-base"
        / "tools" / "iemocap_explorer" / "outputs" / "iemocap_utterances.csv"
    )
    # Also check local path
    local_csv = Path("tools/iemocap_explorer/outputs/iemocap_utterances.csv")
    if local_csv.exists():
        csv_path = local_csv
    
    logger.info(f"Loading dataset from: {csv_path}")
    
    splitter = LOSOSplitter(
        csv_path=csv_path,
        label_map=cfg["dataset"]["label_map"],
        num_folds=cfg["evaluation"]["loso_folds"],
    )
    logger.info(f"\n{splitter.summary()}\n")
    
    # --- Run LOSO ---
    fold_results = []
    
    if args.fold is not None:
        # Single fold mode
        folds_to_run = [args.fold]
        logger.info(f"Running single fold: {args.fold}")
    else:
        folds_to_run = list(range(cfg["evaluation"]["loso_folds"]))
        logger.info(f"Running all {len(folds_to_run)} LOSO folds")
    
    for fold_idx in folds_to_run:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"FOLD {fold_idx + 1}/{cfg['evaluation']['loso_folds']}")
        logger.info(f"{'=' * 70}")
        
        train_df, test_df = splitter.get_fold(fold_idx)
        
        result = run_fold(
            cfg=cfg,
            train_df=train_df,
            test_df=test_df,
            fold_idx=fold_idx,
            text_encoder=text_encoder,
            audio_encoder=audio_encoder,
            collator=collator,
            device=device,
        )
        
        fold_results.append(result)
        logger.info(f"Fold {fold_idx + 1} DONE: {result.summary()}")
    
    # --- Aggregate results ---
    if len(fold_results) > 1:
        logger.info(f"\n{'=' * 70}")
        logger.info("LOSO CROSS-VALIDATION RESULTS")
        logger.info(f"{'=' * 70}")
        
        was = [r.wa for r in fold_results]
        uas = [r.ua for r in fold_results]
        f1s = [r.f1_macro for r in fold_results]
        
        for i, r in enumerate(fold_results):
            logger.info(f"  Fold {i + 1}: {r.summary()}")
        
        logger.info(f"\n  Mean WA:       {np.mean(was):.4f} ± {np.std(was):.4f}")
        logger.info(f"  Mean UA:       {np.mean(uas):.4f} ± {np.std(uas):.4f}")
        logger.info(f"  Mean F1-Macro: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        
        # Save summary
        summary_path = output_dir / "loso_results.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Model: {cfg['model']['name']}\n")
            f.write(f"Mean WA: {np.mean(was):.4f} ± {np.std(was):.4f}\n")
            f.write(f"Mean UA: {np.mean(uas):.4f} ± {np.std(uas):.4f}\n")
            f.write(f"Mean F1-Macro: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")
            for i, r in enumerate(fold_results):
                f.write(f"Fold {i + 1}: {r.summary()}\n")
        logger.info(f"\nResults saved to {summary_path}")
    
    logger.info("\nDONE.")


if __name__ == "__main__":
    main()
