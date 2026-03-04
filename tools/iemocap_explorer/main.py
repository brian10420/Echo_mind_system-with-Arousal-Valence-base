#!/usr/bin/env python3
"""
IEMOCAP Explorer - Main Entry Point
====================================
Run this script to generate a full dataset analysis report.

Usage:
    cd /data/Brian/Echo_mind_system-with-Arousal-Valence-base/tools/iemocap_explorer
    python main.py

Outputs:
    - Console report with all statistics
    - PNG plots in ./outputs/
    - CSV export of all parsed utterances in ./outputs/
    - Text report saved to ./outputs/report.txt
"""

import sys
import logging
import argparse
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

from config import IEMOCAP_ROOT, OUTPUT_DIR
from parser import parse_iemocap
from statistics import compute_statistics, print_report
from visualizer import generate_all_plots


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="IEMOCAP Dataset Explorer — Analyze before you train"
    )
    parser.add_argument(
        "--iemocap-root", type=str, default=str(IEMOCAP_ROOT),
        help="Path to IEMOCAP_full_release directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Directory to save plots and reports"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Skip audio file scanning (faster, no duration stats)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (text report only)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    iemocap_root = Path(args.iemocap_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # Step 1: Validate paths
    # ============================================================
    logger.info(f"IEMOCAP root: {iemocap_root}")
    if not iemocap_root.exists():
        logger.error(f"IEMOCAP directory not found: {iemocap_root}")
        logger.error("Please update IEMOCAP_ROOT in config.py or use --iemocap-root flag")
        sys.exit(1)
    
    # Quick sanity check
    session1 = iemocap_root / "Session1"
    if not session1.exists():
        logger.error(f"Session1 not found in {iemocap_root}. Is this the right path?")
        sys.exit(1)
    
    # ============================================================
    # Step 2: Parse entire dataset
    # ============================================================
    logger.info("=" * 50)
    logger.info("STEP 1/4: Parsing IEMOCAP dataset...")
    logger.info("=" * 50)
    
    utterances = parse_iemocap(
        root=iemocap_root,
        include_audio=not args.no_audio,
        include_transcripts=True,
    )
    
    if not utterances:
        logger.error("No utterances found! Check evaluation file paths.")
        # Debug: show what's in the session directories
        for session in ["Session1", "Session2"]:
            session_path = iemocap_root / session / "dialog"
            if session_path.exists():
                logger.info(f"Contents of {session_path}:")
                for item in sorted(session_path.iterdir()):
                    logger.info(f"  {item.name}/")
        sys.exit(1)
    
    logger.info(f"Successfully parsed {len(utterances)} utterances")
    
    # ============================================================
    # Step 3: Compute statistics
    # ============================================================
    logger.info("=" * 50)
    logger.info("STEP 2/4: Computing statistics...")
    logger.info("=" * 50)
    
    stats = compute_statistics(utterances)
    
    # ============================================================
    # Step 4: Print report
    # ============================================================
    logger.info("=" * 50)
    logger.info("STEP 3/4: Generating report...")
    logger.info("=" * 50)
    
    report = print_report(stats)
    
    # Save report to file
    report_path = output_dir / "report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    # ============================================================
    # Step 5: Export CSV
    # ============================================================
    if stats.df is not None:
        csv_path = output_dir / "iemocap_utterances.csv"
        stats.df.to_csv(csv_path, index=False)
        logger.info(f"CSV export saved to {csv_path}")
    
    # ============================================================
    # Step 6: Generate plots
    # ============================================================
    if not args.no_plots:
        logger.info("=" * 50)
        logger.info("STEP 4/4: Generating plots...")
        logger.info("=" * 50)
        
        generate_all_plots(stats, output_dir)
    
    # ============================================================
    # Summary
    # ============================================================
    logger.info("=" * 50)
    logger.info("DONE! All outputs saved to:")
    logger.info(f"  {output_dir}")
    logger.info("")
    logger.info("Files generated:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        logger.info(f"  {f.name:45s} ({size_kb:.1f} KB)")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()