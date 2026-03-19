"""
Baseline Benchmark Runner
=========================
Trains all models on a given config sequentially and prints a comparison table.

Usage:
    # Run all models on long_seq.yaml
    uv run run_baselines.py --config configs/long_seq.yaml

    # Run a subset
    uv run run_baselines.py --config configs/long_seq.yaml --models mamba_fusion mamba_v2_fusion

    # Quick smoke-test (1 fold, 3 epochs)
    uv run run_baselines.py --config configs/long_seq.yaml --fold 0 --epochs 3

    # Re-parse results from a previous run without retraining
    uv run run_baselines.py --config configs/long_seq.yaml --parse-only
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODELS = [
    "late_fusion",
    "cross_attention",
    "mamba_fusion",
    "mamba_dual_head",
    "mamba_v2_fusion",
    "mamba_v2_dual_head",
]

# Transformer self-attention is O(T²): at T≈2750 batch=16 kills VRAM.
# These per-model overrides are applied on top of the config when using long sequences.
LONG_SEQ_BATCH_OVERRIDES = {
    "cross_attention": 4,   # T²=7.5M per head; 4×6layers fits in 24GB
    "late_fusion":     8,   # pooling before MLP, still benefits from lower batch
}


# ─────────────────────────────────────────────────────────────────────────────
# Result parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_loso_results(results_path: Path) -> dict | None:
    """Parse loso_results.txt into a dict of metrics.

    Falls back to computing means from per-fold lines when Mean lines are
    absent (e.g. single-fold runs with --fold N).
    """
    if not results_path.exists():
        return None

    text = results_path.read_text()
    result = {}

    for line in text.splitlines():
        m = re.match(r"Mean ([\w-]+): ([\d.]+) ± ([\d.]+)", line)
        if m:
            key = m.group(1).lower().replace("-", "_")   # F1-Macro → f1_macro
            result[f"mean_{key}"] = float(m.group(2))
            result[f"std_{key}"] = float(m.group(3))

        m = re.match(r"Fold (\d+): WA=([\d.]+) \| UA=([\d.]+) \| F1m=([\d.]+)", line)
        if m:
            fold = int(m.group(1))
            result.setdefault("folds", {})[fold] = {
                "wa": float(m.group(2)),
                "ua": float(m.group(3)),
                "f1m": float(m.group(4)),
            }

    if not result:
        return None

    # Compute means from fold data if Mean lines were not written (single-fold run)
    if "mean_ua" not in result and "folds" in result:
        folds = list(result["folds"].values())
        for key in ("wa", "ua", "f1m"):
            vals = [f[key] for f in folds]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            out_key = "f1_macro" if key == "f1m" else key
            result[f"mean_{out_key}"] = mean
            result[f"std_{out_key}"] = std

    return result


def get_output_dir(cfg_path: Path, model_name: str) -> Path:
    """Resolve output directory for a model from config."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg["paths"]["output_dir"])
    return base / model_name


# ─────────────────────────────────────────────────────────────────────────────
# Training runner
# ─────────────────────────────────────────────────────────────────────────────

def run_model(
    cfg_path: Path,
    model: str,
    extra_args: list[str],
) -> tuple[int, float]:
    """Run main.py for one model. Returns (returncode, elapsed_seconds)."""
    cmd = [
        "uv", "run", "main.py",
        "--config", str(cfg_path),
        "--model", model,
        *extra_args,
    ]
    print(f"\n{'='*60}")
    print(f"  Training: {model}")
    print(f"  Command:  {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - t0

    return proc.returncode, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(records: list[dict]) -> None:
    """Print aligned comparison table."""
    if not records:
        print("No results to display.")
        return

    header = f"{'Model':<20} {'UA':>8} {'±':>6} {'WA':>8} {'±':>6} {'F1-Mac':>8} {'±':>6} {'Time':>10}"
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    # Sort by mean UA descending (None results go last)
    records = sorted(
        records,
        key=lambda r: r.get("mean_ua", -1),
        reverse=True,
    )

    for r in records:
        model = r["model"]
        if r.get("mean_ua") is not None:
            ua   = f"{r['mean_ua']:.4f}"
            ua_s = f"{r['std_ua']:.4f}"
            wa   = f"{r['mean_wa']:.4f}"
            wa_s = f"{r['std_wa']:.4f}"
            f1   = f"{r['mean_f1_macro']:.4f}"
            f1_s = f"{r['std_f1_macro']:.4f}"
        else:
            ua = wa = f1 = "FAILED"
            ua_s = wa_s = f1_s = "—"

        elapsed = r.get("elapsed")
        time_str = f"{elapsed/60:.1f} min" if elapsed else "—"

        print(f"{model:<20} {ua:>8} {ua_s:>6} {wa:>8} {wa_s:>6} {f1:>8} {f1_s:>6} {time_str:>10}")

    print(sep)

    # Best model
    best = next((r for r in records if r.get("mean_ua") is not None), None)
    if best:
        print(f"\n  Best UA: {best['model']}  →  {best['mean_ua']:.4f} ± {best['std_ua']:.4f}")

    print(f"{'='*len(header)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run all baselines and compare results.")
    parser.add_argument("--config", default="configs/long_seq.yaml", help="Config YAML path")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        metavar="MODEL", help=f"Models to run (default: all). Choices: {ALL_MODELS}")
    parser.add_argument("--fold", type=int, default=None, help="Single fold (skip others)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for all models")
    parser.add_argument("--long-seq", action="store_true",
                        help="Apply per-model batch size overrides for long sequences (T≈2750)")
    parser.add_argument("--parse-only", action="store_true",
                        help="Skip training; just parse existing loso_results.txt files")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    # Extra args forwarded to main.py
    extra_args = []
    if args.fold is not None:
        extra_args += ["--fold", str(args.fold)]
    if args.epochs is not None:
        extra_args += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        extra_args += ["--batch-size", str(args.batch_size)]

    records = []
    failed = []

    for model in args.models:
        out_dir = get_output_dir(cfg_path, model)
        results_path = out_dir / "loso_results.txt"

        elapsed = None

        if not args.parse_only:
            model_args = list(extra_args)
            if args.long_seq and model in LONG_SEQ_BATCH_OVERRIDES and args.batch_size is None:
                model_args += ["--batch-size", str(LONG_SEQ_BATCH_OVERRIDES[model])]
            rc, elapsed = run_model(cfg_path, model, model_args)
            if rc != 0:
                print(f"\n[WARNING] {model} exited with code {rc}. Skipping.\n")
                failed.append(model)
                records.append({"model": model, "elapsed": elapsed})
                continue
        else:
            if not results_path.exists():
                print(f"[SKIP] {model}: no results at {results_path}")
                records.append({"model": model})
                continue

        parsed = parse_loso_results(results_path)
        if parsed:
            records.append({"model": model, "elapsed": elapsed, **parsed})
        else:
            print(f"[WARNING] Could not parse results for {model} at {results_path}")
            records.append({"model": model, "elapsed": elapsed})

    print_summary(records)

    if failed:
        print(f"[WARNING] Failed models: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
