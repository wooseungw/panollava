#!/usr/bin/env python3
"""CORA model evaluation: inference + 5 metrics + optional LLM judge.

Usage:
  # Full pipeline: load model, run inference, compute metrics
  python scripts/eval.py --checkpoint runs/.../finetune/last.ckpt --test-csv data/test.csv

  # Metrics-only from existing predictions CSV
  python scripts/eval.py --csv-input outputs/predictions.csv

  # With LLM judge (requires OPENAI_API_KEY)
  python scripts/eval.py --csv-input outputs/predictions.csv --llm-judge
  python scripts/eval.py --checkpoint runs/.../last.ckpt --test-csv data/test.csv --llm-judge
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    checkpoint: str,
    test_csv: str,
    output_dir: Path,
    batch_size: int = 1,
    save_every: int = 500,
) -> Path:
    """Load model from checkpoint, generate predictions for test CSV, save to CSV.

    Groups queries by image to avoid re-computing panoramic tile projections.
    Saves incremental checkpoints every ``save_every`` samples so that progress
    is not lost on crashes.  On restart the existing partial predictions are
    loaded and already-processed samples are skipped.
    """
    import time as _time

    from PIL import Image as _Image

    from cora.inference.generator import PanoramaGenerator

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    partial_path = output_dir / "predictions_partial.csv"

    # ── Resume from partial checkpoint if available ──
    completed_ids: set[int] = set()
    predictions: list[dict] = []
    if partial_path.exists():
        prev = pd.read_csv(partial_path)
        predictions = prev.to_dict("records")
        completed_ids = {int(r["sample_id"]) for r in predictions}
        logger.info(
            "Resuming from partial checkpoint: %d samples already done",
            len(completed_ids),
        )

    logger.info("Loading model from %s", checkpoint)
    generator = PanoramaGenerator(checkpoint_path=checkpoint)

    logger.info("Loading test data from %s", test_csv)
    df = pd.read_csv(test_csv)

    # Determine column names (support multiple conventions)
    image_col = _find_col(df, "url", "image_path", "file_name", "image")
    query_col = _find_col(df, "query", "instruction", "original_query", "prompt")
    ref_col = _find_col(df, "annotation", "reference", "response")

    if image_col is None or query_col is None:
        raise ValueError(
            f"Test CSV must have image + query columns. "
            f"Found columns: {list(df.columns)}"
        )

    # Group by image for pixel_values caching (avoids re-computing E2P tiles)
    groups = df.groupby(image_col, sort=False)
    unique_images = len(groups)
    total = len(df)
    logger.info(
        "Running inference on %d samples (%d unique images, %.1f queries/image)",
        total, unique_images, total / unique_images,
    )

    done = len(completed_ids)
    t_start = _time.time()
    last_save = done

    for img_path, group_df in groups:
        img_path = str(img_path)

        # Skip entire image group if all queries already completed
        pending_rows = [
            (i, row) for i, row in group_df.iterrows()
            if i not in completed_ids
        ]
        if not pending_rows:
            continue

        # Compute pixel_values ONCE per image
        try:
            pil_image = _Image.open(img_path).convert("RGB")
            pixel_values = generator.processor.image_processor(pil_image)
        except Exception as exc:
            logger.warning("Image processing failed for %s: %s", img_path, exc)
            for i, row in pending_rows:
                predictions.append({
                    "sample_id": i,
                    "image_path": img_path,
                    "query": str(row[query_col]),
                    "prediction": "",
                    "reference": str(row[ref_col]) if ref_col else "",
                })
            done += len(pending_rows)
            continue

        # Generate for each query with cached pixel_values
        for i, row in pending_rows:
            query = str(row[query_col])
            reference = str(row[ref_col]) if ref_col else ""

            try:
                pred = generator.generate_with_pixel_values(pixel_values, query)
            except Exception as exc:
                logger.warning("Inference failed for sample %d: %s", i, exc)
                pred = ""

            predictions.append({
                "sample_id": i,
                "image_path": img_path,
                "query": query,
                "prediction": pred,
                "reference": reference,
            })
            done += 1

            if done % 100 == 0 or done == total:
                elapsed = _time.time() - t_start
                newly_done = done - len(completed_ids)
                if newly_done > 0:
                    rate = elapsed / newly_done
                    eta = rate * (total - done)
                else:
                    eta = 0.0
                logger.info(
                    "  Progress: %d / %d (%.1f%%) — %.1fs elapsed, ~%.0fs ETA",
                    done, total, done / total * 100, elapsed, eta,
                )

            # Incremental save every `save_every` new samples
            if done - last_save >= save_every:
                _save_partial(predictions, partial_path)
                last_save = done

    # Final save
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    logger.info("Predictions saved to %s (%d samples)", predictions_path, len(pred_df))

    # Clean up partial checkpoint
    if partial_path.exists():
        partial_path.unlink()
        logger.info("Removed partial checkpoint %s", partial_path)

    return predictions_path


def _save_partial(predictions: list[dict], path: Path) -> None:
    """Save incremental checkpoint of predictions."""
    pd.DataFrame(predictions).to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("  Saved partial checkpoint (%d samples) → %s", len(predictions), path)


def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(predictions_csv: Path, output_dir: Path) -> dict:
    """Compute BLEU, CIDEr, METEOR, ROUGE-L, SPICE from predictions CSV."""
    from cora.evaluation.metrics import CORAEvaluator

    df = pd.read_csv(predictions_csv)

    pred_col = _find_col(df, "prediction", "output", "caption")
    ref_col = _find_col(df, "reference", "annotation", "response")

    if pred_col is None or ref_col is None:
        logger.error("predictions CSV must have prediction + reference columns")
        return {}

    # Drop rows with empty references
    mask = df[ref_col].notna() & (df[ref_col].astype(str).str.strip() != "")
    valid = df[mask]
    logger.info("Computing metrics on %d / %d samples with references", len(valid), len(df))

    evaluator = CORAEvaluator()
    preds = valid[pred_col].astype(str).tolist()
    refs = valid[ref_col].astype(str).tolist()

    metrics = evaluator.evaluate(preds, refs)

    if metrics:
        evaluator.print_summary(metrics)
        metrics_csv = output_dir / "metrics.csv"
        evaluator.evaluate_and_save(
            preds, refs, metrics_csv,
            experiment_name=output_dir.name,
            precomputed=metrics,  # avoid re-running SPICE
        )
    else:
        logger.warning("Metrics computation returned empty (pycocoevalcap installed?)")

    return metrics


# ---------------------------------------------------------------------------
# LLM Judge integration
# ---------------------------------------------------------------------------


def run_llm_judge(
    predictions_csv: Path,
    output_dir: Path,
    model: str = "gpt-4.1-mini",
    batch_by_image: bool = True,
    max_samples: int | None = None,
) -> dict | None:
    """Run LLM-as-a-Judge evaluation on predictions CSV."""
    import json
    import os

    # Import from sibling script
    scripts_dir = str(Path(__file__).parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from llm_judge_eval import LLMJudge, compute_statistics, normalize_input, print_summary

    if not os.getenv("OPENAI_API_KEY"):
        logger.error(
            "OPENAI_API_KEY not set. Set via .env or export OPENAI_API_KEY='sk-...'"
        )
        return None

    df = normalize_input(predictions_csv)
    if df.empty:
        logger.error("No valid records with references found in %s", predictions_csv)
        return None

    logger.info("Starting LLM judge evaluation (%s, %d records)", model, len(df))
    judge = LLMJudge(
        model=model,
        include_image=True,
        base_path=predictions_csv.parent,
    )

    start = time.time()
    results_df = judge.evaluate_batch(
        df,
        max_samples=max_samples,
        batch_by_image=batch_by_image,
    )
    elapsed = time.time() - start

    judge_csv = output_dir / "llm_judge_scores.csv"
    results_df.to_csv(judge_csv, index=False, encoding="utf-8-sig")
    logger.info("LLM judge results saved to %s (%.1fs)", judge_csv, elapsed)

    stats = compute_statistics(results_df)
    stats["model"] = model
    stats["elapsed_seconds"] = elapsed
    print_summary(stats)

    # Save stats JSON
    stats_path = judge_csv.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("LLM judge stats saved to %s", stats_path)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CORA Evaluation: inference + metrics + optional LLM judge",
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (requires --test-csv)",
    )
    input_group.add_argument(
        "--csv-input", type=str, default=None,
        help="Path to existing predictions CSV (skip inference)",
    )

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to test CSV")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")

    # LLM judge options
    parser.add_argument("--llm-judge", action="store_true", help="Run LLM-as-a-Judge evaluation")
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-mini",
                        help="OpenAI model for LLM judge (default: gpt-4.1-mini)")
    parser.add_argument("--judge-max-samples", type=int, default=None,
                        help="Limit LLM judge to N samples")

    # Metric-only mode
    parser.add_argument("--skip-metrics", action="store_true", help="Skip traditional metrics")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    # Force unbuffered output so tee/pipe see progress in real time
    for handler in logging.root.handlers:
        if hasattr(handler, "stream"):
            handler.stream = sys.stdout

    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = None

    # ── Step 1: Get predictions ──
    if args.checkpoint:
        if not args.test_csv:
            parser.error("--checkpoint requires --test-csv")
        predictions_csv = run_inference(
            args.checkpoint, args.test_csv, output_dir, args.batch_size,
        )
    elif args.csv_input:
        predictions_csv = Path(args.csv_input)
        if not predictions_csv.exists():
            raise SystemExit(f"Input CSV not found: {predictions_csv}")
    else:
        parser.error("Provide either --checkpoint (+ --test-csv) or --csv-input")

    # ── Step 2: Traditional metrics ──
    if not args.skip_metrics:
        compute_metrics(predictions_csv, output_dir)

    # ── Step 3: LLM judge (optional) ──
    if args.llm_judge:
        run_llm_judge(
            predictions_csv,
            output_dir,
            model=args.judge_model,
            batch_by_image=True,
            max_samples=args.judge_max_samples,
        )

    logger.info("Evaluation complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
