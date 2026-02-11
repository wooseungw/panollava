"""
CORA Evaluation Script
Usage:
    python scripts/eval.py --checkpoint runs/exp/finetune/best --csv data/test.csv
"""

import sys
import os
import argparse
import logging
import json
import torch
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cora.inference.generator import PanoramaGenerator
from cora.evaluation.metrics import CORAEvaluator

def main():
    parser = argparse.ArgumentParser(description="CORA Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--image_root", type=str, help="Root directory for images if relative paths in CSV")
    parser.add_argument("--output_file", type=str, help="Path to save results JSON")
    parser.add_argument("--save_predictions_csv", action="store_true", help="Also save predictions as CSV next to JSON")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. Load Generator
    logger.info(f"Loading generator from {args.checkpoint}...")
    generator = PanoramaGenerator(args.checkpoint, device=args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_dir = checkpoint_path.parent
    config_path = (checkpoint_dir / "config.yaml").resolve() if (checkpoint_dir / "config.yaml").exists() else None
    
    # 2. Load Data
    logger.info(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Determine columns
    input_col = 'instruction' if 'instruction' in df.columns else ('query' if 'query' in df.columns else 'text')
    image_col = 'url' if 'url' in df.columns else 'image'
    gt_col = 'response' if 'response' in df.columns else ('annotation' if 'annotation' in df.columns else None)
    
    # 3. Generate Predictions
    logger.info("Generating predictions...")
    results = []
    
    image_root = Path(args.image_root) if args.image_root else Path("")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            image_path = image_root / row[image_col]
            prompt = row[input_col]
            
            # Simple generation
            # For efficiency in production, implement batching in generator
            caption = generator.generate(str(image_path), prompt)
            
            res_item = {
                "image_id": str(idx),
                "prompt": prompt,
                "caption": caption,
                "file_name": str(row[image_col])
            }
            if gt_col and gt_col in row:
                res_item["reference"] = str(row[gt_col])
                
            results.append(res_item)
            
        except Exception as e:
            logger.error(f"Error processing {idx}: {e}")

    # 4. Compute Metrics
    metrics_scores = {}
    if gt_col:
        logger.info("Computing metrics...")
        evaluator = CORAEvaluator()
        
        # Format for evaluator
        preds = [{"image_id": r["image_id"], "caption": r["caption"]} for r in results]
        refs = [{"image_id": r["image_id"], "caption": r["reference"]} for r in results if "reference" in r]
        
        if preds and refs:
            metrics_scores = evaluator.compute_metrics(preds, refs)
            logger.info(f"Metrics: {metrics_scores}")
    
    # 5. Save Results
    output_file = args.output_file
    if not output_file:
        checkpoint_name = checkpoint_path.stem
        output_file = str(checkpoint_dir / f"eval_results_{checkpoint_name}.json")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_info = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_dir": str(checkpoint_dir),
        "config_path": str(config_path) if config_path else None,
        "test_csv": str(Path(args.csv).resolve()),
        "image_root": str(Path(args.image_root).resolve()) if args.image_root else None,
        "device": args.device,
        "batch_size": args.batch_size,
        "model": {
            "vision_name": generator.config.models.vision_name,
            "language_model_name": generator.config.models.language_model_name,
            "resampler_type": generator.config.models.resampler_type,
        },
        "image_processing": {
            "crop_strategy": generator.config.image_processing.crop_strategy,
            "fov_deg": generator.config.image_processing.fov_deg,
            "overlap_ratio": generator.config.image_processing.overlap_ratio,
            "stitching_mode": generator.config.image_processing.stitching_mode,
        },
        "experiment": {
            "name": generator.config.experiment.get("name", ""),
            "description": generator.config.experiment.get("description", ""),
            "training_prefix": getattr(generator.config.training, "prefix", None),
        },
    }
        
    final_output = {
        "run_info": run_info,
        "metrics": metrics_scores,
        "predictions": results
    }

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    logger.info(f"Saved results to {output_path}")

    if args.save_predictions_csv:
        pred_rows = []
        for item in results:
            pred_rows.append({
                "image_id": item.get("image_id"),
                "image_path": item.get("file_name"),
                "original_query": item.get("prompt"),
                "prediction": item.get("caption"),
                "reference": item.get("reference", ""),
            })
        csv_path = output_path.with_suffix(".predictions.csv")
        pd.DataFrame(pred_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved predictions CSV to {csv_path}")

if __name__ == "__main__":
    main()
