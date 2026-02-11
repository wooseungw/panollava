"""
Quick throughput/FLOPs sampler for ablation VLMs on a single GPU.

Runs a small, fixed-number sample set through each model, records:
  - load_s: model+processor load time
  - eval_s: generation time (no metrics)
  - per_sample_s: eval_s / num_samples
  - avg_output_tokens: mean decoded token length
  - params_b: model parameter count in billions
  - approx_flops_g: very rough ~2 * params_b * avg_output_tokens (GFLOPs/caption)

Outputs one JSON per model under --output-dir/<model>/throughput.json
and an aggregate summary JSON.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/quic360/test.csv", help="Full dataset CSV path")
    parser.add_argument("--sample-size", type=int, default=64, help="Number of samples to time per model")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "blip2-flan-t5-xl",
            "blip2-opt-2.7b",
            "gemma-3-4b",
            "instructblip-vicuna-7b",
            "internvl3.5-1b",
            "internvl3.5-2b",
            "llava-1.5-7b",
            "llava-1.6-mistral-7b",
            "qwen2.5-vl-3b",
        ],
        help="Model names (must exist in evaluate_vlm_models.VLM_MODELS)",
    )
    parser.add_argument("--output-dir", default="results/ablation_speed", help="Base output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens during generation")
    parser.add_argument("--device", default="cuda", help="Device (e.g., cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Sample seed")
    args = parser.parse_args()

    # GPU pinning (e.g., CUDA_VISIBLE_DEVICES=1)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    # Load evaluate_vlm_models.py locally (avoids package import issues)
    import importlib.util
    eval_path = Path(__file__).parent / "evaluate_vlm_models.py"
    spec = importlib.util.spec_from_file_location("ev", eval_path)
    ev = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(ev)  # type: ignore

    # Disable heavy metrics for speed runs
    ev.compute_text_metrics = lambda preds, refs: {}

    # Sample subset once
    raw_df = pd.read_csv(args.csv).sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
    sample_csv = Path(args.output_dir) / "sample.csv"
    sample_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(sample_csv, index=False)

    summary = []
    for model_name in args.models:
        print(f"\n=== Measuring {model_name} ===", flush=True)
        torch.cuda.empty_cache()
        row = {"model": model_name}
        try:
            load_start = time.perf_counter()
            evaluator = ev.VLMEvaluator(
                model_name=model_name,
                data_csv=str(sample_csv),
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                max_samples=args.sample_size,
                device=args.device,
                image_size=224,
                max_new_tokens=args.max_new_tokens,
            )
            row["load_s"] = time.perf_counter() - load_start
        except Exception as e:
            row["error"] = f"load_failed: {e}"
            summary.append(row)
            print(f"  !! load failed: {e}")
            continue

        try:
            eval_start = time.perf_counter()
            res = evaluator.evaluate()
            row["eval_s"] = time.perf_counter() - eval_start
            preds_path = Path(args.output_dir) / "ablation" / model_name / "predictions.csv"
            if preds_path.is_file():
                pdf = pd.read_csv(preds_path)
                row["avg_output_tokens"] = float(
                    pdf["prediction"].fillna("").map(lambda x: len(str(x).split())).mean()
                )
                row["num_samples"] = len(pdf)
            params = sum(p.numel() for p in evaluator.model.parameters())
            row["params_b"] = params / 1e9
            if row.get("eval_s") and row.get("num_samples"):
                row["per_sample_s"] = row["eval_s"] / row["num_samples"]
            if row.get("avg_output_tokens") and row.get("params_b"):
                row["approx_flops_g"] = 2 * row["params_b"] * row["avg_output_tokens"]
        except Exception as e:
            row["error"] = f"eval_failed: {e}"
            print(f"  !! eval failed: {e}")
        finally:
            del evaluator
            torch.cuda.empty_cache()

        # Persist per-model JSON
        model_out = Path(args.output_dir) / "ablation" / model_name / "throughput.json"
        model_out.parent.mkdir(parents=True, exist_ok=True)
        model_out.write_text(json.dumps(row, indent=2))
        summary.append(row)

    # Summary JSON
    summary_path = Path(args.output_dir) / "throughput_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
