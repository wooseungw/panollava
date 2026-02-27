#!/usr/bin/env python3
"""Baseline VLM LoRA finetuning."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse


def main():
    parser = argparse.ArgumentParser(description="Baseline LoRA Finetuning")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline YAML config")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    from cora.baseline.finetune import BaselineTrainer
    from cora.baseline.config import BaselineConfig
    import yaml

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    if args.model:
        cfg_dict.setdefault("model", {})["name"] = args.model
    if args.output_dir:
        cfg_dict["output_dir"] = args.output_dir
    config = BaselineConfig(**cfg_dict)
    trainer = BaselineTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
