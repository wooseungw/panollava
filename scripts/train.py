#!/usr/bin/env python3
"""CORA 3-stage progressive training."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse


def main():
    parser = argparse.ArgumentParser(description="CORA Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--stage", type=str, choices=["vision", "resampler", "finetune"], default=None, help="Run specific stage only")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (path or 'auto')")
    args = parser.parse_args()

    from cora.training.trainer import CORATrainer

    trainer = CORATrainer(args.config, resume=args.resume)
    trainer.train(stage=args.stage)


if __name__ == "__main__":
    main()
