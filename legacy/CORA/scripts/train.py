"""
CORA Training Entry Script
Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml --stages resampler finetune
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cora.training.trainer import CORATrainer

def main():
    parser = argparse.ArgumentParser(description="CORA Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--stages", nargs="+", help="Specific stages to run (override config)")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from (global resume)")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained model for initialization (start new training)")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    
    # Debug args
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (fast dev run)")
    parser.add_argument("--no_log", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. Initialize Trainer
    trainer = CORATrainer(args.config)
    
    # 2. Apply Overrides
    if args.stages:
        trainer.config.training.stages = args.stages
        
    if args.output_dir:
        trainer.config.training.output_dir = args.output_dir
        # Re-create output dir since init created it already
        trainer.output_dir = Path(args.output_dir) / trainer.config.experiment["name"]
        trainer.output_dir.mkdir(parents=True, exist_ok=True)
        
    if args.no_log:
        trainer.config.training.wandb_project = None
        
    # TODO: Implement debug/fast_dev_run injection if needed in Trainer
    
    # 3. Handle Resume/Pretrained logic
    # The Trainer._run_stage logic handles 'resume_checkpoint'.
    # If args.resume is set, we might want to pass it to the first stage?
    # Or update config.
    if args.resume:
        trainer.config.training.resume_from_checkpoint = args.resume
        
    # 4. Run
    trainer.train()

if __name__ == "__main__":
    main()
