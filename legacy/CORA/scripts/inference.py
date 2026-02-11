"""
CORA Single Image Inference Script
Usage:
    python scripts/inference.py --checkpoint runs/exp/finetune/best --image data/sample.jpg --prompt "Describe the panorama."
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cora.inference.generator import PanoramaGenerator

def main():
    parser = argparse.ArgumentParser(description="CORA Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--prompt", type=str, default="Describe the panorama image.", help="Text prompt")
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # 1. Load Generator
    logger.info(f"Loading model from {args.checkpoint} on {device}...")
    generator = PanoramaGenerator(args.checkpoint, device=device)
    
    # 2. Generate
    logger.info(f"Generating for image: {args.image}")
    output = generator.generate(args.image, args.prompt)
    
    print("\n" + "="*40)
    print(f"Prompt: {args.prompt}")
    print(f"Output: {output}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
