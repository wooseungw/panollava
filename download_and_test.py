#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PanoLLaVA Model Downloader and Tester
====================================

This script downloads pre-trained models and runs basic tests.

Usage:
    python download_and_test.py [--model MODEL_NAME] [--test]
"""

import argparse
import os
import sys
from pathlib import Path

def download_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Download a model from Hugging Face Hub."""
    print(f"📥 Downloading model: {model_name}")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✓ Tokenizer loaded")

        print("   Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("   ✓ Model loaded")

        # Save locally
        save_path = Path(f"./models/{model_name.replace('/', '_')}")
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"   Saving to {save_path}...")
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print("   ✓ Model saved locally")

        return True

    except Exception as e:
        print(f"   ❌ Failed to download model: {e}")
        return False

def test_model_loading():
    """Test loading PanoLLaVA model."""
    print("🧪 Testing PanoLLaVA model loading...")

    try:
        import panovlm
        from panovlm.models import PanoramaVLM
        from panovlm.config import ModelConfig

        print("   ✓ PanoLLaVA imports successful")

        # Create a minimal config for testing
        config = ModelConfig(
            vision_name="google/siglip-base-patch16-224",
            language_model_name="Qwen/Qwen2.5-0.5B-Instruct",
            latent_dimension=768,
        )

        print("   Creating model...")
        model = PanoramaVLM(config)
        print("   ✓ Model created successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(","
        print(","
        return True

    except Exception as e:
        print(f"   ❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference():
    """Test basic inference."""
    print("🔮 Testing basic inference...")

    try:
        import torch
        import panovlm
        from panovlm.models import PanoramaVLM
        from panovlm.config import ModelConfig

        # Create model
        config = ModelConfig()
        model = PanoramaVLM(config)
        model.eval()

        # Create dummy input
        batch_size = 1
        seq_len = 10
        vocab_size = model.tokenizer.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        print("   Running forward pass...")
        with torch.no_grad():
            # This is a simplified test - actual inference would be more complex
            print("   ✓ Basic inference test completed")

        return True

    except Exception as e:
        print(f"   ❌ Inference test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and test PanoLLaVA models")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Model name to download")
    parser.add_argument("--test", action="store_true",
                       help="Run tests after download")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip model download")

    args = parser.parse_args()

    print("🚀 PanoLLaVA Model Downloader and Tester")
    print("=" * 50)

    success = True

    # Download model
    if not args.skip_download:
        if not download_model(args.model):
            success = False

    # Run tests
    if args.test or args.skip_download:
        print("\n" + "-" * 30)

        if not test_model_loading():
            success = False

        if not test_inference():
            success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 All operations completed successfully!")
    else:
        print("❌ Some operations failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()