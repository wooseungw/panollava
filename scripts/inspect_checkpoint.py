#!/usr/bin/env python3
# coding: utf-8
"""
Inspect Lightning checkpoint structure.

체크포인트의 구조를 빠르게 확인합니다.
"""

import torch
import sys
from pathlib import Path

def inspect_checkpoint(ckpt_path: str):
    """Inspect checkpoint structure"""
    print(f"Loading: {ckpt_path}\n")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print("="*80)
    print("Top-level keys:")
    print("="*80)
    for key in ckpt.keys():
        if isinstance(ckpt[key], dict):
            print(f"  {key}: dict with {len(ckpt[key])} items")
        elif isinstance(ckpt[key], (list, tuple)):
            print(f"  {key}: {type(ckpt[key]).__name__} with {len(ckpt[key])} items")
        else:
            print(f"  {key}: {type(ckpt[key]).__name__}")
    
    print("\n" + "="*80)
    print("State dict sample (first 10 keys):")
    print("="*80)
    state_dict = ckpt.get('state_dict', {})
    for i, key in enumerate(list(state_dict.keys())[:10]):
        tensor = state_dict[key]
        print(f"  {key}")
        print(f"    Shape: {tensor.shape}, dtype: {tensor.dtype}")
    
    print(f"\nTotal state dict keys: {len(state_dict)}")
    
    print("\n" + "="*80)
    print("Hyper parameters:")
    print("="*80)
    hparams = ckpt.get('hyper_parameters', {})
    if hparams:
        for key, value in list(hparams.items())[:20]:
            if isinstance(value, (str, int, float, bool)):
                print(f"  {key}: {value}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} items")
            else:
                print(f"  {key}: {type(value).__name__}")
    else:
        print("  No hyper_parameters found")
    
    # Check for specific keys we need
    print("\n" + "="*80)
    print("Config extraction check:")
    print("="*80)
    
    config_keys = [
        'vision_name', 'language_model_name', 'resampler_type',
        'latent_dimension', 'num_latent_tokens',
        'vicreg_similarity_weight', 'overlap_ratio',
        'crop_strategy', 'num_views', 'use_lora'
    ]
    
    for key in config_keys:
        if key in hparams:
            print(f"  ✅ {key}: {hparams[key]}")
        else:
            print(f"  ❌ {key}: NOT FOUND")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    inspect_checkpoint(sys.argv[1])
