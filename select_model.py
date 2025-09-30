#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PanoLLaVA Model Selector
=======================

This script helps you choose and configure models for LoRA fine-tuning.
It shows available models and their specifications.

Usage:
    python select_model.py
"""

import sys
from pathlib import Path
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def get_available_models():
    """Get list of available models for PanoLLaVA."""
    return {
        "qwen": {
            "Qwen2.5-0.5B-Instruct": {
                "size": "0.5B",
                "params": "~0.5B",
                "context": "32K",
                "description": "Fast and lightweight, good for quick experiments"
            },
            "Qwen2.5-1.5B-Instruct": {
                "size": "1.5B",
                "params": "~1.5B",
                "context": "32K",
                "description": "Balanced performance and speed"
            },
            "Qwen2.5-3B-Instruct": {
                "size": "3B",
                "params": "~3B",
                "context": "32K",
                "description": "Good performance for most tasks"
            },
            "Qwen2.5-7B-Instruct": {
                "size": "7B",
                "params": "~7B",
                "context": "128K",
                "description": "High performance, requires more GPU memory"
            },
            "Qwen2.5-14B-Instruct": {
                "size": "14B",
                "params": "~14B",
                "context": "128K",
                "description": "Very high performance, requires significant GPU memory"
            },
        },
        "llama": {
            "Llama-3.2-1B-Instruct": {
                "size": "1B",
                "params": "~1B",
                "context": "128K",
                "description": "Meta's lightweight model, good performance"
            },
            "Llama-3.2-3B-Instruct": {
                "size": "3B",
                "params": "~3B",
                "context": "128K",
                "description": "Meta's balanced model with good instruction following"
            },
        },
        "gemma": {
            "gemma-2-2b-it": {
                "size": "2B",
                "params": "~2B",
                "context": "8K",
                "description": "Google's efficient model"
            },
            "gemma-2-9b-it": {
                "size": "9B",
                "params": "~9B",
                "context": "8K",
                "description": "Google's high-performance model"
            },
        }
    }

def get_vision_models():
    """Get available vision models."""
    return {
        "siglip": {
            "google/siglip-base-patch16-224": {
                "size": "86M",
                "description": "Google's SigLIP model, good for general vision tasks"
            },
            "google/siglip-large-patch16-256": {
                "size": "427M",
                "description": "Larger SigLIP model, better performance but slower"
            },
        },
        "clip": {
            "openai/clip-vit-base-patch32": {
                "size": "151M",
                "description": "OpenAI's CLIP model, widely used and robust"
            },
            "openai/clip-vit-large-patch14": {
                "size": "427M",
                "description": "Larger CLIP model, better performance"
            },
        },
        "dinov2": {
            "facebook/dinov2-base": {
                "size": "86M",
                "description": "Meta's DINOv2 model, excellent for feature extraction"
            },
            "facebook/dinov2-large": {
                "size": "307M",
                "description": "Larger DINOv2 model, superior performance"
            },
        }
    }

def check_huggingface_access(model_name):
    """Check if model is accessible on Hugging Face."""
    try:
        # Simple check - try to get model info
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

def print_model_options():
    """Print available model options."""
    print("🤖 Available Language Models:")
    print("=" * 60)

    models = get_available_models()
    for family, model_dict in models.items():
        print(f"\n🔤 {family.upper()} Models:")
        for model_name, info in model_dict.items():
            accessible = check_huggingface_access(model_name)
            status = "✅" if accessible else "❌"
            print("2d"
                  f"   {status} Accessible" if accessible else f"   {status} Not accessible")

    print("\n👁️  Available Vision Models:")
    print("=" * 60)

    vision_models = get_vision_models()
    for family, model_dict in vision_models.items():
        print(f"\n🔤 {family.upper()} Models:")
        for model_name, info in model_dict.items():
            accessible = check_huggingface_access(model_name)
            status = "✅" if accessible else "❌"
            print("2d"
                  f"   {status} Accessible" if accessible else f"   {status} Not accessible")

def generate_config_file(selected_models):
    """Generate configuration file for selected models."""
    config = {
        "experiment_name": "custom_lora_ablation",
        "output_dir": "./results/custom_lora_ablation",
        "models": [],
        "lora_configs": [
            {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.1},
            {"lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.1},
            {"lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.05},
        ],
        "training": {
            "max_epochs": 10,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "warmup_steps": 100,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,
            "val_check_interval": 0.5,
        },
        "evaluation": {
            "metrics": ["validation_loss", "perplexity"],
            "num_samples": 100,
            "batch_size": 8,
        }
    }

    # Add selected models
    vision_models = get_vision_models()
    lang_models = get_available_models()

    for model_key in selected_models:
        if model_key in ["siglip-base", "siglip-large", "clip-base", "clip-large", "dinov2-base", "dinov2-large"]:
            # Vision model selection
            vision_map = {
                "siglip-base": "google/siglip-base-patch16-224",
                "siglip-large": "google/siglip-large-patch16-256",
                "clip-base": "openai/clip-vit-base-patch32",
                "clip-large": "openai/clip-vit-large-patch14",
                "dinov2-base": "facebook/dinov2-base",
                "dinov2-large": "facebook/dinov2-large",
            }
            vision_name = vision_map[model_key]
            # Add default language model
            config["models"].append({
                "name": f"qwen_0.5b_{model_key.replace('-', '_')}",
                "vision_name": vision_name,
                "language_model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "latent_dimension": 768,
            })
        else:
            # Language model selection
            lang_map = {
                "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
                "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
                "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
                "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
                "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
                "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
                "gemma-2b": "google/gemma-2-2b-it",
                "gemma-9b": "google/gemma-2-9b-it",
            }
            if model_key in lang_map:
                lang_name = lang_map[model_key]
                config["models"].append({
                    "name": model_key.replace("-", "_"),
                    "vision_name": "google/siglip-base-patch16-224",  # Default vision model
                    "language_model_name": lang_name,
                    "latent_dimension": 768,
                })

    # Save config
    import yaml
    config_path = Path("./configs/custom_lora_ablation.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n📝 Configuration saved to: {config_path}")
    return config_path

def interactive_selection():
    """Interactive model selection."""
    print("🎯 PanoLLaVA Model Selector")
    print("=" * 50)
    print("Select models for your LoRA ablation study.")
    print("You can choose multiple models by entering their keys separated by commas.")
    print()

    # Show options
    print("Available Language Models:")
    lang_options = {
        "qwen-0.5b": "Qwen2.5-0.5B (Fast, lightweight)",
        "qwen-1.5b": "Qwen2.5-1.5B (Balanced)",
        "qwen-3b": "Qwen2.5-3B (Good performance)",
        "qwen-7b": "Qwen2.5-7B (High performance)",
        "llama-1b": "Llama-3.2-1B (Meta, efficient)",
        "llama-3b": "Llama-3.2-3B (Meta, balanced)",
        "gemma-2b": "Gemma-2-2B (Google, efficient)",
        "gemma-9b": "Gemma-2-9B (Google, high performance)",
    }

    for key, desc in lang_options.items():
        print(f"  {key}: {desc}")

    print("\nAvailable Vision Models:")
    vision_options = {
        "siglip-base": "SigLIP Base (86M, fast)",
        "siglip-large": "SigLIP Large (427M, better performance)",
        "clip-base": "CLIP Base (151M, robust)",
        "clip-large": "CLIP Large (427M, high performance)",
        "dinov2-base": "DINOv2 Base (86M, excellent features)",
        "dinov2-large": "DINOv2 Large (307M, superior performance)",
    }

    for key, desc in vision_options.items():
        print(f"  {key}: {desc}")

    print("\nEnter model keys (comma-separated): ")
    user_input = input("> ").strip()

    if not user_input:
        print("❌ No models selected. Using defaults.")
        return ["qwen-0.5b", "qwen-1.5b"]

    selected = [model.strip() for model in user_input.split(",")]
    valid_models = list(lang_options.keys()) + list(vision_options.keys())

    invalid = [m for m in selected if m not in valid_models]
    if invalid:
        print(f"❌ Invalid model(s): {invalid}")
        print(f"Valid options: {valid_models}")
        return []

    print(f"✅ Selected models: {selected}")
    return selected

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        # Just list available models
        print_model_options()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Interactive selection
        selected = interactive_selection()
        if selected:
            config_path = generate_config_file(selected)
            print("
🚀 Run your ablation study:"            print(f"   python lora_ablation_study.py --config {config_path}")
        return

    # Default: show help
    print("🎯 PanoLLaVA Model Selector")
    print("=" * 50)
    print()
    print("Usage:")
    print("  python select_model.py --list              # List all available models")
    print("  python select_model.py --interactive       # Interactive model selection")
    print()
    print("Examples:")
    print("  # Quick training with specific model")
    print("  python quick_lora_train.py --model qwen_0.5b --lora-r 16")
    print()
    print("  # Full ablation study")
    print("  python select_model.py --interactive")
    print("  python lora_ablation_study.py --config configs/custom_lora_ablation.yaml")

if __name__ == "__main__":
    main()