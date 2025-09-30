#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PanoLLaVA Development Environment Setup Script
==============================================

This script sets up the development environment for PanoLLaVA.
Run this script to install all dependencies and configure the environment.

Usage:
    python setup_dev.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    print(f"🔧 {description}")
    print(f"   Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, shell=isinstance(cmd, str), check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✓ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version < (3, 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Minimum required: 3.8")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")

    # Install in development mode
    if not run_command([sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
                      "Installing PanoLLaVA with development dependencies"):
        return False

    return True

def setup_pre_commit():
    """Setup pre-commit hooks."""
    print("\n🔗 Setting up pre-commit hooks...")

    if not run_command(["pre-commit", "install"], "Installing pre-commit hooks"):
        print("   ⚠️  Pre-commit not available, skipping...")
        return True

    if not run_command(["pre-commit", "run", "--all-files"], "Running pre-commit on all files"):
        print("   ⚠️  Pre-commit checks failed, but continuing...")

    return True

def check_gpu_availability():
    """Check GPU availability."""
    print("\n🖥️  Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"✓ CUDA available: {device_count} device(s)")
            print(f"✓ Current device: {device_name}")
            return True
        else:
            print("⚠️  CUDA not available, CPU-only mode")
            return True
    except ImportError:
        print("⚠️  PyTorch not installed, skipping GPU check")
        return True

def test_installation():
    """Test that the installation works."""
    print("\n🧪 Testing installation...")

    # Test basic import
    try:
        import panovlm
        print("✓ PanoLLaVA import successful")
    except ImportError as e:
        print(f"❌ PanoLLaVA import failed: {e}")
        return False

    # Test core dependencies
    dependencies = ['torch', 'transformers', 'lightning', 'PIL', 'numpy', 'yaml']
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} available")
        except ImportError:
            print(f"❌ {dep} not available")

    return True

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    if not env_file.exists():
        print("\n📝 Creating .env file...")
        env_content = """# PanoLLaVA Environment Configuration
# ==================================

# Hugging Face Hub token (optional, for private models)
# HF_TOKEN=your_token_here

# Wandb API key (optional, for logging)
# WANDB_API_KEY=your_key_here

# CUDA settings
# CUDA_VISIBLE_DEVICES=0,1,2,3

# Development settings
PYTHONPATH=/data/1_personal/4_SWWOO/panollava/src
"""
        env_file.write_text(env_content)
        print("✓ .env file created")
    else:
        print("✓ .env file already exists")

def main():
    """Main setup function."""
    print("🚀 Setting up PanoLLaVA development environment...")
    print("=" * 60)

    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    success = True

    # Run setup steps
    steps = [
        check_python_version,
        install_dependencies,
        setup_pre_commit,
        check_gpu_availability,
        test_installation,
        create_env_file,
    ]

    for step in steps:
        if not step():
            success = False
            break

    print("\n" + "=" * 60)
    if success:
        print("🎉 Development environment setup completed successfully!")
        print("\nNext steps:")
        print("  1. Activate your virtual environment (if using one)")
        print("  2. Run 'make quick-test' to verify everything works")
        print("  3. Run 'make train-vision' to start training")
        print("  4. Check docs/ for more information")
    else:
        print("❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()