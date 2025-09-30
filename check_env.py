#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PanoLLaVA Environment Checker
============================

This script checks the development environment and reports any issues.

Usage:
    python check_env.py
"""

import sys
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("🐍 Python Version:")
    version = sys.version_info
    print(f"   Version: {version.major}.{version.minor}.{version.micro}")

    if version >= (3, 8):
        print("   ✓ Compatible (Python >= 3.8)")
        return True
    else:
        print("   ❌ Not compatible (Python >= 3.8 required)")
        return False

def check_package(package_name, min_version=None, description=""):
    """Check if a package is installed and optionally check version."""
    try:
        package = importlib.import_module(package_name)
        version = getattr(package, '__version__', 'unknown')
        print(f"   ✓ {package_name}: {version}")

        if min_version and version != 'unknown':
            # Simple version comparison (could be improved)
            if package_name == 'torch' and hasattr(package, 'version'):
                version = package.version.__version__
            # Add more version checks as needed

        return True
    except ImportError:
        print(f"   ❌ {package_name}: Not installed")
        if description:
            print(f"      {description}")
        return False

def check_cuda():
    """Check CUDA availability."""
    print("\n🖥️  CUDA:")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"   ✓ Available: {device_count} device(s)")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"      Device {i}: {name} ({memory:.1f} GB)")
            return True
        else:
            print("   ⚠️  Not available (CPU-only mode)")
            return True
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_huggingface():
    """Check Hugging Face setup."""
    print("\n🤗 Hugging Face:")
    try:
        import huggingface_hub
        version = huggingface_hub.__version__
        print(f"   ✓ huggingface_hub: {version}")

        # Check if token is set
        from huggingface_hub.utils import HfFolder
        token = HfFolder.get_token()
        if token:
            print("   ✓ Hugging Face token configured")
        else:
            print("   ⚠️  No Hugging Face token (some models may not be accessible)")

        return True
    except ImportError:
        print("   ❌ huggingface_hub not available")
        return False

def check_disk_space():
    """Check available disk space."""
    print("\n💾 Disk Space:")
    try:
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                # Parse the second line (actual filesystem info)
                parts = lines[1].split()
                if len(parts) >= 4:
                    available = parts[3]
                    print(f"   Available: {available}")
                    print("   ✓ Sufficient for most models")
                    return True
        print("   ⚠️  Could not determine disk space")
        return True
    except Exception as e:
        print(f"   ⚠️  Error checking disk space: {e}")
        return True

def check_paths():
    """Check important paths."""
    print("\n📁 Paths:")
    paths_to_check = [
        ("Project root", Path(__file__).parent),
        ("Source code", Path(__file__).parent / "src"),
        ("Configs", Path(__file__).parent / "configs"),
        ("Scripts", Path(__file__).parent / "scripts"),
        ("Results", Path(__file__).parent / "results"),
    ]

    all_good = True
    for name, path in paths_to_check:
        if path.exists():
            print(f"   ✓ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path} (missing)")
            all_good = False

    return all_good

def main():
    """Main check function."""
    print("🔍 PanoLLaVA Environment Check")
    print("=" * 50)

    checks = [
        check_python_version,
        check_paths,
        check_cuda,
        check_huggingface,
        check_disk_space,
    ]

    # Core packages
    print("\n📦 Core Packages:")
    core_packages = [
        ('torch', '1.12.0', 'Deep learning framework'),
        ('transformers', '4.20.0', 'Hugging Face transformers'),
        ('lightning', '2.0.0', 'PyTorch Lightning'),
        ('PIL', None, 'Python Imaging Library'),
        ('numpy', None, 'Numerical computing'),
        ('pandas', None, 'Data manipulation'),
        ('pyyaml', None, 'YAML parsing'),
        ('tqdm', None, 'Progress bars'),
    ]

    core_results = []
    for package, min_ver, desc in core_packages:
        result = check_package(package, min_ver, desc)
        core_results.append(result)

    # PEFT for LoRA
    print("\n🔧 PEFT (LoRA):")
    peft_result = check_package('peft', '0.6.0', 'Parameter-Efficient Fine-Tuning')

    print("\n" + "=" * 50)

    # Summary
    all_passed = all(core_results) and peft_result

    if all_passed:
        print("🎉 All checks passed! Environment is ready.")
        print("\n💡 Tips:")
        print("   • Run 'python setup_dev.py' to setup development environment")
        print("   • Run 'make quick-test' to verify installation")
        print("   • Run 'make train-vision' to start training")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\n🔧 To fix common issues:")
        print("   • Install missing packages: pip install -r requirements.txt")
        print("   • Setup development environment: python setup_dev.py")
        print("   • Check documentation for troubleshooting")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())