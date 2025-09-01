# coding: utf-8
"""
Lightweight validator for PanoramaVLM checkpoints and LoRA I/O.

Usage examples:
  python scripts/validate_ckpt_and_lora.py \
    --ckpt runs/siglipv2qwen25Instruct_e2p_finetune_mlp/best_v1.ckpt \
    --lora runs/siglipv2qwen25Instruct_e2p_finetune_mlp/lora_weights

  # If you have a HF-style folder (pytorch_model.bin), you can try full load:
  python scripts/validate_ckpt_and_lora.py \
    --model-dir runs/siglipv2qwen25Instruct_e2p_finetune_mlp/hf_model \
    --lora runs/siglipv2qwen25Instruct_e2p_finetune_mlp/lora_weights

This script is safe to run on CPU and does not require network for file checks.
If model loading is attempted, it may require a cached HF model and enough RAM.
"""

import argparse
import json
from pathlib import Path
import sys

import torch


def check_ckpt_structure(ckpt_path: Path) -> dict:
    result = {"path": str(ckpt_path), "ok": False, "details": {}}
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as e:
        result["error"] = f"Failed to load ckpt: {e}"
        return result

    hparams = ckpt.get("hyper_parameters", {})
    state_dict = ckpt.get("state_dict", {})
    model_keys = [k for k in state_dict.keys() if k.startswith("model.")]
    result["details"].update({
        "hparams_keys": list(hparams.keys()),
        "state_dict_len": len(state_dict),
        "model_prefix_count": len(model_keys),
        "sample_model_keys": model_keys[:8],
    })
    result["ok"] = len(model_keys) > 0
    return result


def check_lora_folder(lora_dir: Path) -> dict:
    result = {"path": str(lora_dir), "ok": False, "details": {}}
    adapter_cfg = lora_dir / "adapter_config.json"
    adapter_sft = lora_dir / "adapter_model.safetensors"
    adapter_bin = lora_dir / "adapter_model.bin"
    if not adapter_cfg.exists():
        result["error"] = "adapter_config.json not found"
        return result

    # parse config
    try:
        with adapter_cfg.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        result["error"] = f"Failed to parse adapter_config.json: {e}"
        return result

    result["details"]["adapter_config"] = {
        "peft_type": cfg.get("peft_type"),
        "task_type": cfg.get("task_type"),
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "lora_dropout": cfg.get("lora_dropout"),
        "target_modules": cfg.get("target_modules"),
        "base_model_name_or_path": cfg.get("base_model_name_or_path"),
    }

    # probe weight file
    weights_path = adapter_sft if adapter_sft.exists() else adapter_bin if adapter_bin.exists() else None
    if weights_path is None:
        result["error"] = "No adapter weight file found (adapter_model.safetensors or adapter_model.bin)"
        return result

    # list a few tensor keys
    try:
        if str(weights_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            st = load_file(str(weights_path))
        else:
            st = torch.load(str(weights_path), map_location="cpu")
        keys = list(st.keys())
        # some representative lora keys (may vary per arch)
        lora_keys = [k for k in keys if ".lora_A.weight" in k or ".lora_B.weight" in k]
        result["details"]["weights_file"] = str(weights_path)
        result["details"]["num_tensors"] = len(keys)
        result["details"]["num_lora_tensors"] = len(lora_keys)
        result["details"]["sample_lora_keys"] = lora_keys[:8]
        result["ok"] = len(lora_keys) > 0
    except Exception as e:
        result["error"] = f"Failed to read weights: {e}"
        return result

    return result


def try_model_load(model_dir: Path, lora_dir: Path | None, device: str = "cpu") -> dict:
    """Attempt to instantiate and optionally attach LoRA. This may require cached HF models."""
    out = {"attempted": False, "ok": False, "error": None, "lora_ok": None}
    try:
        from panovlm.model import PanoramaVLM
        out["attempted"] = True
        m = PanoramaVLM.from_pretrained_dir(str(model_dir), device=device)
        if lora_dir is not None:
            ok = m.load_lora_weights(str(lora_dir))
            out["lora_ok"] = bool(ok)
        out["ok"] = True
    except Exception as e:
        out["error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, help="Path to a .ckpt file", default=None)
    ap.add_argument("--lora", type=str, help="Path to a LoRA folder", default=None)
    ap.add_argument("--model-dir", type=str, help="HF-style folder with pytorch_model.bin", default=None)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    if args.ckpt:
        p = Path(args.ckpt)
        print("[CKPT] Validating:", p)
        res = check_ckpt_structure(p)
        print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.lora:
        lp = Path(args.lora)
        print("[LoRA] Validating folder:", lp)
        res = check_lora_folder(lp)
        print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.model_dir:
        mp = Path(args.model_dir)
        lp = Path(args.lora) if args.lora else None
        print("[MODEL] Trying full load (may require cached HF weights):", mp)
        res = try_model_load(mp, lp, device=args.device)
        print(json.dumps(res, ensure_ascii=False, indent=2))

    if not (args.ckpt or args.lora or args.model_dir):
        print("Nothing to validate. Pass --ckpt, --lora, or --model-dir.")
        sys.exit(1)


if __name__ == "__main__":
    main()

