from __future__ import annotations
from logging import getLogger
logger = getLogger(__name__)
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
import math
import os
import random

import torch


# ==================== Checkpoint I/O ====================
def safe_load_checkpoint(checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """안전한 체크포인트 로딩"""
    try:
        if str(checkpoint_path).endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as load_safetensors
                state_dict = load_safetensors(checkpoint_path)
                logger.info(f"Successfully loaded safetensors: {checkpoint_path}")
                return {"state_dict": state_dict}
            except ImportError:
                logger.error("safetensors package not installed. Install with: pip install safetensors")
                return None
        else:
            try:
                # PyTorch 2.0+ 보안 기능 지원
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
                return checkpoint
            except TypeError:
                # 이전 PyTorch 버전 지원
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
                return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def save_checkpoint_safely(model_state: Dict[str, Any], path: Union[str, Path]):
    """안전한 체크포인트 저장"""
    try:
        # 디렉토리 생성
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # safetensors 우선 시도
        try:
            from safetensors.torch import save_file as save_safetensors
            safetensor_path = str(path).replace(".ckpt", ".safetensors")
            save_safetensors(model_state, safetensor_path)
            logger.info(f"Model saved as safetensors: {safetensor_path}")
        except ImportError:
            torch.save(model_state, path)
            logger.info(f"Model saved as pytorch checkpoint: {path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


# ==================== General Utilities ====================
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Reproducibility helper: set seeds for Python, NumPy, and Torch."""
    try:
        import numpy as np  # lazy import to avoid hard dependency at module import time
    except Exception as _e:
        np = None  # type: ignore
        logger.warning(f"NumPy not available for set_seed: {_e}")

    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
        else:
            torch.backends.cudnn.benchmark = True
    except Exception as _e:
        logger.warning(f"[set_seed] Torch seed setup skipped: {_e}")


def safe_save_pretrained(model, save_path: str, **kwargs) -> bool:
    """
    HuggingFace save_pretrained wrapper that avoids hub/network args and
    prefers safetensors, with graceful fallback to PyTorch.
    Returns True on success, False otherwise.
    """
    if not hasattr(model, 'save_pretrained'):
        return False
    safe_kwargs = {
        'push_to_hub': False,
        'token': False,
        'safe_serialization': kwargs.get('safe_serialization', True),
        **kwargs
    }
    # strip hub-specific keys if present
    for param in ['repo_id', 'from_id', 'to_id', 'hub_model_id']:
        safe_kwargs.pop(param, None)
    try:
        model.save_pretrained(save_path, **safe_kwargs)
        return True
    except Exception as e:
        logger.warning(f"Failed to save with SafeTensors: {e}")
        try:
            fallback_kwargs = {k: v for k, v in safe_kwargs.items() if k != 'safe_serialization'}
            model.save_pretrained(save_path, **fallback_kwargs)
            return True
        except Exception as e2:
            logger.error(f"Failed to save model completely: {e2}")
            return False


def infer_hw(num_patches: int) -> Tuple[int, int]:
    """Given number of patches, infer a plausible (H, W) grid.

    - Returns (h, h) if perfect square
    - Otherwise searches divisors from sqrt(N) downwards for a factorization
    - Raises ValueError if no factorization is found (should not happen for integers > 0)
    """
    if num_patches <= 0:
        raise ValueError(f"num_patches must be positive, got {num_patches}")
    height = int(math.sqrt(num_patches))
    if height * height == num_patches:
        return height, height
    for h in range(height, 0, -1):
        if num_patches % h == 0:
            return h, num_patches // h
    raise ValueError(f"그리드 추정 실패: 패치 수={num_patches}")





