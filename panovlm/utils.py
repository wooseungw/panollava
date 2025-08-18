from __future__ import annotations
from logging import getLogger
logger = getLogger(__name__)
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

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




