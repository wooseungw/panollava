from __future__ import annotations
import gc
from logging import getLogger
logger = getLogger(__name__)
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union
import psutil
import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
# 전역 설정
class Config:
    """Training configuration management"""
    
    STAGE_DEFAULTS = {
        "vision": {
            "epochs": 1, 
            "lr": 5e-6, 
            "batch_size": 8,   # 32에서 8로 감소
            "vicreg_loss_weight": 1.0, 
            "max_text_length": 32
        },
        "resampler": {
            "epochs": 1, 
            "lr": 2e-6, 
            "batch_size": 4,   # 16에서 4로 감소
            "vicreg_loss_weight": 0.0, 
            "max_text_length": 64
        },
        "finetune": {
            "epochs": 1, 
            "lr": 2e-6, 
            "batch_size": 4,   # 16에서 4로 감소
            "vicreg_loss_weight": 0.0, 
            "max_text_length": 128
        }
    }
    
    MEMORY_THRESHOLDS = {
        "warning": 0.85,  # 85% 메모리 사용 시 경고
        "critical": 0.95  # 95% 메모리 사용 시 긴급 처리
    }

@contextmanager
def memory_monitor():
    """메모리 사용량 모니터링 컨텍스트"""
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        yield
    finally:
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Memory usage: {start_memory:.1f}MB -> {end_memory:.1f}MB (Δ{end_memory-start_memory:+.1f}MB)")

def check_memory_usage():
    """메모리 사용량 체크 및 필요시 가비지 컬렉션"""
    memory_percent = psutil.virtual_memory().percent / 100
    
    if memory_percent > Config.MEMORY_THRESHOLDS["critical"]:
        logger.warning(f"Critical memory usage: {memory_percent:.1%}. Forcing garbage collection.")
        gc.collect(generation=2)  # 가장 접근하기 어려운 객체까지 수집
        torch.cuda.empty_cache()
    elif memory_percent > Config.MEMORY_THRESHOLDS["warning"]:
        logger.warning(f"High memory usage: {memory_percent:.1%}")

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

def get_gpu_memory_info():
    """가능한 경우 GPU 메모리 정보 반환"""
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            cached_memory = torch.cuda.memory_reserved(device) / (1024**3)  # GB
            free_memory = total_memory - cached_memory
            return {
                "total": total_memory,
                "allocated": allocated_memory, 
                "cached": cached_memory,
                "free": free_memory,
                "utilization": (cached_memory / total_memory) * 100
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return None
    return None

def auto_adjust_batch_size(initial_batch_size: int, available_memory_gb: float) -> int:
    """사용 가능한 메모리에 따라 배치 크기 자동 조정"""
    # GPU 메모리 정보 추가 고려
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        logger.info(f"GPU Memory - Total: {gpu_info['total']:.1f}GB, Free: {gpu_info['free']:.1f}GB, Utilization: {gpu_info['utilization']:.1f}%")
        
        # GPU 메모리 기반 조정 (더 엄격한 기준)
        if gpu_info['free'] < 2:
            return max(1, initial_batch_size // 8)
        elif gpu_info['free'] < 4:
            return max(1, initial_batch_size // 4)
        elif gpu_info['free'] < 8:
            return max(1, initial_batch_size // 2)
    
    # RAM 기반 조정 (기존 로직)
    if available_memory_gb < 8:
        return max(1, initial_batch_size // 4)
    elif available_memory_gb < 16:
        return max(1, initial_batch_size // 2)
    else:
        return initial_batch_size


