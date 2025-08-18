#!/usr/bin/env python3
# coding: utf-8
"""
PanoramaVLM 모델 유틸리티 함수들
==============================

훈련과 평가 간 일관된 모델 처리를 위한 유틸리티 함수들을 제공합니다.
새로운 통합 인터페이스와 기존 코드 간의 브릿지 역할을 합니다.
"""

import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_panorama_model(
    checkpoint_path: Union[str, Path], 
    lora_weights_path: Optional[Union[str, Path]] = None,
    device: str = "auto",
    use_new_interface: bool = True,
    **kwargs
):
    """
    통합된 PanoramaVLM 모델 로딩 함수
    
    훈련과 평가에서 일관된 방식으로 모델을 로딩할 수 있도록 합니다.
    새로운 인터페이스를 우선 사용하고, 실패 시 기존 방식으로 폴백합니다.
    
    Args:
        checkpoint_path: 체크포인트 파일 또는 디렉토리 경로
        lora_weights_path: LoRA 가중치 경로 (선택적)
        device: 디바이스 설정 ("auto", "cuda", "cpu")
        use_new_interface: 새로운 인터페이스 사용 여부
        **kwargs: 추가 모델 파라미터들
        
    Returns:
        로드된 모델 인스턴스
        
    Example:
        # 기본 사용법
        model = load_panorama_model("runs/best.ckpt")
        
        # 평가용 (eval.py와 호환)
        model = load_panorama_model("runs/best.ckpt", device="cuda")
        
        # 훈련용 (train.py에서 resume 시)
        model = load_panorama_model("runs/best.ckpt", use_new_interface=False)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if use_new_interface:
        try:
            # 새로운 통합 인터페이스 사용
            from panovlm.model import PanoramaVLM
            
            logger.info(f"🚀 새로운 인터페이스로 모델 로딩: {checkpoint_path}")
            
            if checkpoint_path.is_dir():
                # 디렉토리인 경우 from_pretrained 사용
                model = PanoramaVLM.from_pretrained(str(checkpoint_path), device=device, **kwargs)
            else:
                # 파일인 경우 from_checkpoint 사용
                model = PanoramaVLM.from_checkpoint(
                    str(checkpoint_path), 
                    lora_weights_path=str(lora_weights_path) if lora_weights_path else None,
                    device=device, 
                    **kwargs
                )
            
            logger.info("✅ 새로운 인터페이스로 로딩 성공")
            return model
            
        except Exception as e:
            logger.warning(f"⚠️ 새로운 인터페이스 로딩 실패: {e}")
            logger.info("🔄 기존 방식으로 폴백...")
    
    # 기존 방식 사용 (훈련 코드와 호환)
    try:
        from train import VLMModule
        
        logger.info(f"🔧 기존 인터페이스로 모델 로딩: {checkpoint_path}")
        
        # 디바이스 설정
        if device == "auto":
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)
        
        # Lightning 방식으로 로딩
        model = VLMModule.load_from_checkpoint(
            str(checkpoint_path),
            map_location=device_obj,
            strict=False,
            **kwargs
        )
        
        # LoRA 가중치 로딩 (필요한 경우)
        if lora_weights_path and Path(lora_weights_path).exists():
            logger.info(f"🔧 LoRA 가중치 로딩: {lora_weights_path}")
            success = model.model.load_lora_weights(str(lora_weights_path))
            if success:
                logger.info("✅ LoRA 가중치 로딩 성공")
            else:
                logger.warning("⚠️ LoRA 가중치 로딩 실패")
        
        model.eval()
        model = model.to(device_obj)
        
        logger.info("✅ 기존 인터페이스로 로딩 성공")
        return model
        
    except Exception as e:
        logger.error(f"❌ 모든 로딩 방식 실패: {e}")
        raise


def save_panorama_model(
    model, 
    save_path: Union[str, Path],
    save_format: str = "all",
    save_lora_separately: bool = True
):
    """
    PanoramaVLM 모델을 다양한 형식으로 저장
    
    Args:
        model: 저장할 모델 (VLMModule 또는 PanoramaVLM)
        save_path: 저장 경로
        save_format: 저장 형식 ("all", "hf", "lightning", "safetensors")
        save_lora_separately: LoRA 가중치 별도 저장 여부
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 실제 PanoramaVLM 모델 추출
    if hasattr(model, 'model') and hasattr(model.model, 'save_pretrained'):
        # VLMModule인 경우
        panorama_model = model.model
        lightning_model = model
    elif hasattr(model, 'save_pretrained'):
        # 이미 PanoramaVLM인 경우
        panorama_model = model
        lightning_model = None
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {type(model)}")
    
    logger.info(f"💾 모델 저장 중: {save_path} (형식: {save_format})")
    
    if save_format in ("all", "hf"):
        # HuggingFace 스타일 저장
        hf_path = save_path / "hf_model"
        panorama_model.save_pretrained(str(hf_path), save_lora_separately=save_lora_separately)
        logger.info(f"✅ HuggingFace 스타일 저장: {hf_path}")
    
    if save_format in ("all", "lightning") and lightning_model is not None:
        # Lightning 체크포인트 저장
        lightning_path = save_path / "lightning_model.ckpt"
        torch.save(lightning_model.state_dict(), lightning_path)
        logger.info(f"✅ Lightning 체크포인트 저장: {lightning_path}")
    
    if save_format in ("all", "safetensors"):
        # SafeTensors 저장
        safetensors_path = save_path / "model.safetensors"
        try:
            from safetensors.torch import save_file
            save_file(panorama_model.state_dict(), safetensors_path)
            logger.info(f"✅ SafeTensors 저장: {safetensors_path}")
        except ImportError:
            logger.warning("⚠️ SafeTensors 패키지가 없어 건너뜀")
    
    logger.info(f"🎉 모델 저장 완료: {save_path}")


def get_model_info(model) -> Dict[str, Any]:
    """
    모델의 상세 정보를 반환
    
    Args:
        model: 분석할 모델
        
    Returns:
        모델 정보 딕셔너리
    """
    # 실제 PanoramaVLM 모델 추출
    if hasattr(model, 'model'):
        panorama_model = model.model
        is_lightning = True
    else:
        panorama_model = model
        is_lightning = False
    
    info = {
        "model_type": type(panorama_model).__name__,
        "is_lightning_wrapper": is_lightning,
        "total_parameters": sum(p.numel() for p in panorama_model.parameters()),
        "trainable_parameters": sum(p.numel() for p in panorama_model.parameters() if p.requires_grad),
        "device": str(next(panorama_model.parameters()).device),
        "training_mode": panorama_model.training,
    }
    
    # LoRA 정보 추가
    if hasattr(panorama_model, 'get_lora_info'):
        lora_info = panorama_model.get_lora_info()
        info["lora_info"] = lora_info
    
    # 모델 구성 정보 추가
    try:
        info.update({
            "vision_model": getattr(panorama_model.vision_encoder.config, 'name_or_path', 'unknown'),
            "language_model": getattr(panorama_model.language_model.config, 'name_or_path', 'unknown'),
            "max_text_length": getattr(panorama_model, 'max_text_length', 'unknown'),
            "vicreg_loss_weight": getattr(panorama_model, 'vicreg_loss_weight', 'unknown'),
        })
    except Exception:
        pass
    
    return info


def print_model_info(model):
    """
    모델 정보를 보기 좋게 출력
    """
    info = get_model_info(model)
    
    print("=" * 60)
    print("🔍 모델 정보")
    print("=" * 60)
    
    print(f"📊 기본 정보:")
    print(f"   - 모델 타입: {info['model_type']}")
    print(f"   - Lightning 래퍼: {info['is_lightning_wrapper']}")
    print(f"   - 디바이스: {info['device']}")
    print(f"   - 훈련 모드: {info['training_mode']}")
    
    print(f"\n📈 파라미터 정보:")
    total = info['total_parameters']
    trainable = info['trainable_parameters']
    print(f"   - 전체 파라미터: {total:,}")
    print(f"   - 훈련 가능: {trainable:,} ({trainable/total*100:.1f}%)")
    
    if 'lora_info' in info and info['lora_info'].get('is_lora_enabled', False):
        lora = info['lora_info']
        print(f"\n🎯 LoRA 정보:")
        print(f"   - 활성화: {lora['is_lora_enabled']}")
        print(f"   - Rank: {lora.get('lora_r', 'N/A')}")
        print(f"   - Alpha: {lora.get('lora_alpha', 'N/A')}")
        print(f"   - Dropout: {lora.get('lora_dropout', 'N/A')}")
    
    if 'vision_model' in info:
        print(f"\n🖼️  모델 구성:")
        print(f"   - Vision: {info['vision_model']}")
        print(f"   - Language: {info['language_model']}")
        print(f"   - Max text length: {info['max_text_length']}")
        print(f"   - VICReg weight: {info['vicreg_loss_weight']}")
    
    print("=" * 60)


# 편의 함수들
def quick_load(checkpoint_path: str, **kwargs):
    """빠른 모델 로딩 (가장 간단한 방법)"""
    return load_panorama_model(checkpoint_path, **kwargs)


def load_for_training(checkpoint_path: str, **kwargs):
    """훈련용 모델 로딩 (Lightning 방식)"""
    return load_panorama_model(checkpoint_path, use_new_interface=False, **kwargs)


def load_for_inference(checkpoint_path: str, **kwargs):
    """추론용 모델 로딩 (새 인터페이스)"""
    return load_panorama_model(checkpoint_path, use_new_interface=True, **kwargs)


if __name__ == "__main__":
    # 사용 예시
    import argparse
    
    parser = argparse.ArgumentParser(description="모델 유틸리티 테스트")
    parser.add_argument("--checkpoint", required=True, help="체크포인트 경로")
    parser.add_argument("--info-only", action="store_true", help="정보만 출력")
    
    args = parser.parse_args()
    
    try:
        print(f"🚀 모델 로딩 테스트: {args.checkpoint}")
        model = quick_load(args.checkpoint)
        
        print_model_info(model)
        
        if not args.info_only:
            print("\n✅ 모델 로딩 및 정보 출력 성공!")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")