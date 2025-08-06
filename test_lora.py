#!/usr/bin/env python3
"""
LoRA 지원 여부를 테스트하는 스크립트
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from panovlm.model import PanoramaVLM

def test_lora_support():
    """LoRA 지원 기능을 테스트합니다."""
    
    print("=== PanoLLaVA LoRA 지원 테스트 ===\n")
    
    # 1. PEFT 라이브러리 가용성 확인
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        print("✓ PEFT 라이브러리가 설치되어 있습니다.")
    except ImportError as e:
        print(f"✗ PEFT 라이브러리가 설치되지 않았습니다: {e}")
        print("  설치 명령: pip install peft")
        return False
    
    # 2. 모델 초기화 테스트
    try:
        print("\n📦 모델 초기화 중...")
        model = PanoramaVLM(
            vision_model_name="google/siglip-base-patch16-224",
            language_model_name="Qwen/Qwen2.5-0.5B",  # 더 가벼운 모델로 테스트
            resampler_type="mlp",
            max_text_length=64
        )
        print("✓ PanoramaVLM 모델이 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"✗ 모델 초기화 실패: {e}")
        return False
    
    # 3. LoRA 설정 테스트
    try:
        print("\n🔧 LoRA 설정 중...")
        success = model.setup_lora_for_finetune(
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1
        )
        
        if success:
            print("✓ LoRA 설정이 성공적으로 완료되었습니다.")
        else:
            print("✗ LoRA 설정에 실패했습니다.")
            return False
            
    except Exception as e:
        print(f"✗ LoRA 설정 중 오류 발생: {e}")
        return False
    
    # 4. LoRA 정보 확인
    try:
        print("\n📊 LoRA 정보 확인 중...")
        lora_info = model.get_lora_info()
        
        print(f"  - PEFT 사용 가능: {lora_info.get('peft_available', False)}")
        print(f"  - LoRA 활성화: {lora_info.get('is_lora_enabled', False)}")
        
        if lora_info.get('is_lora_enabled', False):
            print(f"  - LoRA Rank: {lora_info.get('lora_r', 'N/A')}")
            print(f"  - LoRA Alpha: {lora_info.get('lora_alpha', 'N/A')}")
            print(f"  - LoRA Dropout: {lora_info.get('lora_dropout', 'N/A')}")
            print(f"  - Target Modules: {lora_info.get('target_modules', 'N/A')}")
            
    except Exception as e:
        print(f"✗ LoRA 정보 확인 중 오류 발생: {e}")
        return False
    
    # 5. 파라미터 카운트 확인
    try:
        print("\n📈 파라미터 수 확인 중...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  - 총 파라미터: {total_params:,}")
        print(f"  - 훈련 가능한 파라미터: {trainable_params:,}")
        print(f"  - 훈련 비율: {trainable_params/total_params*100:.2f}%")
        
        if trainable_params < total_params:
            print("✓ LoRA를 통한 파라미터 효율적 학습이 설정되었습니다.")
        
    except Exception as e:
        print(f"✗ 파라미터 확인 중 오류 발생: {e}")
        return False
    
    print("\n🎉 모든 LoRA 테스트가 성공적으로 완료되었습니다!")
    print("\n3단계 finetune에서 LoRA를 사용하려면:")
    print("python train.py --stage finetune --use-lora --lora-rank 16 --lora-alpha 32")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="LoRA 지원 여부 테스트")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세한 출력")
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    success = test_lora_support()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
