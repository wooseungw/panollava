#!/usr/bin/env python3
"""
설정 시스템 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panovlm.config import ConfigManager

def test_config_loading():
    """설정 로딩 테스트"""
    print("=== ConfigManager 테스트 ===")

    # ConfigManager 생성 및 설정 로딩
    config_manager = ConfigManager("config.yaml")
    config_manager.load_config()  # 설정 로딩 추가

    print(f"사용 가능한 스테이지: {config_manager.get_available_stages()}")
    print(f"기본 스테이지: {config_manager.get_default_stage()}")

    # 각 스테이지 설정 확인
    for stage_name in config_manager.get_available_stages():
        print(f"\n--- 스테이지: {stage_name} ---")
        stage_config = config_manager.get_stage_config(stage_name)
        if stage_config:
            print(f"데이터 설정: {stage_config.data}")
            print(f"손실 함수: VICReg={stage_config.loss.vicreg.get('enabled', False)}, LM={stage_config.loss.language_modeling.get('enabled', True)}")
            print(f"최적화: LR={stage_config.optimizer.lr}, Epochs={stage_config.optimizer.epochs}, Batch={stage_config.optimizer.batch_size}")
        else:
            print("설정 로딩 실패")

    # 실험 설정 확인
    experiment_config = config_manager.get_experiment_config()
    print("\n=== 실험 설정 ===")
    print(f"모델: Vision={experiment_config.model_config.get('vision_name', 'N/A')}")
    print(f"언어 모델: {experiment_config.model_config.get('language_model_name', 'N/A')}")
    print(f"Resampler: {experiment_config.model_config.get('resampler_type', 'N/A')}")
    print(f"LoRA: {experiment_config.lora_config.get('use_lora', False)}")

    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    test_config_loading()