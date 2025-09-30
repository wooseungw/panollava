#!/bin/bash

# PanoLLaVA YAML 기반 훈련 실행 가이드 (train.py 전용)
# 사전 준비: pip install pyyaml

set -e

echo "🔧 PanoLLaVA Training Examples (train.py + YAML)"
echo "==============================================="

# 1. 설정 파일 미리보기 (스테이지별 요약 확인)
echo "1️⃣ 설정 미리보기:"
python train.py --config config.yaml --preview

echo
# 2. 전체 파이프라인 실행 (YAML stage_order 순서대로)
echo "2️⃣ 전체 파이프라인 실행:"
python train.py --config config.yaml

echo
# 3. 특정 스테이지만 실행 (인덱스 또는 이름)
echo "3️⃣ 개별 스테이지 실행:"
python train.py --config config.yaml --stage 1      # 첫 번째 스테이지
python train.py --config config.yaml --stage 2      # 두 번째 스테이지
python train.py --config config.yaml --stage 3      # 세 번째 스테이지
# 또는 명시적 이름 사용 (config.yaml stage_order 참고)
# python train.py --config config.yaml --stage vision_pretraining
# python train.py --config config.yaml --stage resampler_training
# python train.py --config config.yaml --stage instruction_tuning

echo
# 4. 스테이지 재실행 / 강제 실행
#    이미 완료된 스테이지를 다시 돌리고 싶다면 환경변수를 이용합니다.
echo "4️⃣ 완료 스테이지 강제 재실행:"
PANOVLM_FORCE_STAGES=vision_pretraining python train.py --config config.yaml --stage vision_pretraining


echo "\n🧪 권장 워크플로우"
echo "========================"

echo "Step 1: 설정 확인"
python train.py --config config.yaml --preview

echo "Step 2: 1단계 학습"
python train.py --config config.yaml --stage vision_pretraining

echo "Step 3: 2단계 학습 (필요 시 체크포인트 자동 연결)"
python train.py --config config.yaml --stage resampler_training

echo "Step 4: 3단계 학습"
python train.py --config config.yaml --stage instruction_tuning


echo "✅ 훈련 완료 후 확인"
echo "  - 체크포인트: runs/<prefix>_*"
echo "  - 로그: training.log"
echo "  - 상태 파일: runs/<prefix>_stage_state.json"


echo "\n🔍 참고"
echo "================"
echo "- train.py는 YAML(.yaml/.yml) 설정만 지원합니다."
echo "- 하이퍼파라미터는 config.yaml 스테이지 블록에서 수정하세요."
echo "- stage 오버라이드 없이 실행하면 stage_order 순서로 전체 파이프라인을 수행합니다."
