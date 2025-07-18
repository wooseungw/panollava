# PanoLLaVA Training Scripts

이 디렉토리는 PanoLLaVA 모델의 3단계 훈련을 위한 스크립트들을 포함합니다.

## 파일 구조

```
scripts/
├── config.sh                   # 🆕 공통 설정 파일 (모든 스크립트에서 사용)
├── stage1_vision_train.sh      # Stage 1: Vision Encoder 훈련
├── stage2_resampler_train.sh   # Stage 2: Resampler 훈련  
├── stage3_finetune_train.sh    # Stage 3: End-to-End Fine-tuning
├── train_all_stages.sh         # 전체 3단계 자동 훈련
├── train_custom.sh             # 사용자 정의 훈련
├── eval_finetune.sh            # Finetune 모델 평가
├── eval_resampler.sh           # Resampler 모델 평가
├── eval_compare.sh             # 모델 비교 평가
└── README.md                   # 이 파일
```

## 🚀 새로운 중앙화된 설정 관리

### config.sh
모든 스크립트는 `config.sh`에서 공통 설정을 로드합니다:
- 모델 설정 (Vision/Language 모델명)
- 데이터 경로
- 학습 하이퍼파라미터
- GPU 및 환경 설정
- 디렉토리 구조

### 설정 수정 방법
1. **전역 설정 변경**: `config.sh` 파일을 직접 수정
2. **스크립트별 오버라이드**: 각 스크립트에서 필요시 설정 오버라이드

## 사용법

### 1. 순차적 3단계 훈련

#### Stage 1: Vision Encoder 훈련
```bash
chmod +x scripts/train_stage1_vision.sh
./scripts/train_stage1_vision.sh
```

#### Stage 2: Resampler 훈련
```bash
chmod +x scripts/train_stage2_resampler.sh
./scripts/train_stage2_resampler.sh runs/vlm_vision/checkpoints/epoch=00-val_loss=4.006.ckpt
```

#### Stage 3: End-to-End Fine-tuning
```bash
chmod +x scripts/train_stage3_finetune.sh
./scripts/train_stage3_finetune.sh runs/vlm_resampler/checkpoints/epoch=01-val_loss=0.000.ckpt
```

### 2. 자동 전체 훈련

```bash
chmod +x scripts/train_all_stages.sh
./scripts/train_all_stages.sh
```

### 3. 사용자 정의 훈련

```bash
chmod +x scripts/train_custom.sh

# 특정 스테이지 훈련
./scripts/train_custom.sh --stage vision --epochs 5 --batch-size 16

# 전체 훈련
./scripts/train_custom.sh --stage all --data-dir /path/to/data

# 체크포인트에서 재시작
./scripts/train_custom.sh --stage finetune --resume runs/vlm_resampler/checkpoints/best.ckpt

# 도움말
./scripts/train_custom.sh --help
```

## 훈련 단계 설명

### Stage 1: Vision Encoder 훈련
- **목표**: 파노라마 이미지의 시각적 표현 학습
- **손실 함수**: VICReg Loss
- **훈련 대상**: Vision Encoder만
- **특징**: 인접한 파노라마 뷰 간의 일관성 학습

### Stage 2: Resampler 훈련  
- **목표**: 시각적 특징을 언어 모델에 맞는 형태로 변환
- **손실 함수**: Autoregressive Loss
- **훈련 대상**: Vision Encoder + Resampler + Projection Layer
- **특징**: 시각-언어 정렬 학습

### Stage 3: End-to-End Fine-tuning
- **목표**: 최종 멀티모달 성능 최적화
- **손실 함수**: Autoregressive Loss
- **훈련 대상**: Resampler + Projection Layer (Language Model 고정)
- **특징**: 전체 시스템의 통합 최적화

## 설정 파라미터

### 기본 설정
- **Vision Model**: `google/siglip-base-patch16-224`
- **Language Model**: `Qwen/Qwen2.5-0.5B`
- **Resampler**: `mlp`
- **Data**: `data/quic360/train.csv`, `data/quic360/valid.csv`

### Stage별 기본 하이퍼파라미터

| Stage | Epochs | Batch Size | Learning Rate | Max Text Length |
|-------|--------|------------|---------------|-----------------|
| Vision | 3 | 32 | 5e-6 | 32 |
| Resampler | 5 | 16 | 2e-5 | 64 |
| Finetune | 10 | 8 | 1e-5 | 128 |

## 출력 구조

```
runs/
├── vlm_vision/
│   ├── checkpoints/           # Stage 1 체크포인트
│   └── model_final.safetensors
├── vlm_resampler/
│   ├── checkpoints/           # Stage 2 체크포인트
│   └── model_final.safetensors
└── vlm_finetune/
    ├── checkpoints/           # Stage 3 체크포인트
    └── model_final.safetensors  # 최종 모델
```

## 로그 파일

모든 훈련 로그는 `logs/` 디렉토리에 저장됩니다:
- `logs/stage1_vision_YYYYMMDD_HHMMSS.log`
- `logs/stage2_resampler_YYYYMMDD_HHMMSS.log`
- `logs/stage3_finetune_YYYYMMDD_HHMMSS.log`
- `logs/full_pipeline_YYYYMMDD_HHMMSS.log`

## 모니터링

- **WandB**: 모든 훈련 메트릭이 WandB에 자동으로 로깅됩니다
- **로컬 로그**: 콘솔 출력과 파일 로깅이 동시에 진행됩니다
- **체크포인트**: 각 epoch마다 validation loss 기준으로 최적 모델 저장

## 문제 해결

### 메모리 부족
- 배치 크기를 줄여보세요: `--batch-size 8`
- 워커 수를 줄여보세요: `--num-workers 2`

### 데이터 파일 오류
- 데이터 경로를 확인하세요: `--data-dir /correct/path`
- CSV 파일 형식을 확인하세요

### 체크포인트 로딩 실패
- 체크포인트 파일 경로를 확인하세요
- 파일 권한을 확인하세요

## 커스터마이징

스크립트를 수정하여 다음을 변경할 수 있습니다:
- 모델 아키텍처
- 하이퍼파라미터
- 데이터 경로
- 로깅 설정

자세한 설정은 `train.py`의 argparse 옵션을 참조하세요.