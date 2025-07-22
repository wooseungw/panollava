# Configuration Files

## 📁 설정 파일 구조

```
config/
├── model_config.yaml     # 기본 모델 및 데이터 설정
├── vision_stage.yaml     # Stage 1: VICReg 비전 학습
├── resampler_stage.yaml  # Stage 2: Resampler 학습
└── finetune_stage.yaml   # Stage 3: 전체 파인튜닝
```

## 🎯 설정 파일 설명

### `model_config.yaml` - 기본 설정
실제 코드에서 사용하는 모든 기본 설정이 포함됩니다:

- **모델 설정**: SigLIP + Qwen + MLP resampler
- **데이터 설정**: 배치 크기, 워커 수, 텍스트 길이
- **이미지 설정**: 크기, 크롭 전략 (e2p)
- **하드웨어 설정**: GPU, 정밀도, 최적화
- **로깅 설정**: WandB 프로젝트 정보

### Stage별 설정 파일들
각 학습 단계별로 최적화된 설정:

- **`vision_stage.yaml`**: VICReg loss를 사용한 비전 인코더 학습
- **`resampler_stage.yaml`**: Resampler 모듈만 학습  
- **`finetune_stage.yaml`**: 전체 모델 파인튜닝 (낮은 LR)

## 🚀 사용 방법

```bash
# Stage 1: 비전 학습
python train.py --stage vision

# Stage 2: Resampler 학습  
python train.py --stage resampler

# Stage 3: 전체 파인튜닝
python train.py --stage finetune
```

## ⚙️ 설정 오버라이드

명령행에서 주요 설정들을 오버라이드할 수 있습니다:

```bash
python train.py --stage vision \
    --csv_train ./data/my_train.csv \
    --batch_size 8 \
    --lr 2e-4 \
    --epochs 5
```

## 🧹 제거된 불필요한 설정들

- **캐시 디렉토리**: 모델 자동 다운로드 사용
- **복잡한 데이터 증강**: 기본 전처리만 사용
- **사용하지 않는 메트릭**: 현재 구현된 기능만 유지
- **분산 학습 설정**: 단일 GPU 환경에 최적화
- **TensorBoard**: WandB로 통일
