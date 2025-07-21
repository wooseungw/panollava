# Panorama VLM Training Scripts

YAML 기반 설정 시스템을 사용한 3단계 학습 파이프라인

## 📁 구조

```
scripts/
├── train_vision.sh       # Stage 1: Vision 학습 (Linux/macOS)
├── train_resampler.sh    # Stage 2: Resampler 학습 (Linux/macOS)  
├── train_finetune.sh     # Stage 3: LoRA 파인튜닝 (Linux/macOS)
├── train_all.sh          # 전체 파이프라인 실행 (Linux/macOS)
├── train_vision.bat      # Stage 1: Vision 학습 (Windows)
├── train_resampler.bat   # Stage 2: Resampler 학습 (Windows)
├── train_finetune.bat    # Stage 3: LoRA 파인튜닝 (Windows)
├── train_all.bat         # 전체 파이프라인 실행 (Windows)
└── old_scripts/          # 기존 스크립트 백업
```

## 🚀 사용법

### 환경 설정

**필수 환경 변수:**
```bash
export CSV_TRAIN="path/to/train.csv"
export CSV_VAL="path/to/val.csv"
```

**선택적 환경 변수:**
```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="my-panorama-vlm"
```

### Linux/macOS

**개별 Stage 실행:**
```bash
# Stage 1: Vision Encoder 학습
./scripts/train_vision.sh

# Stage 2: Resampler 학습  
./scripts/train_resampler.sh

# Stage 3: LoRA 파인튜닝
./scripts/train_finetune.sh
```

**전체 파이프라인 실행:**
```bash
./scripts/train_all.sh
```

### Windows

**개별 Stage 실행:**
```cmd
REM Stage 1: Vision Encoder 학습
scripts\train_vision.bat

REM Stage 2: Resampler 학습
scripts\train_resampler.bat

REM Stage 3: LoRA 파인튜닝
scripts\train_finetune.bat
```

**전체 파이프라인 실행:**
```cmd
scripts\train_all.bat
```

## ⚙️ 설정 커스터마이징

### 1. YAML 설정 파일 수정

각 stage별 설정은 `configs/stages/` 디렉토리에서 수정:

```
configs/
├── base.yaml           # 기본 설정
└── stages/
    ├── vision.yaml     # Stage 1 설정
    ├── resampler.yaml  # Stage 2 설정
    └── finetune.yaml   # Stage 3 설정
```

### 2. 환경 변수로 오버라이드

```bash
# 학습률 변경
export PANO_VLM_TRAINING_LEARNING_RATE=1e-4

# 배치 크기 변경
export PANO_VLM_DATA_BATCH_SIZE=8

# LoRA 설정 변경
export PANO_VLM_MODEL_LORA_R=16
export PANO_VLM_MODEL_LORA_ALPHA=32
```

### 3. 명령행 인자로 오버라이드

```bash
./scripts/train_finetune.sh --lr 2e-4 --batch-size 4
```

## 📊 LoRA 설정

Stage 3 파인튜닝에서 사용되는 LoRA 파라미터:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `r` | 16 | LoRA rank (낮을수록 효율적) |
| `alpha` | 32 | 학습 가중치 (보통 r의 2배) |
| `dropout` | 0.1 | 정규화 드롭아웃 |

## 🔧 고급 사용법

### 커스텀 설정 파일 사용

```bash
python train.py \
    --config-stage finetune \
    --config-override my_custom_config.yaml
```

### 체크포인트에서 재시작

```bash
./scripts/train_resampler.sh --resume-from ./runs/e2p_vision_mlp/best.ckpt
```

### 다중 GPU 사용

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
./scripts/train_all.sh
```

## 📈 결과 확인

학습 완료 후 결과는 다음 위치에 저장됩니다:

```
runs/
├── e2p_vision_mlp/     # Stage 1 결과
├── e2p_resampler_mlp/  # Stage 2 결과
└── e2p_finetune_mlp/   # Stage 3 결과 (LoRA 어댑터)
```

## 🐛 문제 해결

### 일반적인 문제들

1. **CUDA OOM 에러**
   ```bash
   export PANO_VLM_DATA_BATCH_SIZE=1
   ```

2. **체크포인트 없음 에러**
   - 이전 stage가 완료되었는지 확인
   - `--resume-from` 인자로 수동 지정

3. **YAML 설정 에러**
   - `configs/` 디렉토리 존재 확인
   - YAML 문법 검증

### 로그 확인

```bash
tail -f training.log
```

## 📝 마이그레이션 노트

기존 스크립트에서 새로운 YAML 기반 시스템으로 마이그레이션:

- ✅ 기존 스크립트는 `old_scripts/`로 백업됨
- ✅ 모든 기능이 새로운 시스템에서 지원됨
- ✅ 환경 변수 이름이 `PANO_VLM_*` 형식으로 변경됨
- ✅ 더 나은 설정 관리와 오버라이드 기능 제공

## 🆕 새로운 기능

### YAML 기반 설정 시스템
- 계층적 설정 관리 (base + stage override)
- 환경 변수를 통한 런타임 오버라이드
- 타입 안전성과 설정 검증

### LoRA 지원
- Stage 3에서 효율적인 파인튜닝
- 메모리 사용량 대폭 감소
- 빠른 수렴과 좋은 성능

### 자동 체크포인트 탐지
- 이전 stage 결과를 자동으로 찾아서 연결
- 수동 지정 없이도 파이프라인 실행 가능

### 향상된 로깅
- WandB 통합 로깅
- 설정 정보 자동 기록
- 디버깅 정보 출력

## 🔄 사용 예시

### 빠른 시작
```bash
# 데이터 준비
export CSV_TRAIN="data/quic360/train.csv"
export CSV_VAL="data/quic360/valid.csv"

# 전체 파이프라인 실행
./scripts/train_all.sh
```

### 커스텀 LoRA 설정으로 파인튜닝
```bash
# LoRA 파라미터 설정
export PANO_VLM_MODEL_LORA_R=32
export PANO_VLM_MODEL_LORA_ALPHA=64
export PANO_VLM_MODEL_LORA_DROPOUT=0.05

# Stage 3만 실행
./scripts/train_finetune.sh
```

### 개발 모드 (작은 배치 크기)
```bash
export PANO_VLM_DATA_BATCH_SIZE=1
export PANO_VLM_TRAINING_EPOCHS=1
./scripts/train_vision.sh
```