# PanoLLaVA Training Scripts

이 디렉토리는 PanoLLaVA 모델의 3단계 훈련을 위한 스크립트들을 포함합니다.

## 파일 구조

### Linux/macOS (Bash Scripts)
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
└── test_config.sh              # 설정 테스트
```

### 🆕 Windows (Batch Files)
```
scripts/
├── config.bat                  # 윈도우용 공통 설정 파일
├── stage1_vision_train.bat     # Stage 1: Vision Encoder 훈련
├── stage2_resampler_train.bat  # Stage 2: Resampler 훈련  
├── stage3_finetune_train.bat   # Stage 3: End-to-End Fine-tuning
├── train_all_stages.bat        # 전체 3단계 자동 훈련
├── train_custom.bat            # 사용자 정의 훈련
├── eval_finetune.bat           # Finetune 모델 평가
├── eval_resampler.bat          # Resampler 모델 평가
├── eval_compare.bat            # 모델 비교 평가
└── test_config.bat             # 설정 테스트
```

## 🚀 새로운 중앙화된 설정 관리

### config.sh / config.bat
모든 스크립트는 공통 설정 파일에서 설정을 로드합니다:
- 모델 설정 (Vision/Language 모델명)
- 데이터 경로
- 학습 하이퍼파라미터
- GPU 및 환경 설정
- 디렉토리 구조

### 설정 수정 방법
1. **전역 설정 변경**: `config.sh` 파일을 직접 수정
2. **스크립트별 오버라이드**: 각 스크립트에서 필요시 설정 오버라이드

## 사용법

### Linux/macOS 사용법

#### 1. 순차적 3단계 훈련

**Stage 1: Vision Encoder 훈련**
```bash
chmod +x scripts/stage1_vision_train.sh
./scripts/stage1_vision_train.sh
```

**Stage 2: Resampler 훈련**
```bash
chmod +x scripts/stage2_resampler_train.sh
./scripts/stage2_resampler_train.sh
```

**Stage 3: End-to-End Fine-tuning**
```bash
chmod +x scripts/stage3_finetune_train.sh
./scripts/stage3_finetune_train.sh
```

#### 2. 자동 전체 훈련
```bash
chmod +x scripts/train_all_stages.sh
./scripts/train_all_stages.sh
```

#### 3. 사용자 정의 훈련
```bash
chmod +x scripts/train_custom.sh
./scripts/train_custom.sh --stage vision --epochs 5 --batch-size 16
```

### 🆕 Windows 사용법

#### 1. 순차적 3단계 훈련

**Stage 1: Vision Encoder 훈련**
```cmd
scripts\stage1_vision_train.bat
```

**Stage 2: Resampler 훈련**
```cmd
scripts\stage2_resampler_train.bat
```

**Stage 3: End-to-End Fine-tuning**
```cmd
scripts\stage3_finetune_train.bat
```

#### 2. 자동 전체 훈련
```cmd
scripts\train_all_stages.bat
```

#### 3. 사용자 정의 훈련
```cmd
scripts\train_custom.bat --stage vision --epochs 5 --batch-size 16
```

### 공통 고급 사용법

#### Linux/macOS 고급 옵션
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

#### Windows 고급 옵션
```cmd
REM 특정 스테이지 훈련
scripts\train_custom.bat --stage vision --epochs 5 --batch-size 16

REM 전체 훈련
scripts\train_custom.bat --stage all --data-dir C:\path\to\data

REM 체크포인트에서 재시작
scripts\train_custom.bat --stage finetune --resume runs\vlm_resampler\checkpoints\best.ckpt

REM 도움말
scripts\train_custom.bat --help
```

### 모델 평가

#### Linux/macOS 평가
```bash
# Finetune 모델 평가
./scripts/eval_finetune.sh data/quic360/test.csv

# Resampler 모델 평가
./scripts/eval_resampler.sh data/quic360/test.csv

# 모델 비교 평가
./scripts/eval_compare.sh data/quic360/test.csv
```

#### Windows 평가
```cmd
REM Finetune 모델 평가
scripts\eval_finetune.bat data\quic360\test.csv

REM Resampler 모델 평가
scripts\eval_resampler.bat data\quic360\test.csv

REM 모델 비교 평가
scripts\eval_compare.bat data\quic360\test.csv
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

## 🆕 Windows 특별 사항

### 환경 요구사항
- Python 3.8 이상
- CUDA 지원 GPU (권장)
- PowerShell (타임스탬프 생성용)

### 윈도우 특별 기능
- **에러 처리**: 각 단계에서 오류 발생 시 자동으로 일시정지
- **경로 처리**: 윈도우 경로 형식 자동 지원 (백슬래시)
- **배치 파일 호출**: `call` 명령어로 설정 파일 로드
- **환경 변수**: Windows 환경 변수 형식 사용 (`%VAR%`)

### 윈도우 사용 팁
1. **관리자 권한**: GPU 사용 시 관리자 권한으로 명령 프롬프트 실행 권장
2. **긴 경로**: 파일 경로가 길 경우 따옴표 사용: `"C:\very\long\path\to\file"`
3. **워커 수**: Windows에서는 `NUM_WORKERS=8`로 기본 설정 (Linux보다 낮음)
4. **일시정지**: 각 스크립트 실행 후 `pause` 명령으로 결과 확인 가능

### 설정 테스트
```cmd
REM 설정이 제대로 로드되는지 테스트
scripts\test_config.bat
```

## 커스터마이징

스크립트를 수정하여 다음을 변경할 수 있습니다:
- 모델 아키텍처
- 하이퍼파라미터
- 데이터 경로
- 로깅 설정

### 설정 파일 수정
- **Linux/macOS**: `scripts/config.sh` 편집
- **Windows**: `scripts/config.bat` 편집

자세한 설정은 `train.py`의 argparse 옵션을 참조하세요.