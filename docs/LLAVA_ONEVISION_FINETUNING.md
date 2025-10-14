# LLaVA-OneVision-4B Fine-tuning Guide

이 가이드는 LLaVA-OneVision-4B를 LoRA로 fine-tuning하고 평가하는 방법을 설명합니다.

## 개요

`finetune_llava_onevision.py` 스크립트는 다음을 수행합니다:

1. **LoRA Fine-tuning**: LLaVA-OneVision-4B 모델을 Quic360 데이터셋으로 LoRA 학습
2. **평가**: Fine-tuned 모델로 테스트 데이터셋 평가 (eval.py 메트릭 사용)
3. **시각화**: 샘플 예측 결과 시각화

## 설치 요구사항

```bash
# 필수 패키지 설치
pip install transformers torch peft datasets qwen-vl-utils

# 평가 메트릭 (선택적)
pip install nltk rouge-score sacrebleu pycocoevalcap
```

## 빠른 시작

### 기본 실행 (Quic360 데이터셋)

```bash
python scripts/finetune_llava_onevision.py \
    --train_csv data/quic360/train.csv \
    --val_csv data/quic360/valid.csv \
    --test_csv data/quic360/test.csv \
    --output_dir ablation/finetuning/llava-onevision-4b \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --lora_rank 16
```

### 평가만 수행 (이미 학습된 모델)

```bash
python scripts/finetune_llava_onevision.py \
    --test_csv data/quic360/test.csv \
    --output_dir ablation/finetuning/llava-onevision-4b \
    --skip_training
```

### 빠른 테스트 (작은 데이터셋)

```bash
python scripts/finetune_llava_onevision.py \
    --train_csv data/quic360/train.csv \
    --val_csv data/quic360/valid.csv \
    --test_csv data/quic360/test.csv \
    --output_dir ablation/finetuning/llava-onevision-4b-test \
    --epochs 1 \
    --batch_size 2 \
    --save_steps 100 \
    --max_new_tokens 64
```

## 주요 파라미터

### 데이터 관련

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--train_csv` | `data/quic360/train.csv` | 학습 데이터 CSV 파일 |
| `--val_csv` | `data/quic360/valid.csv` | 검증 데이터 CSV 파일 |
| `--test_csv` | `data/quic360/test.csv` | 테스트 데이터 CSV 파일 |
| `--image_column` | `url` | 이미지 경로 컬럼명 |
| `--instruction_column` | `query` | 질문 컬럼명 |
| `--response_column` | `annotation` | 정답 컬럼명 |

### 학습 관련

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--epochs` | `3` | 학습 에폭 수 |
| `--batch_size` | `4` | 배치 크기 |
| `--learning_rate` | `2e-4` | 학습률 |
| `--max_length` | `512` | 최대 시퀀스 길이 |
| `--save_steps` | `500` | 체크포인트 저장 간격 |
| `--logging_steps` | `10` | 로깅 간격 |

### LoRA 관련

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--lora_rank` | `16` | LoRA rank (r) |
| `--lora_alpha` | `32` | LoRA alpha (α) |
| `--lora_dropout` | `0.1` | LoRA dropout |

### 평가 관련

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--max_new_tokens` | `128` | 생성 최대 토큰 수 |
| `--num_viz_samples` | `10` | 시각화할 샘플 수 |

### 실행 제어

| 파라미터 | 설명 |
|---------|------|
| `--skip_training` | 학습 건너뛰고 평가만 수행 |
| `--skip_evaluation` | 평가 건너뛰기 |
| `--device` | 디바이스 지정 (기본: cuda) |

## 출력 구조

실행 후 `--output_dir`에 다음 파일들이 생성됩니다:

```
ablation/finetuning/llava-onevision-4b/
├── args.json                          # 실행 인자
├── lora_weights/                      # LoRA 가중치
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── checkpoints/                       # 학습 체크포인트
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── ...
├── training_metrics.json              # 학습 메트릭
└── evaluation/                        # 평가 결과
    ├── eval_metrics.json              # 평가 메트릭 (BLEU, METEOR, ROUGE, etc.)
    ├── predictions.csv                # 예측 결과 (cleaned + raw)
    ├── llava_onevision_4b_predictions_*.csv  # eval.py 생성
    ├── llava_onevision_4b_metrics_*.json     # eval.py 생성
    └── visualizations/                # 시각화 결과
        ├── sample_000.png
        ├── sample_001.png
        └── ...
```

## 평가 메트릭

스크립트는 `eval.py`의 `calculate_evaluation_metrics` 함수를 사용하여 다음 메트릭을 계산합니다:

- **BLEU-4**: sacrebleu 사용 (표준 토큰화)
- **METEOR**: 단어 동의어 및 어간 매칭
- **ROUGE-L**: 최장 공통 부분 수열
- **SPICE**: 시맨틱 프로포지션 비교
- **CIDEr**: TF-IDF 가중치 n-gram 일치도
- **CLIP-S**: 이미지-텍스트 유사도 (선택적)
- **RefCLIP-S**: 참조 기반 CLIP 점수 (선택적)

### 텍스트 정리 (basic_cleanup)

평가 전에 `basic_cleanup` 함수로 다음을 제거합니다:
- 특수 토큰: `<image>`, `<|im_start|>`, etc.
- 역할 태그: `USER:`, `ASSISTANT:`, etc.
- 과도한 공백

대소문자와 구두점은 보존되어 실제 모델 성능을 반영합니다.

## 데이터셋 형식

CSV 파일은 다음 컬럼을 포함해야 합니다:

| 컬럼명 | 설명 | 필수 |
|--------|------|------|
| `url` | 이미지 파일 경로 | O |
| `query` | 질문/instruction | O |
| `annotation` | 정답 응답 | O |

예시:
```csv
url,query,annotation
data/quic360/images/pano_001.jpg,Describe this panorama image.,This is a panoramic view of a living room with modern furniture.
data/quic360/images/pano_002.jpg,What objects do you see?,I see a sofa, a coffee table, and a TV.
```

## LoRA 타겟 모듈

LLaVA-OneVision-4B (Qwen2.5 기반)의 기본 LoRA 타겟 모듈:

```python
target_modules = [
    "q_proj",      # Query projection
    "k_proj",      # Key projection
    "v_proj",      # Value projection
    "o_proj",      # Output projection
    "gate_proj",   # MLP gate
    "up_proj",     # MLP up
    "down_proj",   # MLP down
]
```

## 고급 사용법

### 1. 커스텀 LoRA 설정

```bash
python scripts/finetune_llava_onevision.py \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --learning_rate 1e-4
```

### 2. 긴 시퀀스 학습

```bash
python scripts/finetune_llava_onevision.py \
    --max_length 1024 \
    --batch_size 2 \
    --max_new_tokens 256
```

### 3. 빠른 반복 (자주 저장)

```bash
python scripts/finetune_llava_onevision.py \
    --save_steps 100 \
    --logging_steps 5 \
    --epochs 5
```

## 시각화

평가 후 `evaluation/visualizations/` 디렉토리에 샘플 예측 결과가 저장됩니다:

- 이미지
- Instruction
- Prediction (파란색)
- Reference (초록색)

## 문제 해결

### 1. CUDA Out of Memory

```bash
# 배치 크기 줄이기
python scripts/finetune_llava_onevision.py --batch_size 1

# 시퀀스 길이 줄이기
python scripts/finetune_llava_onevision.py --max_length 256
```

### 2. qwen_vl_utils 없음

```bash
# qwen_vl_utils 설치
pip install qwen-vl-utils

# 또는 최신 버전 설치
pip install git+https://github.com/QwenLM/Qwen-VL-utils.git
```

### 3. peft 없음

```bash
pip install peft
```

### 4. eval.py 메트릭 없음

```bash
pip install nltk rouge-score sacrebleu pycocoevalcap
```

## 성능 최적화

### 메모리 최적화

- **Gradient Checkpointing**: 자동 활성화 (메모리 절약)
- **FP16**: CUDA 사용 시 자동 활성화
- **Batch Size**: GPU 메모리에 맞게 조정

### 속도 최적화

- **Num Workers**: `dataloader_num_workers=4` (CPU 코어 수에 맞게 조정)
- **Save Steps**: 큰 값으로 설정하여 I/O 줄이기

## 참고 사항

1. **모델 크기**: LLaVA-OneVision-4B는 ~4B 파라미터
2. **LoRA 크기**: rank=16일 때 ~수십 MB
3. **학습 시간**: Quic360 전체 데이터셋 기준 ~1-2시간 (1x A100)
4. **평가 시간**: 테스트셋 기준 ~10-20분

## 인용

LLaVA-OneVision 모델을 사용하는 경우:

```bibtex
@article{li2024llava-onevision,
  title={LLaVA-OneVision: Easy Visual Task Transfer},
  author={Li, Bo and Zhang, Yuanhan and Guo, Dong and Zhang, Renrui and Li, Feng and Zhang, Hao and Zhang, Kaichen and Li, Yanwei and Liu, Ziwei and Li, Chunyuan},
  journal={arXiv preprint arXiv:2408.03326},
  year={2024}
}
```

## 라이선스

이 스크립트는 PanoLLaVA 프로젝트의 일부이며 동일한 라이선스를 따릅니다.
