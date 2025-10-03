# VLM Fine-tuning and Evaluation Guide

이 가이드는 Hugging Face Vision-Language Models (VLM)를 LoRA로 fine-tuning하고 평가하는 통합 파이프라인을 설명합니다.

## 파일 구조

```
vlm_finetune_and_eval.py  # 통합 학습 및 평가 스크립트 (LoRA ablation 지원)
scripts/vlm_evaluate.py    # 독립 평가 스크립트
configs/vlm_ablation.yaml  # 학습 설정 파일
```

## 지원 모델

- **LLaVA** (llava, llava_next, llava_gemma)
- **Qwen-VL** (qwen_vl, qwen2_vl, Qwen2.5-VL)
- **BLIP-2** (blip2, blip-2)
- **기타 HuggingFace VLM** (AutoModelForCausalLM 사용)

## 빠른 시작

### 1. 학습 + 자동 평가 (권장)

```bash
python vlm_finetune_and_eval.py \
    --config configs/vlm_ablation.yaml
```

설정 파일 예시 (`configs/vlm_ablation.yaml`):

```yaml
experiment_name: "vlm_lora_ablation"
output_dir: "results/vlm_lora_ablation"

# 데이터 설정
data:
  train_csv: "data/quic360/train.csv"
  val_csv: "data/quic360/valid.csv"
  image_column: "url"
  instruction_column: "instruction"
  response_column: "response"
  num_workers: 4

# 모델 설정
models:
  - name: "qwen_vl_chat"
    hf_model_id: "Qwen/Qwen2.5-VL-0.5B-Instruct"
    model_type: "qwen_vl"
    torch_dtype: "float16"
    
  - name: "llava_1.5_7b"
    hf_model_id: "llava-hf/llava-1.5-7b-hf"
    model_type: "llava"
    torch_dtype: "float16"

# LoRA 조합 ablation
lora_variants:
  - name: "lora_r16"
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: null  # 모델 기본값 사용

  - name: "lora_r32"
    r: 32
    alpha: 64
    dropout: 0.05
    target_modules: null

# 학습 설정
training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 5e-5
  weight_decay: 0.0
  warmup_ratio: 0.03
  logging_steps: 10
  eval_strategy: "no"
  save_strategy: "epoch"
  save_total_limit: 1
  mixed_precision: "fp16"
  gradient_checkpointing: false
  max_grad_norm: 1.0
  seed: 42
  
  # 생성 파라미터 (evaluation용)
  generation_max_new_tokens: 128
  generation_min_new_tokens: 0
  generation_num_beams: 1
  generation_do_sample: false
  generation_temperature: null
  generation_top_p: null
```

### 2. 학습만 수행 (평가 스킵)

설정 파일에서 `evaluation.enabled: false` 설정:

```yaml
# evaluation 섹션 추가
evaluation:
  enabled: false
```

### 3. 학습된 모델 평가만 수행

```bash
python scripts/vlm_evaluate.py \
    --run qwen_vl_chat__lora_r16 \
    --csv data/quic360/test.csv \
    --output-dir eval_results/qwen_vl
```

또는 직접 모델 지정:

```bash
python scripts/vlm_evaluate.py \
    --model-id Qwen/Qwen2.5-VL-0.5B-Instruct \
    --lora-path results/vlm_lora_ablation/qwen_vl_chat__lora_r16/lora_adapter \
    --model-type qwen_vl \
    --csv data/quic360/test.csv \
    --batch-size 2 \
    --max-new-tokens 128
```

## 데이터 포맷

### 학습 데이터 CSV 컬럼

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| url | 이미지 경로 | `data/images/pano_001.jpg` |
| instruction | 질문/지시사항 | `Describe this panoramic view` |
| response | 정답 응답 | `This is an indoor living room...` |

### 평가 데이터 CSV 컬럼

| 컬럼명 | 설명 |
|--------|------|
| url | 이미지 경로 |
| query | 평가 질의 |
| annotation | 참조 정답 (metrics 계산용) |

## 출력 구조

```
results/vlm_lora_ablation/
├── qwen_vl_chat__lora_r16/
│   ├── checkpoints/          # 학습 checkpoint
│   ├── lora_adapter/          # LoRA weights
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── final/                 # 최종 모델 (full)
│   ├── metrics.json           # 학습 metrics
│   ├── predictions.jsonl      # 생성 결과
│   └── generation_metrics.json # 평가 metrics (BLEU, METEOR, ROUGE)
├── qwen_vl_chat__lora_r32/
│   └── ...
└── ablation_summary.json      # 전체 실험 요약
```

## 평가 Metrics

자동으로 계산되는 metrics:
- **Exact Match**: 정확히 일치하는 비율
- **BLEU-4**: N-gram overlap score
- **METEOR**: 의미 기반 유사도
- **ROUGE-L**: Longest Common Subsequence F1
- Token statistics (평균 토큰 수)

## 고급 사용법

### 1. 특정 target modules만 LoRA 적용

```yaml
lora_variants:
  - name: "lora_qkv_only"
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: ["q_proj", "k_proj", "v_proj"]
```

### 2. BFloat16 사용 (A100 GPU)

```yaml
training:
  mixed_precision: "bf16"
  
models:
  - name: "qwen_vl"
    torch_dtype: "bfloat16"
```

### 3. 다중 모델 비교 실험

```yaml
models:
  - name: "qwen_vl_0.5b"
    hf_model_id: "Qwen/Qwen2.5-VL-0.5B-Instruct"
    
  - name: "qwen_vl_2b"
    hf_model_id: "Qwen/Qwen2.5-VL-2B-Instruct"
    
  - name: "llava_1.5_7b"
    hf_model_id: "llava-hf/llava-1.5-7b-hf"
```

각 모델 × 각 LoRA variant 조합이 자동으로 실험됩니다.

### 4. Custom Prompt Template (평가 시)

```bash
python scripts/vlm_evaluate.py \
    --run qwen_vl_chat__lora_r16 \
    --csv data/test.csv \
    --prompt-template "Question: {query}\nAnswer:" \
    --system-message "You are a helpful AI assistant."
```

## Troubleshooting

### OOM (Out of Memory) 에러

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # 증가
  gradient_checkpointing: true
```

### Qwen2.5-VL dynamic resolution 문제

```python
# 이미 자동으로 처리됨 (224x224 fixed)
# processor.image_processor.min_pixels = 224 * 224
# processor.image_processor.max_pixels = 224 * 224
```

### CUDA visible devices 설정

```bash
CUDA_VISIBLE_DEVICES=0,1 python vlm_finetune_and_eval.py --config configs/vlm_ablation.yaml
```

## 참고 자료

- LoRA 논문: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- PEFT 라이브러리: https://github.com/huggingface/peft
- Transformers 문서: https://huggingface.co/docs/transformers

