# VLM 모델 업데이트 가이드

이 문서는 `scripts/evaluate_vlm_models.py`의 지원 모델 변경 사항을 설명합니다.

## 📝 변경 사항 요약

### 추가된 모델

1. **Gemma 3 4B** (`gemma-3-4b`)
   - Model ID: `google/gemma-3-4b-it`
   - 크기: 4B 파라미터
   - 특징: Chat template 사용, 최신 Google VLM
   - 사용 예시:
     ```bash
     python scripts/evaluate_vlm_models.py \
         --data_csv data/quic360/test.csv \
         --models gemma-3-4b \
         --batch_size 2
     ```

2. **Qwen2.5-VL 3B** (`qwen2.5-vl-3b`)
   - Model ID: `Qwen/Qwen2.5-VL-3B-Instruct`
   - 크기: 3B 파라미터
   - 특징: Chat template + `qwen_vl_utils` 필요
   - **추가 설치 필요**:
     ```bash
     pip install qwen-vl-utils
     ```
   - 사용 예시:
     ```bash
     python scripts/evaluate_vlm_models.py \
         --data_csv data/quic360/test.csv \
         --models qwen2.5-vl-3b \
         --batch_size 2
     ```

### 제거된 모델

1. **Qwen-VL-Chat** (`qwen-vl-chat`)
   - 이유: Qwen2.5-VL로 업그레이드

2. **Qwen2-VL-2B** (`qwen2-vl-2b`)
   - 이유: Qwen2.5-VL-3B로 업그레이드

3. **CogVLM2-Llama3-Chat-19B** (`cogvlm2-llama3-chat-19b`)
   - 이유: 모델 크기가 너무 큼 (19B), 메모리 효율성 고려

## 🔧 기술적 변경 사항

### 1. Chat Template 지원 추가

새로운 모델들(Gemma3, Qwen2.5-VL)은 단순 프롬프트 템플릿 대신 **chat template**을 사용합니다.

#### 모델 정의 형식:
```python
"gemma-3-4b": {
    "model_id": "google/gemma-3-4b-it",
    "processor_id": "google/gemma-3-4b-it",
    "model_class": "Gemma3ForConditionalGeneration",
    "processor_class": "AutoProcessor",
    "use_chat_template": True,  # Chat template 사용
},

"qwen2.5-vl-3b": {
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "processor_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "model_class": "Qwen2_5_VLForConditionalGeneration",
    "processor_class": "AutoProcessor",
    "use_chat_template": True,
    "requires_vision_utils": True,  # qwen_vl_utils 필요
},
```

#### Chat Template 처리 로직:
```python
# Gemma3의 경우
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": instruction}
    ]
}]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

# Qwen2.5-VL의 경우
from qwen_vl_utils import process_vision_info

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
```

### 2. 모델 클래스 추가

- `Gemma3ForConditionalGeneration` (transformers)
- `Qwen2_5_VLForConditionalGeneration` (transformers)

### 3. 평가 메트릭 통일

모든 모델이 동일한 평가 지표를 사용하도록 `scripts/eval.py`의 `calculate_evaluation_metrics` 함수를 재사용합니다.

## 📊 현재 지원 모델 목록

| 모델 ID | HuggingFace Model | 크기 | 특징 |
|---------|------------------|------|------|
| `llava-1.5-7b` | llava-hf/llava-1.5-7b-hf | 7B | 기본 프롬프트 |
| `llava-1.6-mistral-7b` | llava-hf/llava-v1.6-mistral-7b-hf | 7B | 기본 프롬프트 |
| `blip2-opt-2.7b` | Salesforce/blip2-opt-2.7b | 2.7B | 기본 프롬프트 |
| `blip2-flan-t5-xl` | Salesforce/blip2-flan-t5-xl | 3B | 기본 프롬프트 |
| `instructblip-vicuna-7b` | Salesforce/instructblip-vicuna-7b | 7B | 기본 프롬프트 |
| `qwen2.5-vl-3b` | Qwen/Qwen2.5-VL-3B-Instruct | 3B | Chat template + vision utils |
| `internvl2-2b` | OpenGVLab/InternVL2-2B | 2B | 기본 프롬프트 |
| `gemma-3-4b` | google/gemma-3-4b-it | 4B | Chat template |

## 🚀 사용 예시

### 모든 경량 모델 평가
```bash
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models blip2-opt-2.7b internvl2-2b qwen2.5-vl-3b gemma-3-4b \
    --batch_size 2 \
    --output_dir results
```

### 최신 모델만 평가
```bash
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models qwen2.5-vl-3b gemma-3-4b \
    --batch_size 1 \
    --max_samples 50
```

### Ablation study 실행
```bash
bash scripts/run_vlm_ablation.sh
```

## 🐛 문제 해결

### Qwen2.5-VL 관련 오류

**에러**: `ModuleNotFoundError: No module named 'qwen_vl_utils'`

**해결**:
```bash
pip install qwen-vl-utils
```

### Gemma3 관련 오류

**에러**: `ImportError: cannot import name 'Gemma3ForConditionalGeneration'`

**해결**:
```bash
pip install --upgrade transformers
```

최소 버전: `transformers >= 4.40.0`

### GPU 메모리 부족

**해결 방법**:
1. 배치 크기 줄이기: `--batch_size 1`
2. 작은 모델 사용: `blip2-opt-2.7b`, `internvl2-2b`
3. 샘플 제한: `--max_samples 50`

## 📚 관련 문서

- [VLM_EVALUATION_GUIDE.md](./VLM_EVALUATION_GUIDE.md): 전체 평가 가이드
- [EVALUATION_SCRIPTS_SUMMARY.md](./EVALUATION_SCRIPTS_SUMMARY.md): 평가 스크립트 요약
- [evaluate_vlm_models.py](../scripts/evaluate_vlm_models.py): 소스 코드

## 🔄 이전 마이그레이션

### qwen-vl-chat → qwen2.5-vl-3b

```bash
# 이전
python scripts/evaluate_vlm_models.py \
    --models qwen-vl-chat

# 새로운
pip install qwen-vl-utils
python scripts/evaluate_vlm_models.py \
    --models qwen2.5-vl-3b
```

### qwen2-vl-2b → qwen2.5-vl-3b

```bash
# 이전
python scripts/evaluate_vlm_models.py \
    --models qwen2-vl-2b

# 새로운
pip install qwen-vl-utils
python scripts/evaluate_vlm_models.py \
    --models qwen2.5-vl-3b
```

### cogvlm2-llama3-chat-19b (제거됨)

19B 파라미터 모델은 메모리 요구사항이 너무 높아 제거되었습니다.
대안으로 `llava-1.5-7b` 또는 `llava-1.6-mistral-7b`를 사용하세요.

## 변경 이력

- **2025-01-XX**: Gemma 3 4B 추가
- **2025-01-XX**: Qwen2.5-VL 3B 추가
- **2025-01-XX**: Qwen-VL-Chat, Qwen2-VL-2B, CogVLM2 제거
- **2025-01-XX**: Chat template 지원 추가

