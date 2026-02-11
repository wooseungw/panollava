# VLM 모델 평가 가이드

이 스크립트는 HuggingFace의 다양한 Vision-Language 모델들을 동일한 데이터셋으로 평가하여 성능을 비교합니다.

## 특징

- **학습 없이 평가**: Pre-trained 모델을 그대로 사용하여 zero-shot 성능 측정
- **이미지 크기 고정**: 모든 이미지를 224x224로 리사이즈하여 공정한 비교
- **동일한 평가 지표**: BLEU-4, METEOR, ROUGE-L, Exact Match 등 vlm_finetune_and_eval.py와 동일한 메트릭 사용

## 지원 모델

현재 지원하는 VLM 모델들:

| 모델 ID | HuggingFace Model | 크기 | 설명 |
|---------|------------------|------|------|
| `llava-1.5-7b` | llava-hf/llava-1.5-7b-hf | 7B | LLaVA 1.5 |
| `llava-1.6-mistral-7b` | llava-hf/llava-v1.6-mistral-7b-hf | 7B | LLaVA 1.6 (Mistral) |
| `blip2-opt-2.7b` | Salesforce/blip2-opt-2.7b | 2.7B | BLIP-2 (OPT) |
| `blip2-flan-t5-xl` | Salesforce/blip2-flan-t5-xl | 3B | BLIP-2 (Flan-T5) |
| `instructblip-vicuna-7b` | Salesforce/instructblip-vicuna-7b | 7B | InstructBLIP |
| `qwen2.5-vl-3b` | Qwen/Qwen2.5-VL-3B-Instruct | 3B | Qwen2.5-VL |
| `internvl2-2b` | OpenGVLab/InternVL2-2B | 2B | InternVL2 |
| `gemma-3-4b` | google/gemma-3-4b-it | 4B | Gemma 3 |

## 설치

필요한 패키지 설치:

```bash
pip install transformers pillow pandas numpy tqdm nltk rouge-score torch
```

**Qwen2.5-VL을 사용하는 경우 추가 설치:**

```bash
pip install qwen-vl-utils
```

NLTK 데이터 다운로드:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

## 사용법

### 기본 사용

```bash
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models llava-1.5-7b blip2-opt-2.7b \
    --output_dir eval_results/vlm_comparison
```

### 전체 옵션

```bash
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models llava-1.5-7b llava-1.6-mistral-7b blip2-opt-2.7b blip2-flan-t5-xl \
    --output_dir eval_results/vlm_comparison \
    --batch_size 4 \
    --max_samples 100 \
    --device cuda \
    --image_size 224 \
    --max_new_tokens 128 \
    --log_level INFO
```

### 단일 모델 평가

```bash
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models llava-1.5-7b \
    --max_samples 50
```

### 모든 경량 모델 평가 (메모리 효율적)

```bash
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models blip2-opt-2.7b qwen2-vl-2b internvl2-2b \
    --batch_size 2
```

## 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--data_csv` | str | **필수** | 평가할 데이터 CSV 파일 경로 |
| `--models` | list[str] | `[llava-1.5-7b, blip2-opt-2.7b]` | 평가할 모델 ID 리스트 |
| `--output_dir` | str | `eval_results` | 결과 저장 디렉토리 |
| `--batch_size` | int | `1` | 배치 크기 (GPU 메모리에 따라 조정) |
| `--max_samples` | int | `None` | 평가할 최대 샘플 수 (디버깅용) |
| `--device` | str | `cuda` | 사용할 디바이스 (cuda/cpu) |
| `--image_size` | int | `224` | 이미지 리사이즈 크기 (정사각형) |
| `--max_new_tokens` | int | `128` | 생성할 최대 토큰 수 |
| `--log_level` | str | `INFO` | 로그 레벨 (DEBUG/INFO/WARNING/ERROR) |

## 데이터 형식

CSV 파일은 다음 컬럼을 포함해야 합니다:

```csv
url,instruction,response
path/to/image1.jpg,"Describe the image.","A beautiful landscape with mountains."
path/to/image2.jpg,"What is in the image?","A cat sitting on a chair."
```

- `url`: 이미지 파일 경로
- `instruction`: 모델에 전달할 질문/명령
- `response`: 정답 (reference)

## 출력 파일

평가 완료 후 다음 파일들이 생성됩니다:

### 1. 메트릭 JSON (`{model_name}_metrics.json`)

```json
{
  "model_name": "llava-1.5-7b",
  "model_id": "llava-hf/llava-1.5-7b-hf",
  "num_samples": 100,
  "image_size": "224x224",
  "metrics": {
    "samples": 100.0,
    "exact_match": 0.05,
    "avg_pred_tokens": 15.3,
    "avg_ref_tokens": 12.8,
    "bleu4": 0.234,
    "meteor": 0.312,
    "rougeL": 0.456
  }
}
```

### 2. 예측 결과 CSV (`{model_name}_predictions.csv`)

| image_path | instruction | reference | prediction |
|------------|-------------|-----------|------------|
| img1.jpg | Describe... | A cat... | The image shows... |
| img2.jpg | What is... | A dog... | I see a... |

### 3. 전체 요약 (`all_models_summary.json`)

모든 모델의 메트릭을 포함한 종합 결과

## 평가 지표

### 1. Exact Match
- 예측이 정답과 완전히 일치하는 비율
- 범위: 0.0 ~ 1.0 (높을수록 좋음)

### 2. BLEU-4
- 4-gram 기반 precision 메트릭
- 기계 번역에서 유래, 생성된 텍스트의 n-gram 겹침 측정
- 범위: 0.0 ~ 1.0 (높을수록 좋음)

### 3. METEOR
- 단어 순서와 동의어를 고려한 메트릭
- BLEU보다 사람의 판단과 높은 상관관계
- 범위: 0.0 ~ 1.0 (높을수록 좋음)

### 4. ROUGE-L
- Longest Common Subsequence 기반 F1 점수
- 요약 품질 평가에 주로 사용
- 범위: 0.0 ~ 1.0 (높을수록 좋음)

### 5. 토큰 통계
- `avg_pred_tokens`: 예측 텍스트의 평균 토큰 수
- `avg_ref_tokens`: 정답 텍스트의 평균 토큰 수

## 예제

### 예제 1: 빠른 테스트

```bash
# 2개 모델, 50개 샘플만 평가
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models llava-1.5-7b blip2-opt-2.7b \
    --max_samples 50 \
    --output_dir eval_results/quick_test
```

### 예제 2: 전체 평가 (경량 모델들)

```bash
# 메모리 효율적인 모델들로 전체 평가
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models blip2-opt-2.7b qwen2.5-vl-3b internvl2-2b gemma-3-4b \
    --batch_size 2 \
    --output_dir eval_results/lightweight_models
```

### 예제 3: 고성능 모델 평가

```bash
# 대형 모델 평가 (GPU 메모리 많이 필요)
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models llava-1.5-7b instructblip-vicuna-7b \
    --batch_size 1 \
    --max_new_tokens 256 \
    --output_dir eval_results/large_models
```

## 이미지 크기 고정 (224x224)

모든 입력 이미지는 평가 전에 자동으로 224x224 크기로 리사이즈됩니다:

```python
img = img.resize((224, 224), Image.Resampling.LANCZOS)
```

이를 통해:
- 모든 모델이 동일한 크기의 입력을 받음
- 공정한 성능 비교 가능
- GPU 메모리 사용량 예측 가능

다른 크기를 원하면 `--image_size` 옵션 사용:

```bash
python scripts/evaluate_vlm_models.py \
    --data_csv data/quic360/test.csv \
    --models llava-1.5-7b \
    --image_size 384  # 384x384로 변경
```

## 문제 해결

### GPU 메모리 부족

```bash
# 배치 크기 줄이기
--batch_size 1

# 또는 작은 모델 사용
--models blip2-opt-2.7b qwen2.5-vl-3b internvl2-2b
```

### 모델 다운로드 실패

HuggingFace 토큰이 필요한 경우:

```bash
export HF_TOKEN=your_token_here
python scripts/evaluate_vlm_models.py ...
```

### NLTK 데이터 오류

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

## 성능 비교 예시

평가 완료 후 터미널에 출력되는 결과 예시:

```
============================================================
모델 성능 비교
============================================================

llava-1.5-7b:
  samples: 100.0000
  exact_match: 0.0500
  avg_pred_tokens: 15.3000
  avg_ref_tokens: 12.8000
  bleu4: 0.2340
  meteor: 0.3120
  rougeL: 0.4560

blip2-opt-2.7b:
  samples: 100.0000
  exact_match: 0.0300
  avg_pred_tokens: 10.2000
  avg_ref_tokens: 12.8000
  bleu4: 0.1890
  meteor: 0.2670
  rougeL: 0.3910
```

## 추가 모델 등록

새로운 모델을 추가하려면 `scripts/evaluate_vlm_models.py`의 `VLM_MODELS` 딕셔너리에 등록:

```python
VLM_MODELS = {
    # ... 기존 모델들 ...
    
    "your-model-name": {
        "model_id": "hf-org/model-name",
        "processor_id": "hf-org/model-name",
        "model_class": "AutoModelForVision2Seq",
        "processor_class": "AutoProcessor",
        "prompt_template": "USER: <image>\n{instruction}\nASSISTANT:",
    },
}
```

## 참고

- 원본 학습 코드: `vlm_finetune_and_eval.py`
- 평가 지표 구현: 동일한 `compute_text_metrics` 함수 사용
- 모든 모델은 HuggingFace Transformers 라이브러리 사용
