# Baseline VLM 실험 결과

## 실험 조건

| 항목 | 설정 |
|------|------|
| **Dataset** | QuIC-360 (test split, 5,349 samples) |
| **Fine-tuning** | LoRA 1 epoch (모든 모델 동일) |
| **Image size** | 256×256 |
| **Max generation tokens** | 128 |
| **Max input length** | 512 |
| **GPU** | 1×A6000 (48GB) |
| **Input format** | ERP (Equirectangular Projection) 단일 이미지 |

## 결과

### 주요 지표 비교

| Model | Params | BLEU-4 ↑ | METEOR ↑ | ROUGE-L ↑ | CIDEr ↑ | SPICE ↑ |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gemma3-4B** | 4B | **0.0421** | **0.1085** | **0.2449** | **0.3383** | **0.1640** |
| InternVL3.5-2B | 2B | 0.0403 | 0.1096 | 0.2402 | 0.3054 | 0.1566 |
| Qwen2.5-VL-3B | 3B | 0.0382 | 0.1113 | 0.2334 | 0.2809 | 0.1435 |
| Qwen2-VL-2B | 2B | 0.0337 | 0.1005 | 0.2301 | 0.2447 | 0.1449 |
| BLIP2-OPT-2.7B | 2.7B | 0.0051 | 0.0448 | 0.1230 | 0.0715 | 0.0848 |
| InternVL3.5-1B | 1B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### 전체 BLEU 스코어

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|-------|:---:|:---:|:---:|:---:|
| Gemma3-4B | 0.2556 | 0.1341 | 0.0742 | 0.0421 |
| InternVL3.5-2B | 0.2714 | 0.1387 | 0.0741 | 0.0403 |
| Qwen2.5-VL-3B | 0.2871 | 0.1386 | 0.0722 | 0.0382 |
| Qwen2-VL-2B | 0.2492 | 0.1238 | 0.0640 | 0.0337 |
| BLIP2-OPT-2.7B | 0.0724 | 0.0274 | 0.0112 | 0.0051 |
| InternVL3.5-1B | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### 평균 생성 토큰 수

| Model | Avg Pred Tokens | Avg Ref Tokens | 비율 |
|-------|:---:|:---:|:---:|
| Qwen2.5-VL-3B | 18.5 | 19.8 | 93% |
| InternVL3.5-2B | 17.1 | 19.8 | 86% |
| Qwen2-VL-2B | 15.9 | 19.8 | 80% |
| Gemma3-4B | 15.2 | 19.8 | 77% |
| BLIP2-OPT-2.7B | 9.2 | 19.8 | 46% |
| InternVL3.5-1B | 0.0 | 19.8 | 0% |

## 분석

### Top-tier (CIDEr > 0.25)

- **Gemma3-4B**: 최고 성능. BLEU-4, CIDEr, SPICE 모두 1위. 파라미터 수(4B)가 가장 크지만, 파라미터 대비 성능이 가장 효율적.
- **InternVL3.5-2B**: Gemma3에 근접. METEOR에서는 가장 높은 점수(0.1096). 2B로 가성비 우수.
- **Qwen2.5-VL-3B**: BLEU-1(0.2871)이 가장 높아 단어 단위 매칭은 우수하지만 긴 구문 정확도(BLEU-4)는 Gemma3보다 낮음.
- **Qwen2-VL-2B**: 전체적으로 Qwen2.5보다 약간 낮음. 세대 차이 확인.

### Low-tier

- **BLIP2-OPT-2.7B**: 생성 길이가 절반 수준(9.2 vs 19.8 tokens). 구조적 한계 — Q-Former 기반이라 panorama ERP에 비적합. 캡션이 짧고 반복적.
- **InternVL3.5-1B**: 빈 예측(avg_pred_tokens=0.0). 코드에서 empty prediction 버그 발견 후 수정 완료(`src/cora/baseline/finetune.py:735-740`), 그러나 **재평가 미실행**.

## 알려진 이슈

| 모델 | 이슈 | 상태 |
|------|------|------|
| InternVL3.5-1B | 빈 예측 출력 (generation 버그) | 코드 수정 완료, **재평가 필요** |
| Gemma3-4B | HuggingFace 403 오류 (권한 문제) | 해결 완료 |
| BLIP2-OPT-2.7B | max_length=256 부족 → truncation | MAX_LENGTH=512로 수정 후 재학습 완료 |

## 실험 재현

```bash
# 전체 baseline 실행
CUDA_VISIBLE_DEVICES=0 bash scripts/run_baseline.sh

# 특정 모델만
CUDA_VISIBLE_DEVICES=0 bash scripts/run_baseline.sh --models "qwen2-vl-2b,gemma3-4b"

# 옵션 변경
CUDA_VISIBLE_DEVICES=0 bash scripts/run_baseline.sh \
  --max-tokens 128 --image-size 256 --max-length 512
```

## TODO

- [ ] InternVL3.5-1B 재평가 (빈 예측 버그 수정 후)
- [ ] CORA 최종 결과와 비교 테이블 작성
- [ ] Gemma3-4B fine-tuning 추가 (현재 zero-shot만)

## 파일 위치

```
runs/baseline/
├── _shared_data/
│   ├── train.csv
│   └── test.csv
├── qwen2-vl-2b_img256_tok128/
│   ├── qwen2-vl-2b/checkpoints/      # LoRA adapter
│   └── eval/
│       ├── predictions.csv            # 5,349 predictions
│       └── metrics.json               # 평가 결과
├── qwen25-vl-3b_img256_tok128/
├── gemma3-4b_img256_tok128/
├── internvl3_5-1b_img256_tok128/      # ⚠️ 빈 예측
├── internvl3_5-2b_img256_tok128/
└── blip2-opt-2.7b_img256_tok128/
```
