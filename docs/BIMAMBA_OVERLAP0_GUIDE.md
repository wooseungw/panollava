# BiMamba Overlap=0 학습 가이드

## 개요

BiMamba 리샘플러를 사용하여 overlap=0으로 Stage 2(resampler)와 Stage 3(finetune)만 학습합니다.
Stage 1(vision)은 기존에 학습된 체크포인트를 재사용합니다.

## 사전 준비

### 1. Vision 체크포인트 확인

```bash
# 사용 가능한 vision 체크포인트 확인
ls -lh runs/*/vision/*/best.ckpt
```

**현재 사용 가능한 체크포인트:**
- `runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/vision/anyres-e2p_bimamba/best.ckpt` (BiMamba, overlap=0.5)
- `runs/siglip2-so400m_Qwen3_mlp_anyres-e2p_PE/vision/anyres-e2p_mlp/best.ckpt` (MLP, overlap=0.5)

### 2. 설정 파일 확인

`configs/bimamba_overlap0.yaml`의 주요 설정:

```yaml
models:
  resampler_type: "bimamba"  # BiMamba 리샘플러

image_processing:
  overlap_ratio: 0.0  # Overlap 0으로 변경

training:
  stages: ["resampler", "finetune"]  # Stage 2-3만 실행
```

## 실행 방법

### 방법 1: 간편 스크립트 실행 (권장)

```bash
bash scripts/train_bimamba_overlap0.sh
```

### 방법 2: 수동 실행

```bash
# Vision 체크포인트 경로 지정
VISION_CKPT="runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/vision/anyres-e2p_bimamba/best.ckpt"

# 학습 실행
python scripts/train.py \
    --config configs/bimamba_overlap0.yaml \
    --resume "$VISION_CKPT"
```

### 방법 3: 다른 Vision 체크포인트 사용

```bash
# MLP로 학습된 vision 사용
VISION_CKPT="runs/siglip2-so400m_Qwen3_mlp_anyres-e2p_PE/vision/anyres-e2p_mlp/best.ckpt"

python scripts/train.py \
    --config configs/bimamba_overlap0.yaml \
    --resume "$VISION_CKPT"
```

## 결과 확인

학습 완료 후 결과는 다음 위치에 저장됩니다:

```
runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/
├── resampler/
│   └── anyres-e2p_bimamba/
│       ├── best.ckpt
│       ├── last.ckpt
│       ├── config.yaml
│       └── checkpoint_metadata.json
└── finetune/
    └── anyres-e2p_bimamba/
        ├── best.ckpt
        ├── last.ckpt
        ├── config.yaml
        ├── checkpoint_metadata.json
        └── lora_weights/  (LoRA 사용 시)
```

## 주요 차이점

| 항목 | 기존 (overlap=0.5) | 새로운 (overlap=0.0) |
|------|-------------------|---------------------|
| Stage 1 (vision) | 재사용 | 재사용 |
| Stage 2 (resampler) | 새로 학습 | 새로 학습 |
| Stage 3 (finetune) | 새로 학습 | 새로 학습 |
| Overlap | 0.5 | 0.0 |
| 타일 수 (90° FOV) | 9개 (8+1 global) | 5개 (4+1 global) |
| 메모리 사용량 | 높음 | 낮음 |
| 학습 속도 | 느림 | 빠름 |

## 평가 실행

학습 완료 후 평가:

```bash
# Best checkpoint로 평가
python scripts/eval.py \
    --config runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/config.yaml \
    --csv-input data/quic360/test.csv
```

결과 저장 위치:
```
results/eval_results/
└── siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/
    └── predictions_20251027_HHMMSS/
        ├── predictions.csv
        └── metrics.json
```

## 트러블슈팅

### 에러: Vision 체크포인트를 찾을 수 없음

```bash
# 사용 가능한 체크포인트 확인
ls -lh runs/*/vision/*/best.ckpt

# 스크립트의 VISION_CHECKPOINT 경로 수정
nano scripts/train_bimamba_overlap0.sh
```

### 에러: OOM (Out of Memory)

`configs/bimamba_overlap0.yaml`에서 batch size 조정:

```yaml
stage_configs:
  resampler:
    batch_size: 1  # 기본값
    accumulate_grad_batches: 8  # 4 → 8로 증가
  
  finetune:
    batch_size: 1
    accumulate_grad_batches: 8
```

### Stage State 충돌

기존 stage state 파일이 있으면 삭제:

```bash
rm runs/*_stage_state.json
```

## 비교 실험

Overlap 효과를 비교하려면:

1. **Overlap=0.5 (기존)**
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```

2. **Overlap=0.0 (새로운)**
   ```bash
   bash scripts/train_bimamba_overlap0.sh
   ```

3. **평가 비교**
   ```bash
   # Overlap=0.5 평가
   python scripts/eval.py \
       --config runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/config.yaml

   # Overlap=0.0 평가 (위 학습 완료 후)
   python scripts/eval.py \
       --config runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/config.yaml
   ```

## FAQ

**Q: Vision 체크포인트는 왜 재사용하나요?**  
A: Vision stage는 VICReg loss로 학습되며, overlap과 무관하게 개별 뷰의 feature를 학습합니다. 따라서 재사용 가능합니다.

**Q: BiMamba 대신 다른 리샘플러를 쓸 수 있나요?**  
A: 네, `configs/bimamba_overlap0.yaml`에서 `resampler_type`을 `mlp`, `qformer`, `perceiver` 등으로 변경하면 됩니다.

**Q: Overlap을 다른 값으로 설정할 수 있나요?**  
A: 네, `overlap_ratio`를 0.0~0.5 사이 값으로 변경 가능합니다 (예: 0.25).

**Q: 학습 시간은 얼마나 걸리나요?**  
A: Overlap=0일 때 타일 수가 줄어들어 약 40-50% 빨라집니다.
