# 체크포인트 디렉토리 기반 평가 - 빠른 시작

## ✨ 새로운 기능: `--checkpoint-dir` 지원

이제 복잡한 config 파일 없이 **체크포인트 디렉토리만으로** 모델을 평가할 수 있습니다!

## 가장 간단한 방법

```bash
# 1. 체크포인트 디렉토리만 지정
python scripts/eval.py --checkpoint-dir runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/

# 2. 평가용 CSV 추가 지정
python scripts/eval.py \
    --checkpoint-dir runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/ \
    --csv-input data/quic360/test.csv
```

## 무엇이 자동화되나요?

### ✅ 자동으로 처리되는 것들

1. **메타데이터 자동 로드** (`checkpoint_metadata.json`)
   - 모델 아키텍처 (vision encoder, language model, resampler)
   - 이미지 처리 설정 (crop_strategy, image_size, fov_deg)
   - 하이퍼파라미터 (lr, batch_size 등)

2. **스마트 체크포인트 선택**
   - `best.ckpt` → `last.ckpt` → 최신 `.ckpt` 순으로 자동 선택

3. **LoRA 가중치 자동 탐색**
   - `lora_weights/` 디렉토리 자동 인식 및 로드

## 실제 사용 예시

### 기본 평가
```bash
python scripts/eval.py \
    --checkpoint-dir runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/
```

### 빠른 테스트 (100개 샘플만)
```bash
python scripts/eval.py \
    --checkpoint-dir runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/ \
    --max-samples 100
```

### 상세 로깅 활성화
```bash
python scripts/eval.py \
    --checkpoint-dir runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/ \
    --log-samples \
    --log-interval 10
```

## 로그 출력 예시

```
============================================================
📂 체크포인트 디렉토리: runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp
============================================================
✅ 메타데이터 로드 성공: .../checkpoint_metadata.json
✅ Using best checkpoint: .../siglip_mlp_anyres-e2p_quic360_epoch05_loss0.4201.ckpt
============================================================
📋 메타데이터에서 로드된 정보:
  - Experiment: ADDDATA_SQ3_1
  - Stage: finetune
  - Vision: google/siglip2-so400m-patch14-224
  - Language: Qwen/Qwen3-0.6B
  - Resampler: mlp
  - Crop Strategy: anyres_e2p
============================================================
✅ 메타데이터를 config에 병합 완료
✅ Auto-found LoRA weights: .../lora_weights
```

## 필요한 디렉토리 구조

```
runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/
├── checkpoint_metadata.json          # 필수: 훈련 설정 정보
├── best.ckpt -> siglip_mlp_...ckpt  # 권장: 최고 성능
├── last.ckpt -> siglip_mlp_...ckpt  # 권장: 마지막 epoch
├── siglip_mlp_anyres-e2p_quic360_epoch05_loss0.4201.ckpt
└── lora_weights/                     # 선택: LoRA 사용 시
```

## 기존 방식과 비교

### 이전 (복잡함 😓)
```bash
# config 파일 준비 필요
# 체크포인트 경로 수동 설정 필요
# 모델 설정 수동 확인 필요
python scripts/eval.py --config configs/my_config.yaml
```

### 지금 (간단함 ✨)
```bash
# 체크포인트 디렉토리만 지정
# 모든 설정 자동 로드
python scripts/eval.py --checkpoint-dir runs/my_experiment/finetune/anyres-e2p_mlp/
```

## 상세 가이드

더 자세한 내용은 [CHECKPOINT_EVAL_GUIDE.md](CHECKPOINT_EVAL_GUIDE.md)를 참조하세요.

## 문제 해결

### 메타데이터가 없는 경우
```
⚠️ 메타데이터 파일을 찾을 수 없습니다
```
→ `--config` 옵션으로 설정 파일 추가 또는 최신 체크포인트 사용

### 체크포인트를 찾을 수 없는 경우
```
FileNotFoundError: 체크포인트 파일을 찾을 수 없습니다
```
→ 디렉토리에 `.ckpt` 파일이 있는지 확인

## 관련 문서
- [CHECKPOINT_EVAL_GUIDE.md](CHECKPOINT_EVAL_GUIDE.md) - 전체 가이드
- [CHECKPOINT_METADATA.md](CHECKPOINT_METADATA.md) - 메타데이터 시스템
- [VLM_EVALUATION_GUIDE.md](VLM_EVALUATION_GUIDE.md) - 평가 메트릭
