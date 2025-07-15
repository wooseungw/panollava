
# Panorama-VLM 학습 및 평가 사용법

## 1. 학습 (train.py)

### 단일 스테이지 학습
```bash
python train.py --stage vision --csv-train <train.csv> --csv-val <val.csv>
```

### 다단계 연속 학습 (체크포인트 자동 인계)
```bash
python train.py --stages vision resampler finetune --csv-train <train.csv> --csv-val <val.csv>
```
- 각 스테이지별로 최적 체크포인트가 저장되며, 다음 스테이지에서 자동으로 이어받아 학습합니다.
- 중간에 중단된 경우, `--resume-from <checkpoint>` 옵션으로 이어서 재시작할 수 있습니다.

#### 주요 옵션
- `--csv-train`, `--csv-val`: 학습/검증 데이터 CSV 경로
- `--vision-name`: 비전 모델 이름 (예: google/siglip-base-patch16-224)
- `--lm-name`: 언어 모델 이름 (예: Qwen/Qwen3-0.6B)
- `--resampler`: 리샘플러 타입 (예: mlp)
- `--batch-size`, `--num-workers`, `--max-txt-len`: 데이터로더 및 토크나이저 설정
- `--wandb-project`, `--wandb-name`: W&B 로깅 설정
- `--resume-from`: 중단된 체크포인트에서 이어서 학습


#### 예시
```bash
# 1. vision만 단일 학습
python train.py --stage vision --csv-train data/quic360/downtest.csv --csv-val data/quic360/downtest.csv

# 2. vision → resampler → finetune 3스테이지 전체 순차 학습
python train.py --stages vision resampler finetune --csv-train data/quic360/downtest.csv --csv-val data/quic360/downtest.csv

# 3. 중간 체크포인트에서 이어서 나머지 스테이지 학습 (예: resampler, finetune만)
python train.py --stages resampler finetune --resume-from runs/vlm_vision/checkpoints/epoch=02-val_loss=0.123.ckpt
```

> ⚠️ **스테이지가 바뀌는 경우 --resume-from 사용 시 주의:**
> - vision → resampler 등 스테이지가 바뀌면 optimizer 파라미터 그룹이 달라져서 PyTorch Lightning의 checkpoint resume 기능이 오류를 낼 수 있습니다.
> - 본 코드는 스테이지가 바뀌는 경우 optimizer state는 무시하고, 모델 가중치만 warm-start(이어받기) 하도록 자동 처리합니다.
> - 즉, 스테이지가 다르더라도 --resume-from으로 이어서 학습이 가능합니다.

---

## 2. 평가 (eval.py)

### 체크포인트 기반 평가
```bash
python eval.py --ckpt <checkpoint.ckpt> --csv-val <val.csv>
```

#### 주요 옵션
- `--ckpt`: 평가할 체크포인트(.ckpt) 경로
- `--csv-val`: 평가용 CSV 파일
- `--vision-name`, `--lm-name`, `--resampler`: 모델 구성 옵션 (학습과 동일하게 지정)
- `--batch-size`, `--max-txt-len`, `--max-new-tokens`: 평가 배치 및 생성 길이 설정

#### 예시
```bash
# 저장된 체크포인트로 평가 및 예측 결과 저장
python eval.py --ckpt runs/vlm_finetune/checkpoints/epoch=00-val_loss=0.123.ckpt --csv-val data/quic360/downtest.csv
```

- 평가 결과는 `eval_outputs.jsonl`로 저장되며, BLEU/METEOR/ROUGE 등 주요 텍스트 메트릭이 자동 계산됩니다.

---
자세한 옵션은 `python train.py --help` 또는 `python eval.py --help` 참고.
