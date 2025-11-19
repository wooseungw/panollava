# Config.yaml 기반 자동 체크포인트 로드 가이드

## 개요
`eval.py`가 저장된 `config.yaml` 파일 경로를 입력받으면 자동으로 `best.ckpt`를 탐색하고 로드하는 기능이 추가되었습니다.

## 🎯 핵심 기능

### 자동 감지 메커니즘
```python
# config.yaml 위치: runs/.../finetune/anyres-e2p_bimamba/config.yaml
# 자동으로 감지:
# 1. config.yaml이 있는 디렉토리 = 체크포인트 디렉토리
# 2. best.ckpt 또는 last.ckpt 자동 탐색
# 3. checkpoint_metadata.json에서 모델 설정 로드
```

### 체크포인트 탐색 우선순위
1. **best.ckpt** (심볼릭 링크 또는 파일)
2. **last.ckpt** (심볼릭 링크 또는 파일)
3. **가장 최근 .ckpt 파일** (수정 시간 기준)

---

## 📝 사용법

### ✨ 방법 1: 저장된 config.yaml 직접 지정 (권장!)

```bash
# 가장 간단한 방법
python scripts/eval.py \
    --config runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/config.yaml \
    --csv-input data/quic360/test.csv

# 실행 과정:
# 1. config.yaml에서 체크포인트 디렉토리 자동 감지
# 2. best.ckpt 자동 탐색
# 3. checkpoint_metadata.json 로드
# 4. 평가 시작
```

**출력 예시**:
```
============================================================
🔍 config.yaml에서 자동 감지된 체크포인트 디렉토리:
   runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba
============================================================
📂 체크포인트 디렉토리: runs/.../finetune/anyres-e2p_bimamba
✅ Using best checkpoint: runs/.../siglip2_bimamba_anyres-e2p_quic360_epoch03_loss0.2341.ckpt
✅ 메타데이터 로드 성공: runs/.../checkpoint_metadata.json
```

### 방법 2: 체크포인트 디렉토리 직접 지정

```bash
python scripts/eval.py \
    --checkpoint-dir runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/ \
    --csv-input data/quic360/test.csv
```

### 방법 3: 글로벌 config + 체크포인트 디렉토리

```bash
python scripts/eval.py \
    --config configs/default.yaml \
    --checkpoint-dir runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/
```

---

## 🔍 디렉토리 구조 이해

### 학습 후 생성되는 구조
```
runs/
└── siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/   ← experiment_name
    └── finetune/                                  ← stage
        └── anyres-e2p_bimamba/                    ← {crop_strategy}_{resampler}
            ├── config.yaml                        ← 🎯 이 파일을 --config로 지정!
            ├── checkpoint_metadata.json           ← 모델 설정 자동 로드
            ├── best.ckpt → siglip2_bimamba_...    ← 자동 탐색됨
            ├── last.ckpt → siglip2_bimamba_...
            ├── siglip2_bimamba_anyres-e2p_quic360_epoch01_loss0.3456.ckpt
            ├── siglip2_bimamba_anyres-e2p_quic360_epoch02_loss0.2789.ckpt
            └── siglip2_bimamba_anyres-e2p_quic360_epoch03_loss0.2341.ckpt
```

### config.yaml 경로 찾기
```bash
# 모든 저장된 config.yaml 찾기
find runs/ -name "config.yaml"

# 출력 예시:
# runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/vision/anyres-e2p_bimamba/config.yaml
# runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/resampler/anyres-e2p_bimamba/config.yaml
# runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/config.yaml
```

---

## ⚙️ 자동 로드되는 설정들

### checkpoint_metadata.json에서 자동 병합
```json
{
  "stage": "finetune",
  "experiment_name": "siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE",
  "model_config": {
    "vision_name": "google/siglip2-so400m-patch16-256",
    "language_model_name": "Qwen/Qwen3-0.6B",
    "resampler_type": "bimamba",
    "latent_dimension": 768,
    "image_size": [256, 256]
  },
  "training_config": {
    "crop_strategy": "anyres_e2p",
    "fov_deg": 90.0,
    "overlap_ratio": 0.5,
    "use_vision_processor": true
  }
}
```

**자동 병합 우선순위**:
1. `checkpoint_metadata.json` (최우선)
2. 저장된 `config.yaml`
3. 글로벌 `default.yaml` (fallback)

---

## 🎬 전체 워크플로우

### 학습 → 평가 전체 과정

```bash
# 1. 학습 실행
python scripts/train.py --config configs/default.yaml

# 생성된 경로 확인
# runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/

# 2. 평가 실행 (config.yaml 경로만 지정)
python scripts/eval.py \
    --config runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/config.yaml \
    --csv-input data/quic360/test.csv \
    --log-samples

# 끝! 모든 설정이 자동으로 로드됨
```

---

## 🚨 문제 해결

### 1. "체크포인트 파일을 찾을 수 없습니다" 에러

**원인**: config.yaml이 체크포인트 디렉토리에 없음

**해결**:
```bash
# config.yaml 위치 확인
ls -la runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/

# .ckpt 파일 확인
ls -la runs/.../finetune/anyres-e2p_bimamba/*.ckpt

# 심볼릭 링크 확인 (Linux/Mac)
ls -la runs/.../finetune/anyres-e2p_bimamba/*.ckpt | grep "^l"
```

### 2. "메타데이터 파일을 찾을 수 없습니다" 경고

**영향**: 경고만 출력, 평가는 진행됨 (config.yaml 사용)

**해결**: `checkpoint_metadata.json`이 누락되었을 수 있음 (학습 시 생성됨)

### 3. Windows에서 심볼릭 링크 작동 안 함

**해결**: best.ckpt/last.ckpt 대신 실제 파일명 확인 후 사용
```bash
# PowerShell
Get-ChildItem runs\...\finetune\anyres-e2p_bimamba\*.ckpt | Sort-Object LastWriteTime

# 가장 최근 파일 사용
```

---

## 📊 비교: 기존 vs 개선

| 항목 | 기존 방법 | 개선된 방법 (✨) |
|------|-----------|-----------------|
| **config 지정** | 글로벌 config만 가능 | 저장된 config.yaml 직접 사용 |
| **체크포인트** | 수동으로 경로 지정 | 자동 탐색 (best.ckpt) |
| **명령어 길이** | 긴 경로 2개 필요 | config.yaml 경로 1개만 |
| **재현성** | 설정 불일치 가능 | 완벽한 재현성 보장 |

### 명령어 비교

**기존**:
```bash
python scripts/eval.py \
    --config configs/default.yaml \
    --checkpoint-dir runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/ \
    --csv-input data/quic360/test.csv
```

**개선 (✨)**:
```bash
python scripts/eval.py \
    --config runs/siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE/finetune/anyres-e2p_bimamba/config.yaml \
    --csv-input data/quic360/test.csv
```

**장점**:
- 명령어 간결화 (40% 단축)
- 설정 자동 일치 (학습 시 사용한 정확한 설정)
- 실수 방지 (잘못된 config 사용 불가)

---

## 🔗 관련 문서
- **CHECKPOINT_EVAL_GUIDE.md**: 체크포인트 평가 상세 가이드
- **EVAL_QUICKSTART.md**: 평가 시스템 빠른 시작
- **CHECKPOINT_METADATA.md**: 메타데이터 구조 설명

---

## 💡 팁

### 여러 스테이지 비교 평가
```bash
# Vision 스테이지 평가
python scripts/eval.py \
    --config runs/exp1/vision/anyres-e2p_bimamba/config.yaml \
    --csv-input data/quic360/test.csv

# Resampler 스테이지 평가
python scripts/eval.py \
    --config runs/exp1/resampler/anyres-e2p_bimamba/config.yaml \
    --csv-input data/quic360/test.csv

# Finetune 스테이지 평가
python scripts/eval.py \
    --config runs/exp1/finetune/anyres-e2p_bimamba/config.yaml \
    --csv-input data/quic360/test.csv
```

### Bash 스크립트로 자동화
```bash
#!/bin/bash
# eval_all_stages.sh

EXPERIMENT="siglip2-so400m_Qwen3_bimamba_anyres-e2p_PE"
CSV_INPUT="data/quic360/test.csv"

for STAGE in vision resampler finetune; do
    echo "Evaluating $STAGE stage..."
    python scripts/eval.py \
        --config runs/$EXPERIMENT/$STAGE/anyres-e2p_bimamba/config.yaml \
        --csv-input $CSV_INPUT \
        --log-samples
done
```

### 실행 권한 부여 및 실행
```bash
chmod +x eval_all_stages.sh
./eval_all_stages.sh
```

---

## ✅ 체크리스트

평가 실행 전 확인사항:
- [ ] config.yaml 파일 존재 확인
- [ ] 해당 디렉토리에 .ckpt 파일 존재 확인
- [ ] checkpoint_metadata.json 존재 확인 (선택사항)
- [ ] CSV 입력 파일 경로 정확한지 확인
- [ ] GPU 메모리 충분한지 확인

평가 성공 확인:
- [ ] "✅ Using best checkpoint" 로그 확인
- [ ] "✅ 메타데이터 로드 성공" 로그 확인
- [ ] 평가 메트릭 정상 출력 확인
- [ ] 결과 JSON 파일 생성 확인
