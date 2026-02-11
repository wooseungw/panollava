# VLM Ablation Study 실행 가이드

## 📋 개요

`scripts/run_vlm_ablation.sh`는 여러 VLM 모델을 동일한 데이터셋으로 평가하여 성능을 비교하는 자동화 스크립트입니다.

---

## 🚀 빠른 시작

### 1. 기본 실행 (전체 모델)

```bash
bash scripts/run_vlm_ablation.sh
```

**평가 모델** (8개):
1. blip2-opt-2.7b (2.7B)
2. internvl2-2b (2B)
3. blip2-flan-t5-xl (3B)
4. qwen2.5-vl-3b (3B) ← 새로운
5. gemma-3-4b (4B) ← 새로운
6. llava-1.5-7b (7B)
7. llava-1.6-mistral-7b (7B)
8. instructblip-vicuna-7b (7B)

**예상 소요 시간**: 2-3시간

---

### 2. 빠른 테스트 (경량 모델만)

스크립트 수정:
```bash
# scripts/run_vlm_ablation.sh 파일에서

# 전체 모델 (주석 처리)
# MODELS=(
#     "blip2-opt-2.7b"
#     ...
# )

# 빠른 테스트용 (주석 해제)
MODELS=(
    "blip2-opt-2.7b"
    "internvl2-2b"
    "qwen2.5-vl-3b"
    "gemma-3-4b"
)
```

**예상 소요 시간**: 30-40분

---

### 3. 최신 모델만 테스트

```bash
# 스크립트에서 다음만 활성화
MODELS=(
    "qwen2.5-vl-3b"
    "gemma-3-4b"
)
```

**예상 소요 시간**: 15-20분

---

## 📊 출력 예시

### 시작 시

```
============================================================
VLM Ablation Study 시작
============================================================
총 모델 수: 8
데이터: data/quic360/test.csv
출력 디렉토리: results
배치 크기: 2
============================================================

평가할 모델:
  1. blip2-opt-2.7b
  2. internvl2-2b
  3. blip2-flan-t5-xl
  4. qwen2.5-vl-3b
  5. gemma-3-4b
  6. llava-1.5-7b
  7. llava-1.6-mistral-7b
  8. instructblip-vicuna-7b
```

### 각 모델 평가 중

```
============================================================
[1/8] 평가 시작: blip2-opt-2.7b
============================================================
시작 시간: 2025-01-15 10:30:00

🧹 텍스트 정리 중 (특수 토큰/역할 태그 제거)...
✓ BLEU-4 (sacrebleu): 0.2345 (원점수: 23.45/100)
  → 토큰화: 13a (Moses), 스무딩: exp, 대소문자: 보존
✓ METEOR: 0.3210
✓ ROUGE-L: 0.4567
✓ SPICE: 0.1890
✓ CIDEr: 0.8765

✓ [1/8] blip2-opt-2.7b 평가 완료
종료 시간: 2025-01-15 10:45:00
GPU 메모리 정리 중...
```

### 완료 시

```
============================================================
VLM Ablation Study 완료
============================================================
총 소요 시간: 145분 32초
완료된 모델: 8/8
실패한 모델: 0/8
결과 위치: results/
============================================================

생성된 결과 파일:
results/blip2-opt-2.7b_metrics.json
results/blip2-opt-2.7b_predictions.csv
results/internvl2-2b_metrics.json
results/internvl2-2b_predictions.csv
...
```

---

## 📁 출력 파일 구조

```
results/
├── blip2-opt-2.7b_metrics.json          # 평가 메트릭
├── blip2-opt-2.7b_predictions.csv       # 예측/정답 비교
├── internvl2-2b_metrics.json
├── internvl2-2b_predictions.csv
├── qwen2.5-vl-3b_metrics.json           # 새로운 모델
├── qwen2.5-vl-3b_predictions.csv
├── gemma-3-4b_metrics.json              # 새로운 모델
├── gemma-3-4b_predictions.csv
├── llava-1.5-7b_metrics.json
├── llava-1.5-7b_predictions.csv
...
└── all_models_summary.json              # 전체 요약
```

### 메트릭 JSON 예시

```json
{
  "model_name": "qwen2.5-vl-3b",
  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
  "num_samples": 500,
  "metrics": {
    "bleu4": 0.2845,
    "meteor": 0.3512,
    "rougeL": 0.4892,
    "spice": 0.2134,
    "cider": 1.0234
  }
}
```

---

## ⚙️ 설정 커스터마이징

### 스크립트 내 설정 변경

```bash
# scripts/run_vlm_ablation.sh 파일 수정

# GPU 선택
export CUDA_VISIBLE_DEVICES=0  # GPU 0 사용
# export CUDA_VISIBLE_DEVICES=1,2  # GPU 1,2 사용

# 데이터셋 경로
DATA_CSV="data/quic360/test.csv"
# DATA_CSV="data/custom/my_test.csv"

# 출력 디렉토리
OUTPUT_DIR="results"
# OUTPUT_DIR="results/ablation_$(date +%Y%m%d)"

# 배치 크기 (GPU 메모리에 따라 조정)
BATCH_SIZE=2  # 기본값
# BATCH_SIZE=1  # 메모리 부족 시
# BATCH_SIZE=4  # 메모리 여유 시

# 생성 토큰 수
MAX_NEW_TOKENS=64  # 기본값
# MAX_NEW_TOKENS=128  # 더 긴 응답
```

---

## 🔍 모델별 특징

| 모델 | 크기 | 특징 | 예상 시간 | 메모리 |
|------|------|------|-----------|--------|
| blip2-opt-2.7b | 2.7B | 가장 경량, 빠름 | 15분 | 8GB |
| internvl2-2b | 2B | 경량, 높은 성능 | 15분 | 8GB |
| blip2-flan-t5-xl | 3B | Flan-T5 기반 | 18분 | 10GB |
| **qwen2.5-vl-3b** | 3B | 최신, chat template | 20분 | 12GB |
| **gemma-3-4b** | 4B | Google 최신 | 22분 | 14GB |
| llava-1.5-7b | 7B | 널리 사용됨 | 25분 | 18GB |
| llava-1.6-mistral-7b | 7B | LLaVA 최신 | 25분 | 18GB |
| instructblip-vicuna-7b | 7B | Instruction 튜닝 | 28분 | 20GB |

---

## 🐛 문제 해결

### GPU 메모리 부족

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결**:
```bash
# 1. 배치 크기 줄이기
BATCH_SIZE=1

# 2. 작은 모델만 평가
MODELS=(
    "blip2-opt-2.7b"
    "internvl2-2b"
)

# 3. GPU 0번만 사용
export CUDA_VISIBLE_DEVICES=0
```

---

### 특정 모델 실패

**증상**:
```
✗ [4/8] qwen2.5-vl-3b 평가 실패
```

**확인 사항**:

1. **qwen-vl-utils 설치**:
   ```bash
   pip install qwen-vl-utils
   ```

2. **transformers 버전**:
   ```bash
   pip install --upgrade transformers
   # 최소 버전: 4.40.0
   ```

3. **로그 확인**:
   - 스크립트 출력에서 에러 메시지 확인
   - CUDA 메모리, 임포트 에러 등

---

### 데이터 파일 없음

**증상**:
```
FileNotFoundError: data/quic360/test.csv
```

**해결**:
```bash
# 데이터 경로 확인
ls -l data/quic360/test.csv

# 또는 다른 데이터셋 사용
DATA_CSV="data/your_dataset/test.csv"
```

---

## 📈 결과 분석

### Python으로 결과 비교

```python
import json
import pandas as pd
from pathlib import Path

# 모든 메트릭 로드
results = {}
result_dir = Path("results")

for metrics_file in result_dir.glob("*_metrics.json"):
    with open(metrics_file) as f:
        data = json.load(f)
        model_name = data['model_name']
        results[model_name] = data['metrics']

# DataFrame으로 변환
df = pd.DataFrame(results).T
df = df.sort_values('bleu4', ascending=False)

print("모델 성능 순위 (BLEU-4 기준):")
print(df)

# CSV로 저장
df.to_csv("results/comparison.csv")
```

### 터미널에서 빠른 확인

```bash
# BLEU-4 점수만 추출
for f in results/*_metrics.json; do
    echo -n "$(basename $f _metrics.json): "
    jq '.metrics.bleu4' $f
done | sort -t: -k2 -rn

# 출력 예시:
# qwen2.5-vl-3b: 0.2845
# gemma-3-4b: 0.2756
# llava-1.5-7b: 0.2634
# ...
```

---

## 🔄 변경 이력

### 최신 (2025-01-XX)

**추가된 모델**:
- ✅ gemma-3-4b (Google 최신 VLM)
- ✅ qwen2.5-vl-3b (Qwen 최신 버전)

**제거된 모델**:
- ❌ qwen-vl-chat (업그레이드)
- ❌ qwen2-vl-2b (업그레이드)
- ❌ cogvlm2-llama3-chat-19b (메모리 부담)

**개선 사항**:
- ✅ sacrebleu 자동 적용 (표준 BLEU)
- ✅ basic_cleanup 자동 적용 (특수 토큰 제거)
- ✅ 진행 상황 추적 개선
- ✅ 에러 처리 개선
- ✅ 소요 시간 측정

---

## 📚 관련 문서

- [evaluate_vlm_models.py 가이드](./VLM_EVALUATION_GUIDE.md)
- [sacrebleu 업데이트](./SACREBLEU_UPDATE.md)
- [평가 스크립트 요약](./EVALUATION_SCRIPTS_SUMMARY.md)
- [모델 업데이트](./VLM_MODEL_UPDATES.md)

---

## 💡 팁

### 1. 백그라운드 실행

장시간 실행되므로 백그라운드에서 실행:

```bash
nohup bash scripts/run_vlm_ablation.sh > ablation.log 2>&1 &

# 진행 상황 확인
tail -f ablation.log
```

### 2. 특정 모델만 재평가

```bash
# 스크립트 수정
MODELS=(
    "gemma-3-4b"  # 이 모델만 재평가
)
```

### 3. 결과 백업

```bash
# 결과 백업
cp -r results results_backup_$(date +%Y%m%d)

# 또는 자동 백업
OUTPUT_DIR="results/ablation_$(date +%Y%m%d_%H%M%S)"
```

---

## ✅ 체크리스트

실행 전 확인:

- [ ] GPU 사용 가능: `nvidia-smi`
- [ ] Conda 환경 활성화: `conda activate pano`
- [ ] 데이터 파일 존재: `ls data/quic360/test.csv`
- [ ] sacrebleu 설치: `pip list | grep sacrebleu`
- [ ] qwen-vl-utils 설치 (Qwen2.5-VL 사용 시)
- [ ] 충분한 디스크 공간 (최소 10GB)

실행 후 확인:

- [ ] 모든 모델 완료: "완료된 모델: 8/8"
- [ ] 결과 파일 생성: `ls results/*_metrics.json`
- [ ] BLEU 점수 합리적 (0.1~0.5 범위)

