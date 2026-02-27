# Setup Guide — Models & Datasets

## 1. 환경 설정

```bash
# 1) conda 환경 생성
conda create -n pano python=3.12 -y
conda activate pano

# 2) 패키지 설치 (editable mode)
pip install -e ".[dev]"

# 3) 선택: BiMamba resampler 사용 시
source fix_mamba_cuda.sh

# 4) 선택: 평가 메트릭 (BLEU, CIDEr, SPICE 등)
bash install_eval_metrics.sh
```

---

## 2. 모델 다운로드

모든 모델은 HuggingFace Hub에서 자동 다운로드됩니다.  
`from_pretrained()` 호출 시 자동으로 받아지지만, 미리 받아두려면 아래 커맨드를 사용하세요.

### 필수 설치

```bash
pip install huggingface_hub
huggingface-cli login   # HF_TOKEN 필요 (gemma3는 gate 모델)
```

### 사용 모델 목록

| 모델 | HuggingFace ID | 크기 | 용도 |
|------|----------------|------|------|
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | ~6GB | PanoAdapt B1 (주실험) |
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | ~14GB | 대형 모델 실험 |
| Qwen2-VL-2B | `Qwen/Qwen2-VL-2B-Instruct` | ~4GB | 경량 실험 |
| InternVL3-2B | `OpenGVLab/InternVL3-2B-hf` | ~4GB | PanoAdapt B2 (주실험) |
| InternVL3-1B | `OpenGVLab/InternVL3-1B-hf` | ~2GB | 경량 실험 |
| InternVL2.5-2B | `OpenGVLab/InternVL2_5-2B` | ~4GB | |
| InternVL2.5-4B | `OpenGVLab/InternVL2_5-4B` | ~8GB | |
| Gemma3-4B | `google/gemma-3-4b-it` | ~8GB | PanoAdapt B3 (gate 모델) |
| BLIP2-2.7B | `Salesforce/blip2-opt-2.7b` | ~6GB | baseline |

### 미리 받아두기 (선택)

```bash
# 주요 3개 모델 (약 18GB)
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
huggingface-cli download OpenGVLab/InternVL3-2B-hf
huggingface-cli download google/gemma-3-4b-it   # gate: HF 사이트에서 접근 신청 필요

# 전체 다운로드 (약 60GB)
for MODEL in \
    "Qwen/Qwen2.5-VL-3B-Instruct" \
    "Qwen/Qwen2.5-VL-7B-Instruct" \
    "Qwen/Qwen2-VL-2B-Instruct" \
    "OpenGVLab/InternVL3-2B-hf" \
    "OpenGVLab/InternVL3-1B-hf" \
    "OpenGVLab/InternVL2_5-2B" \
    "OpenGVLab/InternVL2_5-4B" \
    "Salesforce/blip2-opt-2.7b"; do
    echo "Downloading $MODEL..."
    huggingface-cli download "$MODEL"
done

# Gemma3는 별도 (gate 모델)
huggingface-cli download google/gemma-3-4b-it
```

> **Gemma3 접근 신청**: https://huggingface.co/google/gemma-3-4b-it 에서 "Access repository" 클릭 후 `huggingface-cli login` 필요.

모델은 기본적으로 `~/.cache/huggingface/hub/`에 저장됩니다.  
경로 변경: `export HF_HOME=/your/storage/path`

---

## 3. 데이터셋

### 3-1. QUIC-360 (주요 학습/평가 데이터)

PanoAdapt baseline 실험에 사용되는 파노라마 VQA 데이터셋입니다.

#### CSV 포맷

```
url,instruction,response
/path/to/image.jpg,What do you see?,"A wide panoramic scene..."
```

| 컬럼 | 설명 |
|------|------|
| `url` | 이미지 로컬 경로 또는 Flickr URL |
| `instruction` | 질문 텍스트 |
| `response` | 정답 텍스트 |

#### 다운로드 방법 (Flickr 이미지)

QUIC-360 이미지는 Flickr에서 다운로드해야 합니다.

```bash
# 1) refer360 데이터셋의 원본 CSV 준비
#    (Refer360 저자에게 요청 또는 refer360 GitHub 참고)
#    → CSV에 Flickr URL이 포함되어 있음

# 2) 이미지 다운로드 (16 threads 병렬)
python scripts/download_quic360_images.py

# 다운로드 경로 수정이 필요하면 스크립트 상단의 SAVE_DIR 변경
# 기본값: /data/1_personal/4_SWWOO/refer360/data/quic360_format/images
```

> Refer360 데이터셋: https://github.com/volkancirik/refer360

#### 로컬 CSV 경로 설정

이미지를 다른 경로에 받았다면, CSV의 경로를 일괄 수정:

```python
import pandas as pd

NEW_IMG_DIR = "/your/path/to/quic360/images"

for split in ["train", "test"]:
    df = pd.read_csv(f"data/{split}.csv")
    # Flickr URL → 로컬 경로로 변환
    df["url"] = df["url"].apply(
        lambda u: f"{NEW_IMG_DIR}/{u.split('/')[-1]}" if u.startswith("http") else u
    )
    df.to_csv(f"data/{split}.csv", index=False)
```

그 다음 config YAML에서 경로 지정:

```yaml
data_train_csv: "/your/path/to/train.csv"
data_test_csv: "/your/path/to/test.csv"
```

#### 통계

| Split | 샘플 수 |
|-------|---------|
| Train | 7,929 |
| Test  | 5,349 |

---

### 3-2. Stage 1 데이터셋 (VICReg 학습용, 선택)

CORA 3-stage 학습의 Vision Stage에 사용하는 이미지-only 데이터셋입니다.  
PanoAdapt baseline만 사용한다면 **불필요**합니다.

| 데이터셋 | 출처 | 용도 |
|---------|------|------|
| Stanford2D3D | [Stanford](http://buildingparser.stanford.edu/dataset.html) | 실내 파노라마 |
| ZInD | [Zillow](https://github.com/zillow/zind) | 실내 파노라마 |
| SUN360 | [MIT CSAIL](http://sun360.mit.edu/) | 실외/실내 파노라마 |

```bash
# 다운로드 후 CSV 생성
python scripts/build_stage1_csv.py
# → data/stage1_train.csv, data/stage1_val.csv 생성
```

스크립트 상단의 `DATASETS` 딕셔너리에서 각 데이터셋 경로를 설정하세요.

---

## 4. 빠른 시작

### PanoAdapt baseline 실험 (권장)

```bash
conda activate pano

# 스모크 테스트 (GPU 1개, ~5분)
python -c "
from cora.baseline.finetune import BaselineTrainer
from cora.baseline.config import BaselineConfig
import yaml
with open('configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml') as f:
    cfg = BaselineConfig(**yaml.safe_load(f))
cfg.training.num_epochs = 0.001
BaselineTrainer(cfg).train()
print('OK')
"

# 본 학습
python scripts/baseline_finetune.py \
    --config configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml

# 평가
python scripts/baseline_eval.py \
    --config configs/baseline/panoadapt_pe_densecl_qwen25_3b.yaml \
    --output-dir runs/baseline/panoadapt_qwen25-vl-3b/eval
```

### GPU 지정

```bash
export CUDA_VISIBLE_DEVICES=0   # GPU 0 사용
export CUDA_VISIBLE_DEVICES=1   # GPU 1 사용
export CUDA_VISIBLE_DEVICES=0,1 # 멀티 GPU
```

### 실험 큐 스크립트

```bash
# Qwen2.5 실험 (GPU 0)
bash scripts/run_qwen25_retry.sh

# VICReg-pairwise InternVL (GPU 1 Phase 2)
bash scripts/run_gpu1_phase2.sh
```

---

## 5. 디렉토리 구조 (필수)

```
panollava/
├── src/cora/              # 패키지 (pip install -e . 으로 설치)
├── scripts/               # 실행 스크립트
├── configs/baseline/      # 실험별 YAML config
├── data/                  # Stage 1 CSV (로컬)
└── runs/baseline/
    └── _shared_data/
        ├── train.csv      # QUIC-360 train (경로 설정 필요)
        └── test.csv       # QUIC-360 test (경로 설정 필요)
```

`runs/baseline/_shared_data/`가 없으면 생성:

```bash
mkdir -p runs/baseline/_shared_data
cp /your/path/to/quic360/train.csv runs/baseline/_shared_data/train.csv
cp /your/path/to/quic360/test.csv  runs/baseline/_shared_data/test.csv
```

---

## 6. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError: cora` | 패키지 미설치 | `pip install -e .` |
| `CUDA out of memory` | 배치 크기 과대 | `batch_size: 1` + `gradient_accumulation_steps: 4` |
| Gemma3 `401 Unauthorized` | HF gate 모델 | HF 사이트에서 접근 신청 후 `huggingface-cli login` |
| `mamba_ssm` ImportError | Mamba 미설치 | `source fix_mamba_cuda.sh` |
| SPICE 평가 실패 | Java 미설치 | `sudo apt install default-jdk` |
| 이미지 로딩 실패 (`_MAX_RETRIES`) | 잘못된 이미지 경로 | CSV의 `url` 컬럼 경로 확인 |
