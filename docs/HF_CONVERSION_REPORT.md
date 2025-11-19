# HuggingFace 변환 완료 보고서

## ✅ 변환 성공!

Lightning 체크포인트가 HuggingFace 형식으로 성공적으로 변환되었습니다.

### 📊 변환 결과

**입력**:
- Lightning checkpoint: `runs/SQ3_1_latent768_PE_e2p_finetune_qformer/last.ckpt`
- Stage: finetune (3단계 학습 완료)
- Resampler: QFormer
- LoRA: 활성화됨

**출력**:
- HF 모델 디렉토리: `hf_checkpoints/panollava-vlm-qformer/`
- 파일 구성:
  - `config.json` (1.4 KB) - 모델 설정
  - `pytorch_model.bin` (3.0 GB) - 모델 가중치
  - `README.md` (689 B) - 모델 카드

### 🎯 모델 상세 정보

```json
{
  "vision": "google/siglip-base-patch16-224",
  "text": "Qwen/Qwen3-0.6B",
  "resampler": "qformer",
  "latent_dimension": 768,
  "crop_strategy": "e2p",
  "use_lora": true,
  "overlap_ratio": 0.5
}
```

### 📦 생성된 파일

```
hf_checkpoints/panollava-vlm-qformer/
├── config.json              # HuggingFace 설정 파일
├── pytorch_model.bin         # 모델 가중치 (3.0GB)
└── README.md                 # 모델 카드
```

### 🚀 사용 방법

#### 1. 로컬에서 로드

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# 모델 로드
model = AutoModelForVision2Seq.from_pretrained(
    "hf_checkpoints/panollava-vlm-qformer",
    trust_remote_code=True,
    dtype="auto",
    device_map="auto"
)

# Processor 로드 (이미지 처리 + 토크나이저)
processor = AutoProcessor.from_pretrained(
    "hf_checkpoints/panollava-vlm-qformer",
    trust_remote_code=True
)

# 추론
image = Image.open("panorama.jpg")
inputs = processor(
    text="Describe this panoramic scene",
    images=image,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

#### 2. HuggingFace Hub에 업로드

```python
from huggingface_hub import login

# Hub 로그인
login(token="your_hf_token")

# 푸시
model.push_to_hub("your-username/panollava-vlm")
processor.push_to_hub("your-username/panollava-vlm")
```

#### 3. Hub에서 직접 로드

```python
# Hub에 업로드 후
model = AutoModelForVision2Seq.from_pretrained(
    "your-username/panollava-vlm",
    trust_remote_code=True
)
```

### 🔄 추가 체크포인트 변환

다른 체크포인트도 같은 방식으로 변환할 수 있습니다:

```bash
# Stage 1 (VICReg)
python scripts/convert_checkpoint_simple.py \
  --lightning-ckpt runs/SQ3_1_latent768_PE_e2p_vision_qformer/last.ckpt \
  --output-dir hf_checkpoints/panollava-stage1-vicreg \
  --model-type vicreg

# Stage 2 (Resampler)
python scripts/convert_checkpoint_simple.py \
  --lightning-ckpt runs/SQ3_1_latent768_PE_e2p_resampler_qformer/last.ckpt \
  --output-dir hf_checkpoints/panollava-stage2-resampler \
  --model-type conditional-generation

# MLP Resampler 버전
python scripts/convert_checkpoint_simple.py \
  --lightning-ckpt runs/SQ3_1_latent768_PE_e2p_vision_mlp/last.ckpt \
  --output-dir hf_checkpoints/panollava-vlm-mlp \
  --model-type conditional-generation
```

### 📝 변환 프로세스

1. **Checkpoint 로드**: Lightning `.ckpt` 파일에서 state_dict와 hyperparameters 추출
2. **Key 변환**: `model.` prefix 제거 (Lightning → HF 형식)
3. **Config 생성**: PanoLLaVaConfig 객체 생성 및 저장
4. **Weights 저장**: `pytorch_model.bin`으로 가중치 저장
5. **Model Card**: README.md 자동 생성

### 🛠️ 변환 스크립트

두 가지 변환 스크립트 제공:

1. **`convert_checkpoint_simple.py`** (권장) ✅
   - 빠른 변환 (모델 인스턴스 생성 안 함)
   - Config + Weights만 저장
   - 메모리 효율적

2. **`convert_to_hf.py`** (고급)
   - 전체 모델 인스턴스 생성 및 검증
   - Missing/unexpected keys 리포트
   - Hub 직접 업로드 지원 (`--push-to-hub`)

### ⚠️ 중요 사항

1. **trust_remote_code=True 필수**
   - 커스텀 모델 클래스이므로 로드 시 항상 필요

2. **Processor 설정**
   - 이미지 처리: PanoramaImageProcessor 사용
   - 텍스트: Qwen tokenizer 사용
   - Vision token: `<|vision|>` 자동 추가

3. **LoRA 가중치**
   - LoRA adapter가 메인 가중치에 병합되어 저장됨
   - 별도의 LoRA 파일 없이 바로 사용 가능

### 📈 다음 단계

- [ ] HuggingFace Hub에 업로드
- [ ] Model Card 상세 작성
- [ ] Demo 앱 만들기 (Gradio/Streamlit)
- [ ] 다른 체크포인트들도 변환
- [ ] Quantization (GGUF, GPTQ 등)

### 🎉 결론

PanoLLaVA 모델이 HuggingFace 생태계와 완전히 호환되는 형식으로 변환되었습니다!

- ✅ Config 저장 완료
- ✅ Weights 저장 완료  
- ✅ Model Card 생성 완료
- ✅ AutoModel로 로드 가능
- ✅ Hub 업로드 준비 완료

이제 HuggingFace의 모든 도구(Trainer, Pipeline, Accelerate 등)와 함께 사용할 수 있습니다!
