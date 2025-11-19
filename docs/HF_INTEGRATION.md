# HuggingFace Integration Guide

본 문서는 PanoLLaVA를 HuggingFace Transformers와 통합한 구현 상세를 설명합니다.

## 📋 개요

### 목적
기존 PanoLLaVA 모델을 HuggingFace Hub에 업로드하고 `from_pretrained()`, `push_to_hub()` 등의 표준 HF API를 사용할 수 있도록 변환

### 주요 변경사항
1. **새로운 디렉토리**: `hf_models/` - HF 호환 래퍼 클래스
2. **이중 목적 설계**: 
   - Full VLM (이미지 → 텍스트)
   - Feature Extractor (이미지 → 임베딩)
3. **3단계 학습 지원**: VICReg → Resampler → Finetune

## 🏗️ 아키텍처

### 기존 구조 (src/panovlm/)
```
src/panovlm/
├── models/
│   ├── model.py              # PanoramaVLM (Lightning 모듈)
│   ├── vision/               # VisionBackbone, ResamplerModule
│   └── language_fusion.py
├── processors/               # PanoramaImageProcessor
├── losses/                   # VicRegLoss
└── config/                   # Config
```

### HuggingFace 구조 (hf_models/)
```
hf_models/
├── __init__.py
├── configuration_panollava.py    # PanoLLaVaConfig (PretrainedConfig)
├── modeling_panollava.py         # 모델 클래스 3개
├── processing_panollava.py       # PanoLLaVaProcessor
├── example_usage.py              # 사용 예시
├── test_hf_compatibility.py      # 호환성 테스트
└── README.md
```

## 🔄 구현 세부사항

### 1. Configuration (configuration_panollava.py)

**클래스**: `PanoLLaVaConfig(PretrainedConfig)`

**주요 파라미터**:
```python
{
  "model_type": "panollava",
  "is_composition": true,
  
  # Vision
  "vision_config": {
    "model_name": "google/siglip-base-patch16-224",
    "image_size": 224,
    "patch_size": 16
  },
  
  # Text
  "text_config": {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "hidden_size": 896
  },
  
  # Resampler
  "resampler_type": "mlp",  # or "qformer", "bimamba"
  "latent_dimension": 768,
  "num_latent_tokens": 64,
  
  # VICReg
  "vicreg_similarity_weight": 25.0,
  "vicreg_variance_weight": 25.0,
  "vicreg_covariance_weight": 1.0,
  "overlap_ratio": 0.3,
  
  # Image Processing
  "crop_strategy": "anyres_e2p",
  "num_views": 4,
  "fov_deg": 90.0,
  
  # LoRA
  "use_lora": false,
  "lora_r": 16,
  "lora_alpha": 32
}
```

**HF 호환성**:
- ✅ `from_pretrained()` - JSON config 자동 로드
- ✅ `save_pretrained()` - config.json 자동 저장
- ✅ `to_dict()` - 직렬화
- ✅ `is_composition=True` - 멀티모달 마커

### 2. Modeling (modeling_panollava.py)

#### 2.1 PanoLLaVaPreTrainedModel (Base)

**역할**: 가중치 초기화, gradient checkpointing 지원

```python
class PanoLLaVaPreTrainedModel(PreTrainedModel):
    config_class = PanoLLaVaConfig
    base_model_prefix = "panollava"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PanoLLaVaVisionEncoder", "PanoLLaVaResampler"]
```

#### 2.2 PanoLLaVaForConditionalGeneration

**용도**: 전체 VLM (비전 → 언어)

**구성요소**:
```python
self.vision_backbone    # VisionBackbone (from src/panovlm)
self.resampler          # ResamplerModule
self.projector          # PanoramaProjector
self.language_model     # AutoModelForCausalLM (Qwen/Llama/Gemma)
self.language_fusion    # LanguageFusion
```

**Forward 흐름**:
1. `pixel_values` → vision_backbone → `[B*V, H'*W', D_vision]`
2. Resampler → `[B*V, N, D_latent]`
3. Projector → `[B*V, N, D_lm]`
4. LanguageFusion.fuse() → `{inputs_embeds, attention_mask, labels}`
5. language_model → `{loss, logits}`

**출력**: `PanoLLaVaCausalLMOutput`

#### 2.3 PanoLLaVaForVICReg

**용도**: Stage 1 VICReg 학습 전용

**구성요소**:
```python
self.vision_backbone
self.resampler
self.vicreg_projector   # VICRegProjector (학습 후 폐기)
```

**Forward 흐름**:
1. `pixel_values [B, V, C, H, W]` → vision features
2. Resampler → `[B, V, N, D]`
3. VICReg projector → `[B, V, N, D']`
4. `_compute_vicreg_overlap_loss()` → invariance, variance, covariance

**출력**: `PanoLLaVaVICRegOutput`

#### 2.4 PanoLLaVaForFeatureExtraction

**용도**: 비전 임베딩 추출 전용 (LM 없음)

**구성요소**:
```python
self.vision_backbone
self.resampler
```

**Forward 흐름**:
1. `pixel_values` → vision features → `[B*V, H'*W', D]`
2. Resampler → `[B, V*N, D]`
3. Pooling (mean/max/first) → `[B, D]`

**출력**: `PanoLLaVaFeatureOutput`

### 3. Processing (processing_panollava.py)

**클래스**: `PanoLLaVaProcessor(ProcessorMixin)`

**역할**: 이미지 + 텍스트 통합 처리

**구성요소**:
```python
self.image_processor  # AutoImageProcessor (from vision model)
self.tokenizer        # AutoTokenizer (from text model)
self.vision_token     # '<|vision|>'
```

**`__call__()` 로직**:
```python
# 이미지 처리
image_inputs = self.image_processor(images)  # {pixel_values}

# 텍스트 처리 (chat template 적용)
if hasattr(tokenizer, 'apply_chat_template'):
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, ...)
text_inputs = self.tokenizer(formatted)  # {input_ids, attention_mask}

# 통합
return {**image_inputs, **text_inputs}
```

## 🔗 기존 코드와의 연결

### Import 구조

**HF 모델에서 기존 컴포넌트 재사용**:

```python
# modeling_panollava.py 내부
from ..src.panovlm.models.vision import VisionBackbone, ResamplerModule
from ..src.panovlm.models.language_fusion import LanguageFusion
from ..src.panovlm.losses.vicreg_overlap import VICRegProjector, compute_vicreg_overlap_loss
```

**장점**:
- ✅ 기존 코드 재사용 (중복 없음)
- ✅ 버그 픽스 자동 반영
- ✅ 단일 구현 유지

### 학습 스크립트 통합

**기존 Lightning 학습 유지**:
```bash
python scripts/train.py --config configs/default.yaml
```

**HF 체크포인트 변환**:
```python
# Lightning checkpoint → HF format
from hf_models import PanoLLaVaForConditionalGeneration

# 1. Load Lightning checkpoint
ckpt = torch.load("runs/stage3_final.ckpt")
state_dict = ckpt['state_dict']

# 2. Create HF model
model = PanoLLaVaForConditionalGeneration.from_pretrained(config)

# 3. Load weights (key mapping may be needed)
model.load_state_dict(state_dict, strict=False)

# 4. Save in HF format
model.save_pretrained("hf_checkpoints/panollava-vlm")
```

## 📦 HuggingFace Hub 워크플로우

### 1. 체크포인트 준비

**Stage별 변환**:

```bash
# Stage 1: VICReg
python scripts/convert_to_hf.py \
  --lightning-ckpt runs/ADDDATA_SQ3_1_latent768_PE_anyres_e2p_vision_mlp/vision_final.ckpt \
  --output-dir hf_checkpoints/panollava-stage1-vicreg \
  --model-type vicreg

# Stage 2: Resampler
python scripts/convert_to_hf.py \
  --lightning-ckpt runs/.../resampler_final.ckpt \
  --output-dir hf_checkpoints/panollava-stage2-resampler \
  --model-type conditional-generation

# Stage 3: Final VLM
python scripts/convert_to_hf.py \
  --lightning-ckpt runs/.../finetune_final.ckpt \
  --output-dir hf_checkpoints/panollava-vlm \
  --model-type conditional-generation
```

### 2. Hub 업로드

```python
from huggingface_hub import HfApi, login

login(token="your_hf_token")

# Model
model.push_to_hub("your-org/panollava-vlm")

# Processor
processor.push_to_hub("your-org/panollava-vlm")

# Config (자동 포함됨)
```

### 3. 사용자 로드

```python
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained(
    "your-org/panollava-vlm",
    trust_remote_code=True  # 중요!
)

model = AutoModelForVision2Seq.from_pretrained(
    "your-org/panollava-vlm",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

**주의**: `trust_remote_code=True` 필수 (커스텀 모델 클래스)

## 🧪 테스트

### 호환성 테스트

```bash
python hf_models/test_hf_compatibility.py
```

**테스트 항목**:
1. ✅ Import 성공
2. ✅ Config 생성 및 직렬화
3. ✅ 모델 구조 생성 (가중치 미포함)
4. ✅ AutoClass 등록
5. ✅ Processor 생성

### 통합 테스트 (실제 가중치)

```python
# Stage 1 테스트
python hf_models/example_usage.py --example 3

# Full VLM 테스트
python hf_models/example_usage.py --example 1

# Feature 추출 테스트
python hf_models/example_usage.py --example 2
```

## 🔧 디버깅 가이드

### Import Error

**문제**: `ModuleNotFoundError: No module named 'panovlm'`

**해결**:
```bash
cd /data/1_personal/4_SWWOO/panollava
pip install -e .
```

### trust_remote_code 경고

**문제**: `trust_remote_code=True` 없이 로드 시도

**해결**:
```python
# 항상 trust_remote_code=True 추가
AutoModelForVision2Seq.from_pretrained(..., trust_remote_code=True)
```

### 가중치 불일치

**문제**: Lightning checkpoint key와 HF key 불일치

**해결**:
```python
# Key mapping 적용
state_dict_mapped = {}
for k, v in lightning_state_dict.items():
    new_key = k.replace('model.', '').replace('module.', '')
    state_dict_mapped[new_key] = v

model.load_state_dict(state_dict_mapped, strict=False)
```

## 📝 TODO

- [ ] `scripts/convert_to_hf.py` 구현 (Lightning → HF 변환)
- [ ] CSV 파일 수정 (unquoted comma 문제)
- [ ] Model Card 템플릿 작성
- [ ] CI/CD for HF Hub 자동 업로드
- [ ] Gradio 데모 앱

## 🎓 참고 문서

- HuggingFace Custom Models: https://huggingface.co/docs/transformers/custom_models
- Vision-Language Models: https://huggingface.co/docs/transformers/model_doc/llava
- PEFT/LoRA: https://huggingface.co/docs/peft/
- Model Hub: https://huggingface.co/docs/hub/models-adding-libraries

## 🙏 기여

이 구현은 PanoLLaVA 프로젝트의 일부로, HuggingFace 생태계와의 통합을 목표로 합니다.
