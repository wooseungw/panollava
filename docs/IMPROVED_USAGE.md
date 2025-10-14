# PanoramaVLM 개선된 사용법 가이드

## 🚀 전체 파이프라인 개선 완료

**훈련부터 평가까지** 모든 과정을 개선했습니다! 이제 모델 로딩이 **단 한 줄**로 가능합니다.

### Before (기존 방식)
```python
# 복잡한 과정...
from train import VLMModule, safe_load_checkpoint
checkpoint = safe_load_checkpoint("runs/best.ckpt")
model = VLMModule.load_from_checkpoint("runs/best.ckpt", stage="finetune")
model.model.load_lora_weights("runs/lora_weights")  # LoRA 별도 로딩
model.eval()
model = model.to("cuda")
```

### After (새로운 방식)  
```python
# 한 줄로 끝!
from panovlm.models.model import PanoramaVLM
model = PanoramaVLM.from_checkpoint("runs/best.ckpt")
```

## 📖 새로운 API 사용법

### 1. 기본 사용법
```python
from panovlm.models.model import PanoramaVLM

# 가장 간단한 방법
model = PanoramaVLM.from_checkpoint("runs/best.ckpt")

# LoRA 자동 감지됨
# 평가 모드 자동 설정
# GPU 자동 감지 및 이동
```

### 2. HuggingFace 스타일
```python
# 디렉토리에서 자동으로 체크포인트 찾기
model = PanoramaVLM.from_pretrained("runs/panorama-vlm-e2p")

# 모델 저장
model.save_pretrained("my_panorama_model")

# 나중에 로딩
model = PanoramaVLM.from_pretrained("my_panorama_model")
```

### 3. 고급 옵션
```python
# 파라미터 오버라이드
model = PanoramaVLM.from_checkpoint(
    "runs/best.ckpt",
    vision_name="google/siglip-large-patch16-384",
    max_text_length=1024,
    device="cuda:1"
)

# LoRA 경로 직접 지정
model = PanoramaVLM.from_checkpoint(
    "runs/best.ckpt",
    lora_weights_path="custom/lora/path"
)

# LoRA 자동 감지 비활성화
model = PanoramaVLM.from_checkpoint(
    "runs/best.ckpt",
    auto_detect_lora=False
)
```

### 4. 모델 팩토리 (반복 사용 시)
```python
# 팩토리 생성
model_factory = PanoramaVLM.create_model_factory(
    "runs/best.ckpt",
    device="cuda:0"
)

# 여러 모델 인스턴스 생성
model1 = model_factory()
model2 = model_factory(max_text_length=256)
```

## 🖼️ 간편 추론 예시

### 기본 추론
```python
from panovlm.models.model import PanoramaVLM
from panovlm.processors.image import PanoramaImageProcessor
from PIL import Image
import torch

# 1. 모델 로딩
model = PanoramaVLM.from_checkpoint("runs/best.ckpt")

# 2. 이미지 전처리 (올바른 방법)
image_processor = PanoramaImageProcessor(
    image_size=(224, 224),  # 모델에 맞는 크기
    crop_strategy="e2p",    # e2p, cubemap, resize, anyres 등
    fov_deg=90,
    overlap_ratio=0.5
)
image = Image.open("panorama.jpg").convert("RGB")
pixel_values = image_processor(image)  # __call__ 메서드 사용

# 3. 추론
with torch.no_grad():
    output = model.generate(
        pixel_values=pixel_values.unsqueeze(0),  # 배치 차원 추가
        max_new_tokens=128,
        temperature=0.7
    )

print(output["text"][0])
```

### ⚠️ 중요한 사용법 주의사항

**잘못된 방법:**
```python
# ❌ preprocess_image 메서드는 존재하지 않습니다
pixel_values = image_processor.preprocess_image(image_path)
```

**올바른 방법:**
```python  
# ✅ __call__ 메서드를 사용하세요
image = Image.open(image_path).convert("RGB")
pixel_values = image_processor(image)
```

### 명령줄 추론 도구
```bash
# 간편한 추론 스크립트 제공
python simple_inference.py \
    --image panorama.jpg \
    --checkpoint runs/best.ckpt \
    --prompt "Describe this panoramic image in detail."
```

## 🔧 기존 코드 마이그레이션

### eval.py 업데이트
기존 `eval.py`는 자동으로 새로운 인터페이스를 사용하도록 업데이트되었습니다:

- ✅ 새로운 인터페이스 우선 사용
- ✅ 실패 시 기존 방식으로 자동 폴백
- ✅ 기존 코드와 100% 호환성 유지

### 기존 훈련 코드
`train.py`는 변경 없이 그대로 사용 가능합니다. 새로운 인터페이스는 **평가 및 추론 전용**입니다.

## 🛠️ 주요 기능들

### 자동 LoRA 감지
```python
# 체크포인트와 같은 디렉토리의 lora_weights 폴더 자동 감지
model = PanoramaVLM.from_checkpoint("runs/best.ckpt")
# → runs/lora_weights가 있으면 자동 로딩
```

### 스마트 디바이스 관리  
```python
# 자동 GPU 감지
model = PanoramaVLM.from_checkpoint("runs/best.ckpt")  # device="auto" 기본값

# 특정 디바이스 지정
model = PanoramaVLM.from_checkpoint("runs/best.ckpt", device="cuda:1")
```

### 설정 유지 및 오버라이드
```python
# 체크포인트의 원본 설정 사용
model = PanoramaVLM.from_checkpoint("runs/best.ckpt")

# 특정 설정만 오버라이드
model = PanoramaVLM.from_checkpoint(
    "runs/best.ckpt",
    max_text_length=1024  # 이 값만 변경, 나머지는 원본 유지
)
```

### 에러 처리 및 디버깅
```python
try:
    model = PanoramaVLM.from_checkpoint("runs/best.ckpt")
    
    # 모델 정보 확인
    print(f"LoRA 활성화: {model.get_lora_info().get('is_lora_enabled')}")
    print(f"총 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
except FileNotFoundError:
    print("체크포인트 파일을 찾을 수 없습니다")
except RuntimeError as e:
    print(f"모델 로딩 실패: {e}")
```

## 📚 전체 예시 파일들

1. **`model_usage_examples.py`** - 모든 사용법의 상세한 예시
2. **`simple_inference.py`** - 명령줄 추론 도구
3. **`IMPROVED_USAGE.md`** - 이 가이드 문서

## 🔄 호환성

- ✅ 기존 `eval.py` 코드와 100% 호환
- ✅ 기존 `train.py` 코드 변경 불필요  
- ✅ 기존 체크포인트 파일 그대로 사용 가능
- ✅ 기존 LoRA 가중치 파일 그대로 사용 가능

## 💡 핵심 장점

1. **간편성**: 복잡한 로딩 과정을 한 줄로 압축
2. **자동화**: LoRA 감지, 디바이스 설정, 평가 모드 등 자동 처리
3. **친숙함**: HuggingFace 스타일의 친숙한 API
4. **안전성**: 에러 처리 및 폴백 메커니즘 내장
5. **호환성**: 기존 코드와 완벽 호환

## 🔧 훈련 코드 개선사항

### 새로운 모델 저장 방식
훈련 완료 후 **3가지 형태**로 자동 저장됩니다:

```
runs/panorama-vlm_e2p_finetune_mlp/
├── best.ckpt              # Lightning 체크포인트 (LoRA 자동 감지됨)
├── hf_model/              # HuggingFace 스타일 모델
├── panorama_model/        # 간편 로딩용 (가장 추천)
├── lora_weights/          # LoRA 가중치 (별도)
└── model_final.safetensors # 기존 방식 (호환성)
```

### 훈련 완료 후 자동 안내
이제 훈련이 끝나면 **사용법을 자동으로 출력**합니다:

```
🎉 훈련 완료! 모델 사용법:
================================================================================

🚀 새로운 간편 사용법:
   # 방법 1: Lightning 체크포인트 (LoRA 자동 감지)
   model = PanoramaVLM.from_checkpoint('runs/.../best.ckpt')

   # 방법 2: HuggingFace 스타일 (가장 간편)
   model = PanoramaVLM.from_pretrained('runs/.../panorama_model')

💡 빠른 추론 테스트:
   python simple_inference.py \
     --checkpoint 'runs/.../best.ckpt' \
     --image your_panorama.jpg
```

## 🛠️ 새로운 유틸리티 함수

### `model_utils.py` - 통합 모델 관리
```python
from model_utils import quick_load, print_model_info

# 빠른 로딩
model = quick_load("runs/best.ckpt")

# 모델 정보 확인
print_model_info(model)

# 훈련용/추론용 구분 로딩
train_model = load_for_training("runs/best.ckpt")  
inference_model = load_for_inference("runs/best.ckpt")
```

### 명령줄 모델 정보 확인
```bash
# 모델 정보만 빠르게 확인
python model_utils.py --checkpoint runs/best.ckpt --info-only
```

## 🔄 완전한 호환성

### 훈련 코드 (`train.py`)
- ✅ **변경 불필요** - 기존 스크립트 그대로 사용
- ✅ 새로운 저장 방식 자동 적용
- ✅ 체크포인트에 더 많은 메타데이터 포함

### 평가 코드 (`eval.py`)  
- ✅ **자동 업데이트** - 새 인터페이스 우선 사용
- ✅ 실패 시 기존 방식으로 자동 폴백
- ✅ 기존 명령행 옵션 그대로 지원

### 기존 체크포인트
- ✅ **완벽 호환** - 기존 파일 그대로 사용 가능
- ✅ LoRA 가중치 자동 감지
- ✅ 메타데이터 자동 추출

## 📁 전체 파일 구조

```
PanoLLaVA/
├── panovlm/
│   └── model.py                 # ✨ 새로운 통합 인터페이스 추가
├── train.py                     # ✨ 개선된 저장 방식 + 사용법 안내
├── eval.py                      # ✨ 새 인터페이스 자동 활용
├── model_utils.py              # 🆕 통합 유틸리티 함수
├── model_usage_examples.py     # 🆕 상세 사용법 예시
├── simple_inference.py         # 🆕 명령줄 추론 도구
└── IMPROVED_USAGE.md           # 🆕 완전한 가이드 (이 문서)
```

## 🎯 핵심 장점 정리

1. **완벽한 일관성**: 훈련-저장-로딩-평가 전 과정 통합
2. **자동화**: LoRA 감지, 디바이스 설정, 메타데이터 처리 등 모든 것이 자동
3. **다양한 방식**: Lightning, HuggingFace, SafeTensors 등 다중 저장/로딩 지원  
4. **안전성**: 폴백 메커니즘으로 기존 코드 완벽 호환
5. **사용성**: 직관적인 API와 자동 사용법 안내

이제 PanoramaVLM을 **더 쉽고 빠르게** 사용할 수 있습니다! 🎉

## 🚀 마이그레이션 가이드

### 기존 사용자
**아무것도 변경할 필요 없습니다!** 기존 스크립트가 그대로 작동하면서 새로운 기능을 자동으로 활용합니다.

### 새로운 사용자
다음 한 줄로 시작하세요:
```python
from panovlm.models.model import PanoramaVLM
model = PanoramaVLM.from_checkpoint("path/to/checkpoint")
```