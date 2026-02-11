# PanoLLaVA 리팩토링 제안서

## 📋 현재 문제점 분석

### 1. 순환 참조 위험 영역

#### 🔴 높은 위험도
```
processors/ ↔ models/
- models/model.py → processors.anyres_integration (compute_vicreg_anyres_loss)
- models/language_fusion.py → processors.universal_text_formatter
- processors/image.py → processors.anyres_e2p (같은 폴더 내)
```

#### 🟡 중간 위험도
```
dataset.py → processors.*
- dataset.py가 processors의 거의 모든 모듈을 import
- 만약 processors가 dataset을 import하면 즉시 순환참조 발생
```

### 2. 폴더 구조 가독성 문제

#### 현재 구조의 문제점
```
src/panovlm/
├── processors/          # 🔴 문제: 역할이 불명확
│   ├── image.py         # 이미지 전처리
│   ├── anyres_e2p.py    # 파노라마 타일링
│   ├── anyres_integration.py  # VICReg 손실 계산 (❌ 손실 함수인데 processors에 위치)
│   ├── pano_llava_processor.py  # 통합 프로세서
│   ├── universal_text_formatter.py  # 텍스트 포맷팅
│   └── vision.py        # Vision wrapper
├── dataset.py           # 🟡 루트에 단독 위치 (data/ 폴더가 없음)
├── model.py             # 🟡 단순 re-export (models/model.py의 wrapper)
└── config.py            # 🟡 config/ 폴더와 중복
```

#### 구체적 문제
1. **`processors/anyres_integration.py`**: 손실 함수(`compute_vicreg_anyres_loss`)인데 processors에 위치
2. **`dataset.py`**: 루트에 고립되어 있고, data 관련 모듈이 분산됨
3. **중복된 진입점**: `model.py`, `config.py`가 하위 폴더의 wrapper 역할만 함
4. **processors 역할 혼재**: 데이터 전처리 + 손실 계산 + 텍스트 포맷팅이 혼재

---

## 🎯 리팩토링 제안

### Phase 1: 손실 함수 재배치 (우선순위: 높음)

#### 이동할 파일
```bash
processors/anyres_integration.py → losses/anyres_integration.py
```

#### 이유
- `compute_vicreg_anyres_loss`는 **손실 함수**이므로 `losses/` 폴더에 위치해야 함
- `losses/` 폴더에는 이미 `vicreg_overlap.py`, `vicreg_projector.py`가 있음
- 일관성 있는 구조 유지

#### 변경 내용
```python
# Before (models/model.py)
from ..processors.anyres_integration import compute_vicreg_anyres_loss

# After (models/model.py)
from ..losses.anyres_integration import compute_vicreg_anyres_loss
```

```python
# losses/__init__.py에 추가
from .anyres_integration import compute_vicreg_anyres_loss

__all__ = [
    "VicRegLoss",
    "compute_vicreg_overlap_loss",
    "VICRegProjector",
    "compute_vicreg_anyres_loss",  # NEW
]
```

---

### Phase 2: 데이터 관련 모듈 통합 (우선순위: 중간)

#### 새로운 구조
```
src/panovlm/
├── data/                    # 📁 NEW: 데이터 관련 통합
│   ├── __init__.py
│   ├── datasets.py          # dataset.py 이름 변경 (복수형)
│   └── collators.py         # 데이터 collation 로직 (필요시)
```

#### 이동 계획
```bash
dataset.py → data/datasets.py
```

#### 업데이트가 필요한 파일들
```python
# scripts/train.py
# Before
from panovlm.dataset import VLMDataModule

# After
from panovlm.data import VLMDataModule
# 또는
from panovlm.data.datasets import VLMDataModule
```

#### 하위 호환성 유지
```python
# panovlm/dataset.py (backward compatibility shim)
"""Deprecated: Use panovlm.data.datasets instead."""
import warnings
from .data.datasets import *

warnings.warn(
    "panovlm.dataset is deprecated. Use panovlm.data.datasets instead.",
    DeprecationWarning,
    stacklevel=2
)
```

---

### Phase 3: Processors 재구성 (우선순위: 중간)

#### 현재 processors의 역할 정리
```
processors/
├── image.py                 # ✅ 이미지 전처리 (유지)
├── anyres_e2p.py           # ✅ 파노라마 타일링 (유지)
├── vision.py               # ✅ Vision wrapper (유지)
├── pano_llava_processor.py # ✅ 통합 프로세서 (유지)
├── universal_text_formatter.py  # 🔄 텍스트 → text/ 폴더로 이동 고려
└── anyres_integration.py   # ❌ losses/로 이동 (Phase 1)
```

#### 선택적 재구성 (텍스트 처리 분리)
```
src/panovlm/
├── processors/
│   ├── image/              # 이미지 관련만
│   │   ├── __init__.py
│   │   ├── panorama.py     # PanoramaImageProcessor
│   │   ├── anyres_e2p.py
│   │   └── vision.py
│   └── text/               # 텍스트 관련만
│       ├── __init__.py
│       └── formatter.py    # UniversalTextFormatter
```

**장점**: 책임이 명확히 분리됨  
**단점**: 기존 import 경로가 모두 변경됨 (대규모 변경)

---

### Phase 4: 루트 레벨 Wrapper 제거 (우선순위: 낮음)

#### 제거 대상
```
src/panovlm/
├── model.py       # ❌ 제거 (models/model.py의 단순 wrapper)
└── config.py      # ❌ 제거 또는 통합 (config/ 폴더와 중복)
```

#### 대안
```python
# panovlm/__init__.py에서 직접 export
from .models.model import PanoramaVLM
from .config import Config, ModelConfig

__all__ = ["PanoramaVLM", "Config", "ModelConfig"]
```

---

## 📊 의존성 그래프 (리팩토링 후)

### 올바른 의존성 방향
```
외부 라이브러리 (torch, transformers, PIL)
    ↑
processors/ (데이터 전처리만)
    ↑
data/ (데이터셋)
    ↑
losses/ (손실 함수, VICReg 포함)
    ↑
models/ (모델 아키텍처)
    ↑
training/ (훈련 로직)
```

### 핵심 원칙
1. **하위 레벨은 상위 레벨을 import하지 않음**
2. **processors는 순수 전처리만 담당** (손실 함수 제외)
3. **losses는 models를 import하지 않음**
4. **models는 config, losses, processors만 import**

---

## 🚀 실행 계획

### Step 1: 손실 함수 이동 (Breaking Change 최소화)
```bash
# 1. 파일 이동
mv src/panovlm/processors/anyres_integration.py src/panovlm/losses/anyres_integration.py

# 2. Import 업데이트
# models/model.py 수정
sed -i 's/from ..processors.anyres_integration/from ..losses.anyres_integration/g' \
    src/panovlm/models/model.py

# 3. losses/__init__.py 업데이트
# (수동으로 export 추가)
```

### Step 2: 하위 호환 Shim 생성 (선택적)
```python
# processors/anyres_integration.py (deprecated shim)
"""
Deprecated: This module has been moved to panovlm.losses.anyres_integration
"""
import warnings
from ..losses.anyres_integration import *

warnings.warn(
    "panovlm.processors.anyres_integration is deprecated. "
    "Use panovlm.losses.anyres_integration instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### Step 3: 데이터 폴더 생성 (점진적 마이그레이션)
```bash
# 1. 폴더 생성
mkdir -p src/panovlm/data

# 2. 파일 이동 + 이름 변경
mv src/panovlm/dataset.py src/panovlm/data/datasets.py

# 3. __init__.py 생성
cat > src/panovlm/data/__init__.py << 'EOF'
"""Data module - contains dataset classes."""
from .datasets import *
__all__ = ["VLMDataModule", "BaseChatPanoDataset", "ChatPanoDataset", "ChatPanoTestDataset"]
EOF

# 4. 하위 호환 shim 생성
cat > src/panovlm/dataset.py << 'EOF'
"""Deprecated: Use panovlm.data.datasets instead."""
import warnings
from .data.datasets import *
warnings.warn("...", DeprecationWarning, stacklevel=2)
EOF
```

### Step 4: Import 업데이트
```bash
# scripts/train.py 등 업데이트
find scripts/ -name "*.py" -exec sed -i \
    's/from panovlm.dataset/from panovlm.data/g' {} \;
```

---

## ✅ 검증 체크리스트

### Phase 1 완료 후
- [ ] `python -m pytest tests/` 통과
- [ ] `python scripts/train.py --config configs/default.yaml --help` 정상 실행
- [ ] Import 경로 변경 확인: `grep -r "processors.anyres_integration" src/`
- [ ] 순환 참조 없음 확인: `python -c "from panovlm.models.model import PanoramaVLM"`

### Phase 2 완료 후
- [ ] Dataset import 정상: `python -c "from panovlm.data import VLMDataModule"`
- [ ] 하위 호환성 확인: `python -c "from panovlm.dataset import VLMDataModule"` (warning 발생)
- [ ] 전체 테스트 통과

### Phase 3 완료 후 (선택적)
- [ ] 새로운 구조로 import 확인
- [ ] 문서 업데이트 (README.md, docs/)
- [ ] `.github/copilot-instructions.md` 업데이트

---

## 🎨 최종 권장 구조

```
src/panovlm/
├── __init__.py           # Main exports
├── processors/           # 📦 데이터 전처리만
│   ├── __init__.py
│   ├── image.py          # PanoramaImageProcessor
│   ├── anyres_e2p.py     # ERP tiling
│   ├── vision.py         # Vision wrapper
│   ├── pano_llava_processor.py
│   └── universal_text_formatter.py
├── data/                 # 📦 데이터셋 (NEW)
│   ├── __init__.py
│   └── datasets.py       # VLMDataModule, ChatPanoDataset
├── losses/               # 📦 손실 함수 (확장됨)
│   ├── __init__.py
│   ├── vicreg_overlap.py
│   ├── vicreg_projector.py
│   └── anyres_integration.py  # ✨ NEW (from processors/)
├── models/               # 📦 모델 아키텍처
│   ├── model.py
│   ├── language_fusion.py
│   ├── vision/
│   └── resampler/
├── config/               # 📦 설정 관리
├── training/             # 📦 훈련 로직
├── evaluation/           # 📦 평가 도구
└── utils/                # 📦 유틸리티
```

---

## 💡 추가 권장사항

### 1. Import 스타일 통일
```python
# ✅ 권장: 절대 import
from panovlm.processors.image import PanoramaImageProcessor

# ❌ 지양: 상대 import (깊이 2 이상)
from ...processors.image import PanoramaImageProcessor
```

### 2. __init__.py 명확화
```python
# processors/__init__.py
"""Image and text preprocessing utilities."""
from .image import PanoramaImageProcessor
from .vision import VisionProcessorWrapper
# ... (명시적 export만)

__all__ = ["PanoramaImageProcessor", "VisionProcessorWrapper", ...]
```

### 3. Circular Import 방지 패턴
```python
# TYPE_CHECKING을 활용한 타입 힌트
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from panovlm.models.model import PanoramaVLM  # 타입 체크용만

# 런타임에는 import하지 않음
```

---

## 📝 마이그레이션 타임라인

### Week 1: Phase 1 (손실 함수 이동)
- 영향도: 낮음
- Breaking changes: 최소
- 작업량: 1-2일

### Week 2: Phase 2 (데이터 폴더 생성)
- 영향도: 중간
- Breaking changes: 하위 호환 shim으로 완화
- 작업량: 2-3일

### Week 3-4: Phase 3 (선택적 - 프로세서 재구성)
- 영향도: 높음
- Breaking changes: 많음
- 작업량: 5-7일
- **권장**: Phase 1, 2 완료 후 안정화 기간을 거쳐 진행

---

## 🔍 자주 묻는 질문 (FAQ)

### Q1: 왜 processors에서 손실 함수를 분리해야 하나요?
**A**: `anyres_integration.py`의 `compute_vicreg_anyres_loss`는 **손실 계산** 로직입니다. Processors는 **데이터 전처리**만 담당해야 하며, 손실 함수는 `losses/` 폴더에 위치해야 의존성 방향이 올바릅니다.

### Q2: 하위 호환성은 어떻게 유지하나요?
**A**: Deprecation shim을 사용합니다. 기존 import 경로도 작동하지만 warning을 출력하여 점진적 마이그레이션을 유도합니다.

### Q3: 모든 Phase를 다 해야 하나요?
**A**: 아니요. **Phase 1만 완료해도 순환 참조 위험이 크게 감소**합니다. Phase 2, 3는 선택적으로 진행 가능합니다.

---

## 📞 Support

리팩토링 중 문제가 발생하면:
1. `git stash` 로 변경사항 임시 저장
2. `python -m pytest tests/` 로 회귀 테스트
3. Issue tracker에 문의

**Good luck with refactoring! 🚀**
