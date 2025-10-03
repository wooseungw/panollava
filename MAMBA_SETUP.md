# Mamba-SSM CUDA Setup Guide

## 문제
`mamba_ssm`이 CUDA 12 라이브러리를 요구하지만 시스템 CUDA가 11.8인 경우 발생하는 문제입니다.

```
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```

## 해결 방법

### 1. 스크립트 사용 (권장)

```bash
# 환경 변수 설정
source fix_mamba_cuda.sh

# Python 스크립트 실행
python train.py
```

### 2. 수동 설정

```bash
# CUDA 12 라이브러리 경로 찾기
CUDA_LIB=$(find /data/3_lib/miniconda3/envs/pano/lib/python*/site-packages/nvidia/cuda_runtime/lib -name "libcudart.so.12" -exec dirname {} \; | head -1)

# 환경 변수 설정
export LD_LIBRARY_PATH="$CUDA_LIB:$LD_LIBRARY_PATH"

# 검증
python -c "import mamba_ssm; print('✅ Success')"
```

### 3. 영구 설정 (conda activate 시 자동)

```bash
# conda 환경 활성화 스크립트에 추가
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/fix_mamba.sh << 'EOF'
#!/bin/bash
CUDA_LIB=$(find $CONDA_PREFIX/lib/python*/site-packages/nvidia/cuda_runtime/lib -name "libcudart.so.12" -exec dirname {} \; | head -1)
export LD_LIBRARY_PATH="$CUDA_LIB:$LD_LIBRARY_PATH"
EOF
chmod +x $CONDA_PREFIX/etc/conda/activate.d/fix_mamba.sh

# 다음부터는 conda activate pano만 하면 자동 적용됨
```

## Optional Import

`bimamba.py`는 `mamba_ssm`을 optional로 import합니다:

```python
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
```

BiMamba 리샘플러를 사용하지 않으면 `mamba_ssm` 없이도 실행 가능합니다.

## 대안: 다른 리샘플러 사용

`mamba_ssm` 설치가 어려운 경우 다른 리샘플러를 사용하세요:

```yaml
models:
  resampler_type: mlp  # or perceiver
```