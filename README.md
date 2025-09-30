# PanoLLaVA: Panoramic Large Vision-Language Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

PanoLLaVA는 파노라마 이미지를 이해하고 분석할 수 있는 대규모 비전-언어 모델입니다.
허깅페이스의 Vision Encoder와 Language Model을 결합하여 파노라마 이미지에 특화된 멀티모달 AI를 구현합니다.

## ✨ Features

- **파노라마 특화**: 360° 파노라마 이미지에 최적화된 모델 아키텍처
- **LoRA 지원**: Parameter-Efficient Fine-Tuning (PEFT)으로 효율적인 미세조정
- **모듈화 설계**: Vision, Language, Resampler 컴포넌트의 유연한 조합
- **다양한 백본**: SigLIP, CLIP, DINOv2 등 최신 비전 모델 지원
- **확장성**: Qwen, Llama, Gemma 등 다양한 언어 모델 지원

## 🚀 Quick Start

### 1. 설치

```bash
# 개발 환경 설정 (권장)
python setup_dev.py

# 또는 수동 설치
pip install -e .
```

### 2. 환경 확인

```bash
python check_env.py
```

### 3. 모델 선택 및 LoRA 파인튜닝

#### 간단한 단일 모델 파인튜닝
```bash
# 사용 가능한 모델 확인
python select_model.py --list

# Qwen 0.5B 모델로 빠른 LoRA 파인튜닝
python quick_lora_train.py --model qwen_0.5b --lora-r 16 --lora-alpha 32
```

#### 대규모 Ablation Study
```bash
# 대화형 모델 선택
python select_model.py --interactive

# Ablation study 실행
python lora_ablation_study.py --config configs/custom_lora_ablation.yaml
```

## 📊 LoRA Ablation Study

PanoLLaVA는 체계적인 LoRA ablation study를 지원합니다:

### 지원 모델들

**언어 모델:**
- Qwen2.5: 0.5B, 1.5B, 3B, 7B, 14B
- Llama-3.2: 1B, 3B
- Gemma-2: 2B, 9B

**비전 모델:**
- SigLIP: Base (86M), Large (427M)
- CLIP: Base (151M), Large (427M)
- DINOv2: Base (86M), Large (307M)

### LoRA 설정 예시

```yaml
# configs/lora_ablation.yaml
experiment_name: "my_ablation_study"
models:
  - name: "qwen_0.5b"
    vision_name: "google/siglip-base-patch16-224"
    language_model_name: "Qwen/Qwen2.5-0.5B-Instruct"
    latent_dimension: 768

lora_configs:
  - lora_r: 16
    lora_alpha: 32
    lora_dropout: 0.1
  - lora_r: 32
    lora_alpha: 64
    lora_dropout: 0.1
```

## 🏗️ Project Structure

```
panollava/
├── src/panovlm/           # 메인 패키지
│   ├── models/           # 모델 아키텍처
│   ├── data/             # 데이터 처리
│   ├── training/         # 학습 관련
│   └── evaluation/       # 평가 도구
├── configs/              # 설정 파일들
├── scripts/              # 실행 스크립트들
├── tests/                # 테스트 코드
├── docs/                 # 문서
├── notebooks/            # 예제 노트북
└── results/              # 실험 결과
```

## 🔧 Development

### 빌드 및 테스트

```bash
# 전체 테스트 실행
make test

# 코드 품질 검사
make lint

# Docker 빌드
make docker-build
```

### 새로운 모델 추가

1. `src/panovlm/models/`에 모델 클래스 구현
2. `src/panovlm/config.py`에 설정 추가
3. `select_model.py`에 모델 정보 추가
4. 테스트 코드 작성

## 📈 Evaluation

PanoLLaVA는 다양한 평가 메트릭을 지원합니다:

- **DINO Similarity**: 비전 피처 유사도 분석
- **Perplexity**: 언어 모델 성능 측정
- **BLEU Score**: 텍스트 생성 품질 평가
- **Custom Metrics**: 사용자의 평가 코드 통합

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning)
- [PEFT](https://github.com/huggingface/peft)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [LLaVA](https://arxiv.org/abs/2304.08485)

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/wooseungw/panollava/issues)
- **Discussions**: [GitHub Discussions](https://github.com/wooseungw/panollava/discussions)

---

**PanoLLaVA**: 파노라마 이미지의 새로운 지평을 열다! 🌅