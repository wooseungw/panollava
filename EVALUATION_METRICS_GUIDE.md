# PanoLLaVA 평가 메트릭 가이드

## 🎯 지원하는 평가 메트릭

### 📝 텍스트 유사도 메트릭

| 메트릭 | 설명 | 범위 | 특징 |
|--------|------|------|------|
| **BLEU-1~4** | N-gram 기반 정밀도 | 0.0~1.0 | 기계번역에서 널리 사용, 높을수록 좋음 |
| **ROUGE-1/2/L** | Recall 기반 유사도 | 0.0~1.0 | 요약 평가에 적합, 높을수록 좋음 |
| **METEOR** | 형태소 및 동의어 고려 | 0.0~1.0 | 언어학적 유사성, 높을수록 좋음 |
| **SPICE** | 의미적 명제 매칭 | 0.0~1.0 | 의미 구조 비교, 높을수록 좋음 |

### 🖼️ 멀티모달 메트릭

| 메트릭 | 설명 | 범위 | 특징 |
|--------|------|------|------|
| **CLIP Score** | 이미지-텍스트 정렬도 | -1.0~1.0 | 시각-언어 일치성, 높을수록 좋음 |
| **CLIP-S** | CLIP 기반 유사도 | -1.0~1.0 | CLIP Score와 동일, 별명 |
| **RefCLIP-S** | 참조-예측 의미 유사도 | -1.0~1.0 | 텍스트 간 의미적 유사성, 높을수록 좋음 |

## 🛠️ 설치 및 설정

### 1. 의존성 설치
```bash
# 자동 설치 스크립트 실행
bash scripts/install_eval_metrics.sh

# 또는 수동 설치
pip install nltk rouge-score
pip install git+https://github.com/salaniz/pycocoevalcap.git
pip install git+https://github.com/openai/CLIP.git
```

### 2. Java 설치 (SPICE용)
```bash
# Ubuntu/Debian
sudo apt-get install openjdk-11-jdk

# CentOS/RHEL  
sudo yum install java-11-openjdk-devel

# macOS
brew install openjdk@11
```

## 🚀 사용 방법

### 기본 평가
```bash
python eval.py \
    --ckpt runs/e2p_finetune_mlp/best.ckpt \
    --csv-input data/quic360/test.csv \
    --batch-size 1
```

### LoRA 모델 평가
```bash
python eval.py \
    --ckpt runs/e2p_finetune_mlp/best.ckpt \
    --lora-weights-path runs/e2p_finetune_mlp/lora_weights \
    --csv-input data/quic360/test.csv
```

### 전용 스크립트 사용
```bash
bash scripts/eval_lora_finetune.sh
```

## 📊 출력 결과

### 1. CSV 파일
- `eval_results/eval_YYMMDD_HHMM.csv`
- 각 샘플별 예측/참조 텍스트와 메타데이터

### 2. JSON 메트릭 파일
- `eval_results/result_YYMMDD_HHMM.json`
- 모든 계산된 메트릭 값들

### 3. 콘솔 출력
```
📊 평가 메트릭 요약
============================================================
🔤 텍스트 유사도 메트릭:
  BLEU-4:     0.2341
  ROUGE-1:    0.4521
  ROUGE-L:    0.3892
  METEOR:     0.3156
  SPICE:      0.2847

🖼️  멀티모달 메트릭:
  CLIP Score: 0.2134 ± 0.0823
  CLIP-S:     0.2134 ± 0.0823
  RefCLIP-S:  0.6721 ± 0.1456

📈 기본 통계:
  평균 예측 길이:     12.4 단어
  평균 참조 길이:     15.2 단어
  길이 비율:         0.82
  빈 예측 비율:      2.3%
  총 평가 샘플:      487 / 500
```

## 🔍 메트릭 해석 가이드

### 높은 성능 기준
- **BLEU-4**: > 0.3 (우수), > 0.2 (양호)
- **ROUGE-L**: > 0.4 (우수), > 0.3 (양호)  
- **METEOR**: > 0.3 (우수), > 0.25 (양호)
- **SPICE**: > 0.2 (우수), > 0.15 (양호)
- **CLIP Score**: > 0.25 (우수), > 0.2 (양호)
- **RefCLIP-S**: > 0.6 (우수), > 0.5 (양호)

### 메트릭별 특징
- **BLEU**: 정확한 단어 매칭 중시, 보수적
- **ROUGE**: 재현율 중시, 관대함
- **METEOR**: 동의어/어간 고려, 균형적
- **SPICE**: 의미 구조 중시, 정교함
- **CLIP Score**: 시각-언어 정렬, 멀티모달
- **RefCLIP-S**: 텍스트 의미 유사성, 언어 중심

## ⚠️ 문제 해결

### METEOR 점수가 0인 경우
- NLTK 데이터 누락: `nltk.download('wordnet')`
- 빈 텍스트: 예측/참조 텍스트 확인
- 토큰화 오류: 특수 문자 전처리

### SPICE 계산 실패
- Java 미설치: JDK 11+ 설치 필요
- pycocoevalcap 오류: 재설치 시도
- 메모리 부족: 배치 크기 축소

### CLIP 메트릭 오류  
- GPU 메모리 부족: 청크 크기 감소
- 이미지 경로 오류: CSV의 image_path 확인
- CLIP 모델 로딩 실패: 네트워크 연결 확인

## 🎯 권장 사항

### 1. 메트릭 조합 사용
- **기본**: BLEU-4 + ROUGE-L + CLIP Score
- **상세**: 위 + METEOR + SPICE + RefCLIP-S
- **빠른 평가**: BLEU-4 + CLIP Score

### 2. 배치 크기 최적화
- **GPU 24GB**: batch_size=4, chunk_size=64
- **GPU 12GB**: batch_size=2, chunk_size=32  
- **GPU 8GB**: batch_size=1, chunk_size=16

### 3. 평가 데이터 준비
- 이미지 경로가 올바른지 확인
- 참조 텍스트가 비어있지 않은지 확인
- CSV 형식이 올바른지 검증

이제 강력하고 포괄적인 평가 시스템을 사용할 수 있습니다! 🚀
