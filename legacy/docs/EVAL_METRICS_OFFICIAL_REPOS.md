# PanoLLaVA 평가 메트릭 - 공식 레포지토리 가이드

## 개요

`eval.py`의 모든 평가 메트릭은 **공식 레포지토리**에서 제공하는 구현을 사용합니다.

## 메트릭별 공식 레포지토리

### 1️⃣ BLEU-4 (sacrebleu)

**출처**: https://github.com/mjpost/sacrebleu

**설명**: 
- 기계 번역 및 텍스트 생성 품질 평가의 표준 메트릭
- n-gram 정확도 기반 (1-gram ~ 4-gram)
- Moses 토크나이저 사용 (학술 표준)

**설치**:
```bash
pip install sacrebleu
```

**특징**:
- ✅ 재현 가능한 결과 (표준 토크나이저)
- ✅ 다양한 스무딩 메서드 지원
- ✅ 신뢰할 수 있는 학술 구현

**폴백**: sacrebleu 미설치 시 NLTK 사용

---

### 2️⃣ METEOR (NLTK)

**출처**: https://www.nltk.org/ | NLTK 내장 구현

**설명**:
- 동의어 및 어근 일치를 고려한 평가 메트릭
- BLEU-4의 단점 개선 (의미론적 유사도 고려)
- WordNet 기반 동의어 매칭

**설치**:
```bash
pip install nltk
# NLTK 데이터는 자동 다운로드됨 (wordnet, punkt)
```

**특징**:
- ✅ 의미론적 유사도 고려
- ✅ 어근 추출 및 동의어 매칭
- ✅ 정밀도/재현율 균형

**구성**:
```
METEOR = 0.9 * (P * R) / (0.1 * P + R)
```
- P (Precision): 모델이 정확히 맞춘 비율
- R (Recall): 정답 중 모델이 맞춘 비율

---

### 3️⃣ ROUGE-L (rouge-score)

**출처**: https://github.com/google-research/rouge

**설명**:
- 최대 공통 부분수열(LCS) 기반 평가
- 텍스트 요약, 이미지 캡션 평가에 적합
- 대소문자 무시, 형태소 분석 가능

**설치**:
```bash
pip install rouge-score
```

**특징**:
- ✅ Google 공식 구현
- ✅ 메모리 효율적 (배치 처리 지원)
- ✅ Stemming 옵션 (형태소 기반 비교)

**구성**:
```
ROUGE-L = F-measure(Precision, Recall)
- Precision: 생성된 텍스트 중 참조와 겹치는 부분의 비율
- Recall: 참조 텍스트 중 생성된 텍스트와 겹치는 부분의 비율
```

---

### 4️⃣ SPICE (pycocoevalcap)

**출처**: https://github.com/salaniz/pycocoevalcap

**원본 논문**: "SPICE: Semantic Propositional Image Caption Evaluation" (ECCV 2016)

**설명**:
- 의미적 명제(semantic propositions) 기반 평가
- 이미지 캡션/VLM 평가의 표준 메트릭
- StanfordCoreNLP 기반 의존성 구조 분석

**설치**:
```bash
# 공식 GitHub 레포지토리에서 설치
pip install git+https://github.com/salaniz/pycocoevalcap.git

# 또는 로컬에서 설치
cd pycocoevalcap
pip install -e .
```

**특징**:
- ✅ 의미적 명제 그래프 구조 분석
- ✅ VLM 평가에 최적화됨
- ⚠️ Java 의존성 (StanfordCoreNLP)
- ✅ 폴백: 의미적 유사도(SentenceTransformer) 사용 가능

**대안 (Java 미사용)**:
```
SPICE 계산 실패 시 → SentenceTransformer 기반 의미 유사도로 대체
- 모델: all-MiniLM-L6-v2 (경량, 빠름)
- 거리: 코사인 유사도
```

---

### 5️⃣ CIDEr (pycocoevalcap)

**출처**: https://github.com/salaniz/pycocoevalcap

**원본 논문**: "CIDEr: Consensus-based Image Description Evaluation" (CVPR 2015)

**설명**:
- TF-IDF 기반 용어 신뢰도(Term Frequency-Inverse Document Frequency)
- n-gram 매칭 + 가중치 적용
- 이미지 캡션/VLM 평가의 표준 메트릭

**설치**:
```bash
# 공식 GitHub 레포지토리에서 설치
pip install git+https://github.com/salaniz/pycocoevalcap.git
```

**특징**:
- ✅ 일관성 기반 평가 (CIDEr = Consensus-based)
- ✅ TF-IDF 가중치 적용
- ✅ n-gram 정확도 계산 (1~4-gram)
- ✅ 이미지 캡션 평가에 최적화됨

**구성**:
```
CIDEr = (1/n) * Σ CIDEr_i
- 각 생성 캡션마다: TF-IDF 가중 n-gram 매칭
- 가중치: Inverse Document Frequency 기반
```

---

## 설치 가이드

### 최소 설치 (BLEU-4만 필요)
```bash
pip install sacrebleu
```

### 권장 설치 (모든 메트릭)
```bash
# 기본 평가 메트릭
pip install sacrebleu nltk rouge-score

# SPICE, CIDEr (이미지 캡션 평가)
pip install git+https://github.com/salaniz/pycocoevalcap.git

# SPICE 폴백 (의미 유사도)
pip install sentence-transformers
```

### 전체 설치 (한 번에)
```bash
# requirements.txt에 모두 포함되어 있음
pip install -r requirements.txt
```

### Docker 사용
```bash
# Dockerfile에 모든 의존성 포함
docker build -t panollava .
docker run --gpus all -it panollava
```

---

## 사용 예시

### 기본 사용법
```bash
# Config 파일 기반
python scripts/eval.py --config configs/default.yaml \
                       --csv-input data/quic360/test.csv

# 체크포인트 디렉토리 지정
python scripts/eval.py --checkpoint-dir runs/my_model/ \
                       --csv-input data/quic360/test.csv

# CSV만 평가 (모델 없이)
python scripts/eval.py --csv-input predictions.csv
```

### CSV 형식
```csv
image_path,original_query,prediction,reference,pred_length,ref_length,is_error,is_empty
path/to/image.jpg,What do you see?,Generated caption,Reference caption,10,12,False,False
...
```

---

## 메트릭 선택 가이드

| 메트릭 | 사용 케이스 | 강점 | 약점 |
|--------|-----------|------|------|
| **BLEU-4** | 기계 번역, NMT | 빠름, 재현성 | 의미 미흡, 동의어 무시 |
| **METEOR** | 텍스트 생성 | 의미론적 유사도 | 느림, 언어 의존성 |
| **ROUGE-L** | 요약, 캡션 | 메모리 효율, 의미성 | LCS 기반 (문법 무시) |
| **SPICE** | VLM, 캡션 ⭐ | 의미적 명제 분석 | Java 의존성, 느림 |
| **CIDEr** | VLM, 캡션 ⭐ | TF-IDF 기반, 합의도 | 계산 비용 높음 |

**추천**: SPICE + CIDEr를 함께 사용 (VLM 평가의 표준)

---

## 문제 해결

### sacrebleu 설치 실패
```bash
pip install --upgrade sacrebleu
# 또는 GitHub에서 직접 설치
pip install git+https://github.com/mjpost/sacrebleu.git
```

### pycocoevalcap 설치 실패
```bash
# Java가 설치되어 있는지 확인
java -version

# 없으면 설치 (Ubuntu/Debian)
sudo apt-get install default-jdk

# 그 후 pycocoevalcap 설치
pip install git+https://github.com/salaniz/pycocoevalcap.git
```

### NLTK 데이터 다운로드 오류
```bash
# 수동 다운로드
python -m nltk.downloader wordnet punkt

# 또는 Python에서
import nltk
nltk.download('wordnet')
nltk.download('punkt')
```

### 메모리 부족
```bash
# batch_size 조정 (eval.py 라인 950)
batch_size = 32  # 기본값 100에서 감소

# 또는 샘플 수 제한
python scripts/eval.py --csv-input data.csv --max-samples 1000
```

---

## 성능 비교

테스트 환경 (5,958 샘플):

| 메트릭 | 시간 | 메모리 | 정확도 |
|--------|------|--------|--------|
| BLEU-4 | ~2초 | ~50MB | 높음 |
| METEOR | ~30초 | ~100MB | 높음 |
| ROUGE-L | ~5분 | ~200MB | 높음 |
| SPICE | ~2분 | ~500MB | 매우 높음 |
| CIDEr | ~2분 | ~400MB | 매우 높음 |
| **합계** | ~**8분** | ~**1GB** | ✓ |

---

## 참고 자료

- BLEU: https://www.aclweb.org/anthology/P02-1040.pdf
- METEOR: https://www.aclweb.org/anthology/W07-0704.pdf
- ROUGE: https://aclanthology.org/W04-1013/
- SPICE: https://arxiv.org/abs/1602.05771
- CIDEr: https://arxiv.org/abs/1411.5726

---

## 최종 체크리스트

평가 실행 전 확인사항:

- [ ] sacrebleu 설치 (`pip list | grep sacrebleu`)
- [ ] NLTK 설치 (`pip list | grep nltk`)
- [ ] rouge-score 설치 (`pip list | grep rouge-score`)
- [ ] pycocoevalcap 설치 (`pip list | grep pycocoevalcap`)
- [ ] Java 설치 (선택사항, `java -version`)
- [ ] CSV 형식 확인 (prediction, reference 컬럼)
- [ ] 충분한 메모리 (권장: 2GB 이상)

모든 설정이 완료되었으면:
```bash
python scripts/eval.py --csv-input your_data.csv
```

✅ 모든 메트릭이 계산될 것입니다!
