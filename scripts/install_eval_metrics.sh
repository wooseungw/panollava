#!/bin/bash

# =============================================================================
# PanoLLaVA 평가 메트릭 의존성 설치 스크립트
# =============================================================================

echo "🔧 PanoLLaVA 평가 메트릭 의존성 설치 중..."

# 기본 평가 메트릭 패키지
echo "📦 기본 평가 메트릭 패키지 설치..."
pip install nltk rouge-score

# NLTK 데이터 다운로드
echo "📥 NLTK 데이터 다운로드..."
python -c "
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)
print('✓ NLTK 데이터 다운로드 완료')
"

# SPICE를 위한 pycocoevalcap 설치
echo "📦 SPICE 메트릭을 위한 pycocoevalcap 설치..."
pip install git+https://github.com/salaniz/pycocoevalcap.git

# CLIP 설치
echo "📦 CLIP 모델 설치..."
pip install git+https://github.com/openai/CLIP.git

# Java 설치 확인 (SPICE가 Java를 필요로 함)
echo "☕ Java 설치 확인..."
if command -v java &> /dev/null; then
    java_version=$(java -version 2>&1 | head -n 1)
    echo "✓ Java 설치됨: $java_version"
else
    echo "⚠️  Java가 설치되지 않았습니다."
    echo "SPICE 메트릭을 사용하려면 Java를 설치하세요:"
    echo "Ubuntu/Debian: sudo apt-get install openjdk-11-jdk"
    echo "CentOS/RHEL: sudo yum install java-11-openjdk-devel"
    echo "macOS: brew install openjdk@11"
fi

echo ""
echo "✅ 의존성 설치 완료!"
echo ""
echo "📊 사용 가능한 평가 메트릭:"
echo "   • BLEU (1, 2, 3, 4)"
echo "   • ROUGE (1, 2, L)" 
echo "   • METEOR"
echo "   • SPICE (Java 필요)"
echo "   • CLIP Score"
echo "   • CLIP-S (Image-Text Similarity)"
echo "   • RefCLIP-S (Reference-Prediction Similarity)"
echo ""
echo "🚀 이제 eval.py를 실행하여 모든 메트릭을 사용할 수 있습니다!"
