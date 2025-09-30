# Instruction Tuning 설정 가이드
## 개요
**3단계 (Fine-tuning)에서 파노라마 이미지 쿼리에 대해 1~2문장으로 답변하도록 instruction tuning을 수행합니다.**

일반적인 VLM 학습 관행에 따라 fine-tuning 단계에서 instruction tuning을 통합하여 수행합니다.

## 설정된 기능들
1. **Fine-tuning 단계에 Instruction Tuning 통합**: 일반적인 VLM 관행에 따라 fine-tuning 단계에서 instruction tuning 수행
2. **짧은 응답 유도**: 1-2문장으로 답변하도록 system message와 프롬프트 설정
3. **Instruction-Response 데이터셋**: `data/quic360/train_instruction.csv` 사용
4. **LoRA 지원**: Fine-tuning 단계에서 LoRA 적용 가능

## 사용법
### 1. 데이터셋 준비
- \`data/quic360/train_instruction.csv\` 파일이 자동 생성됨
- 형식: url,instruction,response

### 2. 학습 실행
\`\`\`bash
# Fine-tuning 단계에서 instruction tuning 수행 (기본)
python scripts/train.py --config configs/default.yaml --stage finetune

# 전체 파이프라인 실행 (vision → resampler → finetune)
python scripts/train.py --config configs/default.yaml
\`\`\`

### 3. 설정 파라미터
- **Learning Rate**: 1e-5 (Instruction tuning에 적합)
- **Epochs**: 3 (충분한 학습을 위해)
- **Max Text Length**: 256 (짧은 응답에 적합)
- **Batch Size**: 2 (메모리 효율적)
- **System Message**: "1-2문장으로 답변하도록 유도"

## 일반적인 VLM 학습 관행
대부분의 최신 VLM 모델들은 다음과 같은 단계로 학습됩니다:

1. **Vision Pre-training**: Vision backbone 학습
2. **Resampler Training**: Vision-to-language projection 학습  
3. **Fine-tuning (with Instruction Tuning)**: End-to-end 학습 + Instruction following 능력 향상
4. **(선택적) 추가 Instruction Tuning**: 특정 도메인 특화

## 기대 효과
- 파노라마 이미지 쿼리에 대해 **1-2문장**으로 **간결하고 정확한** 답변 생성
- 기존의 장황한 응답 대신 **핵심 내용**만 전달
- **Instruction following** 능력 향상
- **메모리 효율적**인 학습 (별도의 단계 불필요)

