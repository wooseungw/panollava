#!/usr/bin/env python3
"""
수정된 Generate 메서드 테스트
"""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from panovlm.model import PanoramaVLM

def test_fixed_generate():
    """수정된 generate 메서드 테스트"""
    print("=== 수정된 Generate 메서드 테스트 ===")
    
    # 모델 초기화
    model = PanoramaVLM(
        vision_model_name="google/siglip-base-patch16-224",
        language_model_name="Qwen/Qwen2.5-0.5B",
        resampler_type="mlp"
    )
    
    print("🔧 주요 개선사항:")
    improvements = [
        "✅ 강제 프롬프트 추가로 빈 문자열 방지",
        "✅ 배치 차원 일관성 확보",
        "✅ 안전한 예외 처리 및 fallback",
        "✅ 최소 길이 보장 (min_length)",
        "✅ 조기 종료 방지 (early_stopping=False)",
        "✅ 개선된 생성 파라미터"
    ]
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    # 다양한 배치 크기 테스트
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        print(f"\n--- 배치 크기 {batch_size} 테스트 ---")
        
        # 더미 파노라마 이미지
        dummy_pixel_values = torch.randn(batch_size, 6, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            try:
                # 1. 기본 캡셔닝 테스트
                print("1. 기본 캡셔닝:")
                result1 = model.generate(
                    pixel_values=dummy_pixel_values,
                    max_new_tokens=20,
                    temperature=0.7
                )
                
                for i, text in enumerate(result1['text']):
                    print(f"   샘플 {i}: '{text}' (길이: {len(text)})")
                
                # 빈 문자열 체크
                empty_count = sum(1 for text in result1['text'] if len(text) == 0)
                if empty_count == 0:
                    print("   ✅ 빈 문자열 없음")
                else:
                    print(f"   ⚠️  빈 문자열 {empty_count}개 발견")
                
                # 2. 질문-답변 테스트
                print("\n2. 질문-답변:")
                question = "What can you see?"
                question_tokens = model.tokenizer(question, return_tensors="pt")
                
                # 배치 크기에 맞게 확장
                if question_tokens["input_ids"].size(0) != batch_size:
                    question_tokens["input_ids"] = question_tokens["input_ids"].repeat(batch_size, 1)
                
                result2 = model.generate(
                    pixel_values=dummy_pixel_values,
                    input_ids=question_tokens["input_ids"],
                    max_new_tokens=25,
                    temperature=0.8
                )
                
                for i, text in enumerate(result2['text']):
                    print(f"   샘플 {i}: '{text}' (길이: {len(text)})")
                
                print(f"   ✅ 배치 크기 {batch_size} 테스트 성공!")
                
            except Exception as e:
                print(f"   ❌ 배치 크기 {batch_size} 실패: {e}")
                import traceback
                traceback.print_exc()

def test_edge_cases():
    """엣지 케이스 테스트"""
    print("\n=== 엣지 케이스 테스트 ===")
    
    model = PanoramaVLM(
        vision_model_name="google/siglip-base-patch16-224",
        language_model_name="Qwen/Qwen2.5-0.5B",
        resampler_type="mlp"
    )
    
    model.eval()
    
    # 1. 4D 이미지 입력 테스트
    print("1. 4D 이미지 입력 테스트:")
    dummy_4d = torch.randn(2, 3, 224, 224)  # 배치, 채널, 높이, 너비
    
    with torch.no_grad():
        try:
            result = model.generate(
                pixel_values=dummy_4d,
                max_new_tokens=15,
                temperature=0.6
            )
            print(f"   ✅ 4D 입력 성공: {result['text'][0]}")
        except Exception as e:
            print(f"   ❌ 4D 입력 실패: {e}")
    
    # 2. 매우 작은 max_new_tokens 테스트
    print("\n2. 매우 작은 max_new_tokens 테스트:")
    dummy_5d = torch.randn(1, 6, 3, 224, 224)
    
    with torch.no_grad():
        try:
            result = model.generate(
                pixel_values=dummy_5d,
                max_new_tokens=3,
                temperature=0.5
            )
            print(f"   ✅ 작은 토큰 수 성공: '{result['text'][0]}'")
        except Exception as e:
            print(f"   ❌ 작은 토큰 수 실패: {e}")
    
    # 3. 매우 낮은/높은 온도 테스트
    print("\n3. 극단적 온도 테스트:")
    temperatures = [0.1, 1.5, 2.0]
    
    for temp in temperatures:
        with torch.no_grad():
            try:
                result = model.generate(
                    pixel_values=dummy_5d,
                    max_new_tokens=10,
                    temperature=temp
                )
                print(f"   ✅ 온도 {temp}: '{result['text'][0]}'")
            except Exception as e:
                print(f"   ❌ 온도 {temp} 실패: {e}")


if __name__ == "__main__":
    test_fixed_generate()
    test_edge_cases()