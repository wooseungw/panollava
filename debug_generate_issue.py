#!/usr/bin/env python3
"""
Generate 메서드 문제 진단 및 해결
"""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def analyze_generate_problems():
    """Generate 메서드 문제 분석"""
    print("=== Generate 메서드 문제 분석 ===")
    
    problems = [
        {
            "문제": "빈 문자열 반환",
            "원인": [
                "1. 언어 모델이 EOS 토큰을 즉시 생성",
                "2. 비전 토큰이 언어 모델에 제대로 전달되지 않음",
                "3. 생성 파라미터가 부적절함",
                "4. 모델이 학습되지 않음"
            ],
            "해결책": [
                "1. 강제 프롬프트 추가",
                "2. 비전 토큰 처리 개선",
                "3. 생성 파라미터 조정",
                "4. EOS 토큰 억제"
            ]
        },
        {
            "문제": "배치 크기 불일치",
            "원인": [
                "1. 비전 토큰: (batch*views, seq, dim)",
                "2. 텍스트 토큰: (batch, seq, dim)",
                "3. 차원 reshape 문제",
                "4. 배치 처리 로직 오류"
            ],
            "해결책": [
                "1. 일관된 배치 크기 처리",
                "2. 명시적 reshape",
                "3. 배치 크기 검증",
                "4. 안전한 차원 변환"
            ]
        }
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{i}. {problem['문제']}")
        print("   원인:")
        for cause in problem['원인']:
            print(f"     {cause}")
        print("   해결책:")
        for solution in problem['해결책']:
            print(f"     {solution}")

def test_current_generate():
    """현재 generate 메서드 테스트"""
    print("\n=== 현재 Generate 메서드 테스트 ===")
    
    try:
        from panovlm.model import PanoramaVLM
        
        model = PanoramaVLM(
            vision_model_name="google/siglip-base-patch16-224",
            language_model_name="Qwen/Qwen2.5-0.5B",
            resampler_type="mlp"
        )
        
        # 테스트 데이터
        dummy_pixel_values = torch.randn(1, 6, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            print("1. 기본 생성 테스트...")
            result = model.generate(
                pixel_values=dummy_pixel_values,
                max_new_tokens=10,
                temperature=0.7
            )
            
            print(f"   생성된 텍스트: '{result['text'][0]}'")
            print(f"   텍스트 길이: {len(result['text'][0])}")
            print(f"   토큰 개수: {result['generated_ids'].shape}")
            
            if len(result['text'][0]) == 0:
                print("   ❌ 빈 문자열 문제 확인됨")
            else:
                print("   ✅ 텍스트 생성 성공")
                
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_generate_problems()
    test_current_generate()