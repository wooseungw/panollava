#!/usr/bin/env python3
# coding: utf-8
"""
PanoramaVLM 간편 추론 스크립트
===========================

통합 인터페이스를 사용한 간단한 추론 예시입니다.
모델은 safetensors 기반 HF 디렉토리에서 로드합니다.

사용법:
    python simple_inference.py --image panorama.jpg --model-dir runs/<run_name>/hf_model
"""

import argparse
import torch
import json
from PIL import Image
from pathlib import Path

def load_global_config():
    """Load global configuration from config.json"""
    config_path = Path("config.json")
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")
    return {}

def main():
    # Load global configuration
    global_config = load_global_config()
    model_config = global_config.get("models", {})
    
    parser = argparse.ArgumentParser(description="PanoramaVLM 간편 추론")
    parser.add_argument("--image", required=True, help="파노라마 이미지 경로")
    parser.add_argument("--model-dir", default=None, help="HF-style 모델 디렉토리 (hf_model 또는 panorama_model)")
    parser.add_argument("--prompt", default="Describe this panoramic image in detail.", 
                       help="입력 프롬프트")
    parser.add_argument("--max-tokens", type=int, default=128, help="최대 생성 토큰 수")
    parser.add_argument("--temperature", type=float, default=0.7, help="생성 온도")
    parser.add_argument("--device", default="auto", help="디바이스 (auto, cuda, cpu)")
    parser.add_argument("--config", help="ModelConfig JSON 파일 경로")
    
    args = parser.parse_args()
    
    # 1. 모델 로딩
    try:
        from panovlm.model import PanoramaVLM
        # 모델 디렉토리 결정: 인자 > config.paths.pretrained_dir
        model_dir = args.model_dir or global_config.get("paths", {}).get("pretrained_dir")
        if not model_dir:
            raise ValueError("--model-dir 또는 config.paths.pretrained_dir 를 지정하세요")
        print(f"📂 모델 디렉토리 로딩: {model_dir}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
        model = PanoramaVLM.from_pretrained_dir(model_dir, device=str(device))
        print("✅ 모델 로딩 완료")
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # 2. 이미지 로딩 및 전처리
    print(f"🖼️  이미지 로딩: {args.image}")
    try:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {args.image}")
        
        from panovlm.processors.image import PanoramaImageProcessor
        image_processor = PanoramaImageProcessor(
            image_size=(224, 224),  # 모델에 맞는 이미지 크기
            crop_strategy="e2p",  # MLP 리샘플러 사용
            fov_deg=90,
            overlap_ratio=0.5  # 오버랩 비율
        )
        
        # PanoramaImageProcessor의 __call__ 메서드 사용 (올바른 방법)
        image = Image.open(image_path).convert("RGB")
        print(f"   - 원본 이미지 크기: {image.size}")
        
        pixel_values = image_processor(image)  # __call__ 메서드 사용
        print(f"   - 전처리 완료: {pixel_values.shape}")
        print(f"   - 뷰 수: {pixel_values.shape[0]}, 크롭 전략: {image_processor.crop_strategy}")
        
    except Exception as e:
        print(f"❌ 이미지 처리 실패: {e}")
        return
    
    # 3. 텍스트 입력 준비
    print(f"💬 프롬프트: {args.prompt}")
    try:
        # 토크나이저 설정
        # 모델의 토크나이저 사용
        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is None:
            raise RuntimeError("Model tokenizer is not available")
        
        # 토크나이징
        inputs = tokenizer(
            args.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        print(f"   - 토큰화 완료: {input_ids.shape}")
        
    except Exception as e:
        print(f"❌ 텍스트 처리 실패: {e}")
        return
    
    # 4. 추론 실행
    print(f"🤖 추론 실행 중...")
    try:
        with torch.no_grad():
            # 입력을 모델과 같은 디바이스로 이동
            # 모델이 nn.Module 래퍼가 아닐 수도 있으므로 device는 입력 텐서 기준으로 설정
            device = torch.device(model.language_model.device) if hasattr(model, 'language_model') else device
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # 생성 실행
            output = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        print(f"✅ 추론 완료!")
        
    except Exception as e:
        print(f"❌ 추론 실패: {e}")
        return
    
    # 5. 결과 출력
    print("=" * 50)
    print("🎯 결과")
    print("=" * 50)
    
    try:
        if isinstance(output, dict) and "text" in output:
            generated_text = output["text"][0]
        elif isinstance(output, torch.Tensor):
            # 토큰 ID에서 텍스트로 변환
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            generated_text = str(output)
        
        print(f"📝 생성된 텍스트:")
        print(f"   {generated_text}")
        
        # 추가 정보
        print(f"\n📊 생성 정보:")
        print(f"   - 입력 이미지: {args.image}")
        print(f"   - 프롬프트: {args.prompt}")
        print(f"   - 최대 토큰: {args.max_tokens}")
        print(f"   - 온도: {args.temperature}")
        print(f"   - 디바이스: {device}")
        
        # LoRA 정보 (있다면)
        lora_info = model.get_lora_info()
        if lora_info.get("is_lora_enabled", False):
            print(f"   - LoRA: Rank {lora_info.get('lora_r')}, Alpha {lora_info.get('lora_alpha')}")
        else:
            print(f"   - LoRA: 비활성화")
        
    except Exception as e:
        print(f"❌ 결과 처리 실패: {e}")
        return
    
    print("=" * 50)
    print("🎉 추론 완료!")


if __name__ == "__main__":
    main()
