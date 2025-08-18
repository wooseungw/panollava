#!/usr/bin/env python3
# coding: utf-8
"""
ModelConfig JSON 파일 생성 도구

사용법:
    python create_config.py --output my_config.json
    python create_config.py --preset siglip_qwen25_small --output my_config.json
    python create_config.py --vision-model google/siglip-large-patch16-384 --output large_config.json
"""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="ModelConfig JSON 파일 생성 도구")
    
    # 출력 설정
    parser.add_argument('--output', '-o', required=True, help='생성할 JSON 파일 경로')
    parser.add_argument('--pretty', action='store_true', help='보기 좋게 포맷팅')
    
    # 사전 정의된 설정
    parser.add_argument('--preset', choices=['siglip_qwen25_small', 'siglip_qwen25_large', 'siglip_qwen25_lora'],
                       help='사전 정의된 설정 사용')
    
    # 모델 설정
    parser.add_argument('--vision-model', default='google/siglip-base-patch16-224',
                       help='Vision 모델 (기본: google/siglip-base-patch16-224)')
    parser.add_argument('--language-model', default='Qwen/Qwen2.5-0.5B-Instruct',
                       help='Language 모델 (기본: Qwen/Qwen2.5-0.5B-Instruct)')
    parser.add_argument('--resampler-type', default='mlp', choices=['mlp'],
                       help='Resampler 타입 (기본: mlp)')
    parser.add_argument('--latent-dimension', type=int, default=768,
                       help='Latent dimension (기본: 768)')
    
    # VICReg 설정
    parser.add_argument('--vicreg-loss-weight', type=float, default=0.0,
                       help='VICReg loss weight (기본: 1.0)')
    parser.add_argument('--vicreg-overlap-ratio', type=float, default=0.5,
                       help='VICReg overlap ratio (기본: 0.5)')
    
    # 텍스트 설정
    parser.add_argument('--max-text-length', type=int, default=512,
                       help='최대 텍스트 길이 (기본: 512)')
    
    # 이미지 설정
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224],
                       help='이미지 크기 [height width] (기본: 224 224)')
    parser.add_argument('--crop-strategy', default='e2p', 
                       choices=['e2p', 'grid', 'sliding_window', 'cubemap'],
                       help='크롭 전략 (기본: e2p)')
    parser.add_argument('--fov-deg', type=float, default=90.0,
                       help='Field of view 각도 (기본: 90.0)')
    parser.add_argument('--overlap-ratio', type=float, default=0.5,
                       help='이미지 겹침 비율 (기본: 0.5)')
    
    # LoRA 설정
    parser.add_argument('--use-lora', action='store_true',
                       help='LoRA 사용')
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank (기본: 16)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha (기본: 32)')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                       help='LoRA dropout (기본: 0.1)')
    
    # 메타데이터
    parser.add_argument('--description', default='',
                       help='설정 설명')
    
    args = parser.parse_args()
    
    # panovlm.config 모듈 import
    try:
        from panovlm.config import ModelConfig, get_preset_config
    except ImportError:
        print("❌ panovlm.config 모듈을 import할 수 없습니다.")
        print("현재 디렉토리가 프로젝트 루트인지 확인하고 PYTHONPATH를 설정해주세요.")
        return 1
    
    # 설정 생성
    if args.preset:
        print(f"📋 사전 정의된 설정 사용: {args.preset}")
        try:
            config = get_preset_config(args.preset)
            
            # 명령줄 인자로 오버라이드
            updates = {}
            if args.description:
                updates['description'] = args.description
            
            if updates:
                config = config.update(**updates)
                
        except Exception as e:
            print(f"❌ preset 로드 실패: {e}")
            return 1
    else:
        print("🔧 개별 파라미터로 설정 생성")
        
        # LoRA 타겟 모듈 설정
        lora_target_modules = None
        if args.use_lora:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        config = ModelConfig(
            vision_name=args.vision_model,
            language_model_name=args.language_model,
            resampler_type=args.resampler_type,
            latent_dimension=args.latent_dimension,
            
            vicreg_loss_weight=args.vicreg_loss_weight,
            vicreg_overlap_ratio=args.vicreg_overlap_ratio,
            
            max_text_length=args.max_text_length,
            
            image_size=tuple(args.image_size),
            crop_strategy=args.crop_strategy,
            fov_deg=args.fov_deg,
            overlap_ratio=args.overlap_ratio,
            
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=lora_target_modules,
            
            description=args.description or f"Generated config using {args.vision_model} + {args.language_model}"
        )
    
    # 유효성 검사
    if not config.validate():
        print("⚠️ 설정 유효성 검사에 실패했지만 저장을 진행합니다.")
    
    # 파일 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        config.save(output_path)
        print(f"✅ 설정 파일 저장 완료: {output_path}")
        
        # 설정 요약 출력
        print("\n📊 생성된 설정 요약:")
        print(f"   Vision Model: {config.vision_name}")
        print(f"   Language Model: {config.language_model_name}")
        print(f"   Latent Dimension: {config.latent_dimension}")
        print(f"   Image Size: {config.image_size}")
        print(f"   Max Text Length: {config.max_text_length}")
        if config.use_lora:
            print(f"   LoRA: Enabled (r={config.lora_r}, alpha={config.lora_alpha})")
        else:
            print(f"   LoRA: Disabled")
        
        print(f"\n🚀 사용법:")
        print(f"   python train.py --config {output_path} --stage vision")
        print(f"   python eval.py --config {output_path} --ckpt your_checkpoint.ckpt")
        
    except Exception as e:
        print(f"❌ 설정 파일 저장 실패: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())