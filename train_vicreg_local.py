#!/usr/bin/env python3
"""
VICReg-L 훈련 스크립트 예제
=========================

기존 train.py를 기반으로 VICReg-L을 지원하는 훈련 스크립트입니다.
"""

import json
import argparse
from pathlib import Path
from panovlm.config import ModelConfig
from panovlm.model import PanoramaVLM
from panovlm.processors.image import PanoramaImageProcessor

def load_config_with_vicreg_local(config_path: str) -> ModelConfig:
    """JSON 설정에서 VICReg-L 설정을 포함한 ModelConfig 생성"""
    
    with open(config_path, 'r') as f:
        json_config = json.load(f)
    
    # 기본 모델 설정
    model_config = ModelConfig(
        vision_name=json_config["models"]["vision_name"],
        language_model_name=json_config["models"]["lm_model"],
        resampler_type=json_config["models"]["resampler"],
        
        # 이미지 처리 설정
        crop_strategy=json_config["data"]["crop_strategy"],
        max_text_length=json_config["data"]["max_text_length"],
        overlap_ratio=json_config["data"]["overlap_ratio"],
        
        # VICReg 설정
        vicreg_loss_weight=json_config["training"]["vision"]["vicreg_loss_weight"],
        vicreg_overlap_ratio=json_config["data"]["overlap_ratio"],
        
        # VICReg-L 설정
        use_vicreg_local=json_config["training"]["vision"].get("use_vicreg_local", False),
        vicreg_local_weight=json_config["training"]["vision"].get("vicreg_local_weight", 0.5),
        vicreg_local_inv_weight=json_config["training"]["vision"].get("vicreg_local_inv_weight", 1.0),
        vicreg_local_var_weight=json_config["training"]["vision"].get("vicreg_local_var_weight", 1.0),
        vicreg_local_cov_weight=json_config["training"]["vision"].get("vicreg_local_cov_weight", 0.01),
        vicreg_local_inv_type=json_config["training"]["vision"].get("vicreg_local_inv_type", "l2"),
        vicreg_local_gamma=json_config["training"]["vision"].get("vicreg_local_gamma", 1.0)
    )
    
    return model_config

def create_data_loader_with_metadata(config_path: str, stage: str = "train"):
    """메타데이터를 포함한 데이터 로더 생성 예제"""
    
    with open(config_path, 'r') as f:
        json_config = json.load(f)
    
    # 이미지 프로세서 (메타데이터 반환 활성화)
    processor = PanoramaImageProcessor(
        crop_strategy=json_config["data"]["crop_strategy"],
        fov_deg=json_config["data"].get("fov_deg", 90.0),
        overlap_ratio=json_config["data"]["overlap_ratio"],
        image_size=tuple(json_config["data"]["image_size"])
    )
    
    print(f"✅ Image processor created: {processor.crop_strategy}")
    print(f"   Views per image: {processor.num_views}")
    
    # 실제 사용 시에는 여기서 PyTorch DataLoader를 반환
    return processor

def train_vision_stage_with_vicreg_local(config_path: str):
    """VICReg-L을 사용한 Vision 단계 훈련 예제"""
    
    print(f"🚀 Training with VICReg-L: {config_path}")
    
    # 1. 설정 로드
    model_config = load_config_with_vicreg_local(config_path)
    processor = create_data_loader_with_metadata(config_path)
    
    print(f"📋 Model Configuration:")
    print(f"   VICReg-L enabled: {model_config.use_vicreg_local}")
    if model_config.use_vicreg_local:
        print(f"   VICReg-L weight: {model_config.vicreg_local_weight}")
        print(f"   INV type: {model_config.vicreg_local_inv_type}")
    
    # 2. 모델 생성
    model = PanoramaVLM(config=model_config)
    
    print(f"✅ Model created:")
    print(f"   Vision encoder: {model_config.vision_name}")
    print(f"   VICReg-L active: {model.use_vicreg_local}")
    
    # 3. 훈련 루프 시뮬레이션
    import torch
    model.train()
    
    # 가상 배치 (실제로는 DataLoader에서)
    dummy_batch = {
        'pixel_values': torch.randn(4, 8, 3, 224, 224),  # [B, V, C, H, W]  
        'view_metadata': [
            {
                'yaw': float(i * 45), 
                'pitch': 0.0, 
                'effective_fov': 90.0,
                'view_index': i % 8
            } 
            for i in range(32)  # B * V = 4 * 8
        ]
    }
    
    # 배치를 모델 입력 형태로 변환
    pixel_values_flat = dummy_batch['pixel_values'].flatten(0, 1)  # [B*V, C, H, W]
    
    # Vision 단계 순전파
    outputs = model(
        pixel_values=pixel_values_flat,
        stage="vision",
        view_metadata=dummy_batch['view_metadata']
    )
    
    print(f"\n📊 Training Step Results:")
    print(f"   Total loss: {outputs['loss'].item():.6f}")
    print(f"   VICReg loss: {outputs['vicreg_loss'].item():.6f}")
    
    if 'vicreg_local_loss' in outputs:
        print(f"   VICReg-L loss: {outputs['vicreg_local_loss'].item():.6f}")
        print(f"   - INV: {outputs['vicreg_local_inv'].item():.6f}")  
        print(f"   - VAR: {outputs['vicreg_local_var'].item():.6f}")
        print(f"   - COV: {outputs['vicreg_local_cov'].item():.6f}")
        print(f"   - Valid pairs: {outputs['vicreg_local_pairs']}")
    
    # 백워드 패스
    outputs['loss'].backward()
    print(f"✅ Backward pass completed")
    
    return outputs

def main():
    parser = argparse.ArgumentParser(description="VICReg-L Training Example")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config_vicreg_local.json',
        choices=[
            'config.json',
            'config_mixed_transition.json', 
            'config_vicreg_local.json'
        ],
        help='Configuration file'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 VICReg-L Training Example")
    print("=" * 60)
    
    try:
        outputs = train_vision_stage_with_vicreg_local(args.config)
        
        print(f"\n🎉 Training simulation completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"   1. Integrate this into your existing training loop")
        print(f"   2. Monitor VICReg-L metrics in WandB/TensorBoard")
        print(f"   3. Adjust hyperparameters based on loss curves")
        print(f"   4. Compare with baseline VICReg results")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()