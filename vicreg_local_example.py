#!/usr/bin/env python3
"""
VICReg-L (Local VICReg) 사용 예제
=====================================

기존 코드에 최소한의 변경으로 VICReg-L을 적용하는 방법을 보여줍니다.
"""

import torch
from panovlm.config import ModelConfig
from panovlm.model import PanoramaVLM
from panovlm.processors.image import PanoramaImageProcessor

def enable_vicreg_local_example():
    """VICReg-L을 활성화하는 예제"""
    
    # 1. ModelConfig에서 VICReg-L 활성화
    config = ModelConfig(
        # 기본 모델 설정
        vision_name="google/siglip-base-patch16-224",
        language_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        resampler_type="mlp",
        latent_dimension=768,
        
        # 기존 VICReg 설정
        vicreg_loss_weight=1.0,
        vicreg_overlap_ratio=0.5,
        
        # VICReg-L 활성화 및 설정
        use_vicreg_local=True,           # 🔥 VICReg-L 활성화
        vicreg_local_weight=0.5,         # 전체 손실에서 VICReg-L 가중치
        vicreg_local_inv_weight=1.0,     # INV 손실 가중치
        vicreg_local_var_weight=1.0,     # VAR 손실 가중치  
        vicreg_local_cov_weight=0.01,    # COV 손실 가중치 (보통 작게)
        vicreg_local_inv_type="l2",      # "l2" 또는 "cos"
        vicreg_local_gamma=1.0,          # 분산 정규화 목표
        
        # 이미지 처리 설정 (메타데이터 생성을 위해)
        crop_strategy="e2p",             # e2p 전략이 메타데이터 지원
        fov_deg=90.0
    )
    
    # 2. 모델 생성
    model = PanoramaVLM(config=config)
    print(f"✅ VICReg-L enabled: {model.use_vicreg_local}")
    print(f"   Local weight: {model.vicreg_local_weight}")
    
    # 3. 이미지 프로세서 (메타데이터 반환 기능 포함)
    processor = PanoramaImageProcessor(
        crop_strategy="e2p", 
        fov_deg=90.0,
        overlap_ratio=0.5
    )
    
    return model, processor, config

def training_example():
    """훈련에서 VICReg-L 사용 예제"""
    model, processor, config = enable_vicreg_local_example()
    
    # 가상 파노라마 이미지 (실제로는 PIL Image 또는 경로)
    dummy_pano = torch.randn(3, 512, 1024)  # [C, H, W] ERP 이미지
    
    # 이미지 처리 (메타데이터 포함)
    pixel_values, view_metadata = processor(dummy_pano, return_metadata=True)
    
    print(f"📷 Generated {len(view_metadata)} views")
    print(f"   First view metadata: {view_metadata[0] if view_metadata else 'None'}")
    
    # 배치 준비
    batch_size = 2
    pixel_values_batch = pixel_values.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    # [B, V, C, H, W] -> [B*V, C, H, W]
    pixel_values_flat = pixel_values_batch.flatten(0, 1)
    
    # 메타데이터도 배치로 확장
    metadata_batch = []
    for b in range(batch_size):
        metadata_batch.extend(view_metadata)
    
    # VICReg-L이 포함된 vision 단계 훈련
    model.train()
    outputs = model(
        pixel_values=pixel_values_flat,
        stage="vision",
        view_metadata=metadata_batch  # 🔥 메타데이터 전달
    )
    
    print("\n🎯 Training outputs:")
    print(f"   Total loss: {outputs['loss'].item():.6f}")
    print(f"   VICReg loss: {outputs['vicreg_loss'].item():.6f}")
    
    if 'vicreg_local_loss' in outputs:
        print(f"   VICReg-L loss: {outputs['vicreg_local_loss'].item():.6f}")
        print(f"   VICReg-L pairs: {outputs['vicreg_local_pairs']}")
        print(f"   VICReg-L INV: {outputs['vicreg_local_inv'].item():.6f}")
        print(f"   VICReg-L VAR: {outputs['vicreg_local_var'].item():.6f}")
        print(f"   VICReg-L COV: {outputs['vicreg_local_cov'].item():.6f}")
    else:
        print("   No VICReg-L loss (check metadata or configuration)")

def gradual_migration_example():
    """점진적 마이그레이션 예제: 기존 -> VICReg-L"""
    
    print("=== 단계 1: 기존 VICReg만 사용 ===")
    config_old = ModelConfig(use_vicreg_local=False)
    model_old = PanoramaVLM(config=config_old)
    print(f"Old model VICReg-L: {model_old.use_vicreg_local}")
    
    print("\n=== 단계 2: VICReg + VICReg-L 혼합 (낮은 가중치) ===")  
    config_mixed = ModelConfig(
        use_vicreg_local=True,
        vicreg_local_weight=0.1  # 낮은 가중치로 시작
    )
    model_mixed = PanoramaVLM(config=config_mixed)
    print(f"Mixed model VICReg-L weight: {model_mixed.vicreg_local_weight}")
    
    print("\n=== 단계 3: VICReg-L 주도 (높은 가중치) ===")
    config_new = ModelConfig(
        use_vicreg_local=True, 
        vicreg_local_weight=1.0,  # 높은 가중치
        vicreg_loss_weight=0.5    # 기존 VICReg 가중치 감소
    )
    model_new = PanoramaVLM(config=config_new)
    print(f"New model VICReg-L weight: {model_new.vicreg_local_weight}")
    print(f"New model VICReg weight: {model_new.vicreg_loss_weight}")

if __name__ == "__main__":
    print("🚀 VICReg-L Integration Example")
    print("=" * 50)
    
    try:
        print("\n📖 Example 1: Basic VICReg-L Training")
        training_example()
        
        print("\n📖 Example 2: Gradual Migration Strategy")
        gradual_migration_example()
        
        print(f"\n✅ All examples completed successfully!")
        print("\n💡 Key points:")
        print("   - Set use_vicreg_local=True in ModelConfig")
        print("   - Use return_metadata=True in image processor")  
        print("   - Pass view_metadata to model.forward()")
        print("   - Monitor vicreg_local_* metrics in training logs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()