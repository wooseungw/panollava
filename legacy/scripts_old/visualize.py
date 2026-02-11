#!/usr/bin/env python3
"""
학습된 PanoLLaVA 모델의 Vision Encoder 시각화 스크립트

학습된 체크포인트에서 vision encoder를 추출하고,
DINOv2 스타일의 PCA 기반 시각화를 수행합니다.

사용법:
    # 기본 사용 (체크포인트에서 vision encoder 추출)
    python scripts/visualize_trained_model.py \
        --checkpoint runs/SQ3_1_latent768_PE_e2p_vision_mlp/last.ckpt \
        --image data/quic360/test/pano_image.jpg \
        --output_dir results/trained_vision_viz

    # 설정 파일과 함께 사용
    python script    # 출력 디렉토리 설정
    if args.output_dir is None:
        args.output_dir = f"visualizations/{checkpoint_name}"

    print("="*70)ualize_trained_model.py \
        --checkpoint runs/SQ3_1_latent768_PE_e2p_vision_mlp/last.ckpt \
        --config configs/default.yaml \
        --image data/quic360/test/pano_image.jpg \
        --crop_strategy anyres_e2p
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import legacy config directly from the module file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "panovlm_config_legacy",
    str(Path(__file__).parent.parent / "src" / "panovlm" / "config.py")
)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
PanoVLMConfig = config_module.PanoVLMConfig

from panovlm.models.model import PanoramaVLM
from panovlm.evaluation.dino import DinoVisualizer
from panovlm.processors.image import PanoramaImageProcessor


def load_trained_model(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    학습된 PanoLLaVA 체크포인트를 로드합니다.
    PanoramaVLM.from_checkpoint 메서드를 사용합니다.

    Args:
        checkpoint_path: 체크포인트 파일 경로
        device: 사용할 디바이스

    Returns:
        model: 로드된 PanoramaVLM 모델
        model_config: 모델 설정
    """
    print(f"📦 체크포인트 로딩: {checkpoint_path}")

    # PanoramaVLM.from_checkpoint 사용 (LoRA 자동 감지 포함)
    model = PanoramaVLM.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        auto_detect_lora=True,
        strict_loading=False
    )

    # 모델 설정 가져오기
    model_config = model.config

    print(f"✅ 모델 로드 완료!")
    print(f"   Vision: {model_config.vision_name}")
    print(f"   Language: {model_config.language_model_name}")
    print(f"   Resampler: {model_config.resampler_type}")

    model.eval()

    return model, model_config


def extract_vision_features_from_model(
    model: PanoramaVLM,
    image_path: str,
    crop_strategy: str = "e2p",
    image_size: int = 224,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    학습된 모델의 vision encoder에서 hidden states를 추출합니다.

    Args:
        model: PanoramaVLM 모델
        image_path: 입력 이미지 경로
        crop_strategy: 이미지 crop 전략
        image_size: 입력 이미지 크기
        device: 사용할 디바이스

    Returns:
        vision_features: Vision encoder의 features (numpy array)
        resampled_features: Resampler를 거친 features (numpy array)
        pixel_values: 전처리된 이미지 텐서
        view_metadata: View 메타데이터
    """
    print(f"\n📸 이미지 로딩: {image_path}")
    image = Image.open(image_path).convert("RGB")

    print(f"🖼️  이미지 전처리 (strategy: {crop_strategy})")
    
    # Panorama image processor 설정
    pano_processor = PanoramaImageProcessor(
        crop_strategy=crop_strategy,
        image_size=(image_size, image_size),
        fov_deg=90,
        overlap_ratio=0.5,
        normalize=True,
        use_vision_processor=True,
        vision_model_name=model.config.vision_name if hasattr(model, 'config') and hasattr(model.config, 'vision_name') else None
    )
    
    # 이미지 전처리
    pixel_values = pano_processor(image)
    
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)  # [V, C, H, W]
    
    num_views = pixel_values.shape[0]
    print(f"✅ {num_views}개 view 생성됨")
    
    # View metadata 추출
    view_metadata = pano_processor.view_metadata if hasattr(pano_processor, 'view_metadata') else None
    
    # anyres_e2p의 경우 0번째 view(global view) 제거
    skip_first_view = (crop_strategy == "anyres_e2p")
    if skip_first_view and num_views > 1:
        print(f"⚠️  anyres_e2p 전략: 0번째 global view 제외 ({num_views} → {num_views-1} views)")
        pixel_values = pixel_values[1:]  # 0번째 제거
        num_views = pixel_values.shape[0]
    
    # 배치 차원 추가
    img_batch = pixel_values.unsqueeze(0).to(device)  # [1, V, C, H, W]
    
    # Vision encoder와 Resampler를 통한 features 추출
    print("🧠 Vision encoder와 Resampler에서 features 추출 중...")
    model.eval()
    
    with torch.no_grad():
        # 1. Vision encoder 처리
        vision_result = model._process_vision_encoder(img_batch)
        vision_features = vision_result["vision_features"]  # [B*V, S, D_vision]
        batch_size = vision_result["batch_size"]
        num_views_from_model = vision_result["num_views"]
        
        print(f"  Vision features shape: {vision_features.shape}")
        
        # 2. Resampler 처리
        resampled_features = None
        if hasattr(model, '_process_resampler') and model.resampler is not None:
            resampled_features = model._process_resampler(
                vision_features, batch_size, num_views_from_model
            )  # [B*V, S, D_latent]
            print(f"  Resampled features shape: {resampled_features.shape}")
        
        # Numpy로 변환
        vision_features_np = vision_features.cpu().detach().numpy()
        resampled_features_np = resampled_features.cpu().detach().numpy() if resampled_features is not None else None
    
    return vision_features_np, resampled_features_np, pixel_values, view_metadata


def visualize_features(
    vision_features,
    resampled_features,
    pixel_values,
    output_dir: str,
    n_components: int = 3,
    remove_cls_token: bool = False,
    bg_removal_method: str = "threshold",
    analyze_similarity: bool = True,
    view_metadata = None,
    checkpoint_name: str = ""
):
    """
    추출된 vision features를 시각화하고 분석합니다.
    
    Args:
        vision_features: Vision encoder의 features (numpy array, shape: [V, S, D])
        resampled_features: Resampler를 거친 features (numpy array, shape: [V, S, D])
        pixel_values: 원본 전처리된 이미지
        output_dir: 출력 디렉토리
        n_components: PCA 주성분 개수
        remove_cls_token: CLS 토큰 제거 여부
        bg_removal_method: 배경 제거 방법
        analyze_similarity: 유사도 분석 수행 여부
        view_metadata: View 메타데이터
        checkpoint_name: 체크포인트 이름 (시각화 제목용)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_views = vision_features.shape[0] if len(vision_features.shape) == 3 else len(vision_features)
    
    print("\n" + "="*70)
    print("🎨 Vision Encoder Features 시각화")
    print("="*70)
    
    # 1. 원본 이미지 저장
    print(f"\n📷 [1/4] 원본 View 이미지 저장")
    fig, axes = plt.subplots(1, num_views, figsize=(4 * num_views, 4))
    if num_views == 1:
        axes = [axes]
    
    for i in range(num_views):
        img = pixel_values[i].permute(1, 2, 0).cpu().numpy()
        # Denormalize
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'View {i+1}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle(f'Input Views - {checkpoint_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    orig_save_path = output_path / f"original_views{checkpoint_name}.png"
    plt.savefig(orig_save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장: {orig_save_path}")
    plt.close()
    
    # 2. Vision Encoder Features 시각화
    print(f"\n📊 [2/4] Vision Encoder Features PCA 시각화")

    # SigLIP은 CLS 토큰이 없으므로 제거하지 않음
    # (196 patches = 14x14로 정사각형을 유지해야 함)
    vision_visualizer = DinoVisualizer(
        hidden_states_list=vision_features,
        remove_cls_token=False  # SigLIP: CLS 토큰 없음
    )
    
    print(f"   PCA 학습 (n_components={n_components}, bg_removal={bg_removal_method})")
    vision_visualizer.fit_pca(
        n_components=n_components,
        use_background_removal=True if bg_removal_method != "none" else False,
        bg_removal_method=bg_removal_method if bg_removal_method != "none" else None,
        use_global_scaling=True
    )
    
    titles = [f'View {i+1}' for i in range(num_views)]
    save_path = output_path / f"vision_encoder_pca{checkpoint_name}.png"
    vision_visualizer.plot_pca_results(
        titles=titles,
        save_path=str(save_path),
        figsize=(4 * num_views, 5)
    )
    print(f"   ✅ 저장: {save_path}")
    
    # 3. Resampled Features 시각화 (있는 경우)
    resampler_visualizer = None
    if resampled_features is not None:
        print(f"\n📊 [3/4] Resampled Features PCA 시각화")
        resampler_visualizer = DinoVisualizer(
            hidden_states_list=resampled_features,
            remove_cls_token=False  # Resampler output은 보통 CLS token이 없음
        )
        
        resampler_visualizer.fit_pca(
            n_components=n_components,
            use_background_removal=True if bg_removal_method != "none" else False,
            bg_removal_method=bg_removal_method if bg_removal_method != "none" else None,
            use_global_scaling=True
        )
        
        save_path_resampled = output_path / f"resampled_features_pca{checkpoint_name}.png"
        resampler_visualizer.plot_pca_results(
            titles=titles,
            save_path=str(save_path_resampled),
            figsize=(4 * num_views, 5)
        )
        print(f"   ✅ 저장: {save_path_resampled}")
        
        # Resampler Dashboard 생성
        if hasattr(resampler_visualizer, 'create_comprehensive_dashboard'):
            resampler_dashboard_path = output_path / f"resampler_dashboard{checkpoint_name}.png"
            try:
                resampler_visualizer.create_comprehensive_dashboard(
                    titles=f"Resampler - {checkpoint_name}",
                    view_metadata=view_metadata,
                    save_path=str(resampler_dashboard_path)
                )
                print(f"   ✅ Dashboard 저장: {resampler_dashboard_path}")
            except Exception as e:
                print(f"   ⚠️  Resampler Dashboard 생성 실패: {e}")
    else:
        print(f"\n⏭️  [3/4] Resampled Features 없음 (건너뛰기)")
    
    # 4. Comprehensive Dashboard 생성 (Vision Encoder)
    print(f"\n📊 [4/4] Vision Encoder Dashboard 생성")
    if hasattr(vision_visualizer, 'create_comprehensive_dashboard'):
        dashboard_path = output_path / f"vision_encoder_dashboard{checkpoint_name}.png"
        try:
            vision_visualizer.create_comprehensive_dashboard(
                titles=f"Vision Encoder - {checkpoint_name}",
                view_metadata=view_metadata,
                save_path=str(dashboard_path)
            )
            print(f"   ✅ 저장: {dashboard_path}")
        except Exception as e:
            print(f"   ⚠️  Dashboard 생성 실패: {e}")
    
    # 5. 유사도 분석
    if analyze_similarity and num_views > 1:
        print("\n" + "="*70)
        print("📈 유사도 분석")
        print("="*70)
        
        # 인접 view 간 유사도 분석
        pairs = [(i, (i + 1) % num_views) for i in range(num_views)]
        
        # Vision Encoder 유사도
        print("\n🔍 Vision Encoder 유사도:")
        try:
            vision_sim = vision_visualizer.get_hidden_similarity(
                pairs=pairs,
                view_metadata=view_metadata
            )
        except Exception as e:
            print(f"   ⚠️  유사도 계산 실패: {e}")
            vision_sim = {}
        
        # Resampler 유사도
        resampler_sim = {}
        if resampler_visualizer is not None:
            print("\n🔍 Resampled Features 유사도:")
            try:
                resampler_sim = resampler_visualizer.get_hidden_similarity(
                    pairs=pairs,
                    view_metadata=view_metadata
                )
            except Exception as e:
                print(f"   ⚠️  유사도 계산 실패: {e}")
        
        # 유사도 결과 저장
        similarity_file = output_path / f"similarity_analysis{checkpoint_name}.txt"
        with open(similarity_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"학습된 모델 Vision Encoder 유사도 분석\n")
            f.write(f"Checkpoint: {checkpoint_name}\n")
            f.write("=" * 70 + "\n\n")
            
            if vision_sim:
                f.write("=" * 70 + "\n")
                f.write("Vision Encoder Features 유사도\n")
                f.write("=" * 70 + "\n")
                for metric, values in vision_sim.items():
                    if values is None or (isinstance(values, list) and all(v is None for v in values)):
                        continue
                    f.write(f"\n{metric}:\n")
                    if isinstance(values, list):
                        for pair, val in zip(pairs, values):
                            if val is not None:
                                f.write(f"  View {pair[0]+1} ↔ View {pair[1]+1}: {val:.4f}\n")
                        valid_values = [v for v in values if v is not None]
                        if valid_values:
                            f.write(f"  평균: {np.mean(valid_values):.4f} ± {np.std(valid_values):.4f}\n")
                    else:
                        f.write(f"  값: {values:.4f}\n")
            
            if resampler_sim:
                f.write("\n" + "=" * 70 + "\n")
                f.write("Resampled Features 유사도\n")
                f.write("=" * 70 + "\n")
                for metric, values in resampler_sim.items():
                    if values is None or (isinstance(values, list) and all(v is None for v in values)):
                        continue
                    f.write(f"\n{metric}:\n")
                    if isinstance(values, list):
                        for pair, val in zip(pairs, values):
                            if val is not None:
                                f.write(f"  View {pair[0]+1} ↔ View {pair[1]+1}: {val:.4f}\n")
                        valid_values = [v for v in values if v is not None]
                        if valid_values:
                            f.write(f"  평균: {np.mean(valid_values):.4f} ± {np.std(valid_values):.4f}\n")
                    else:
                        f.write(f"  값: {values:.4f}\n")
        
        print(f"✅ 유사도 분석 결과 저장: {similarity_file}")
    
    print("\n" + "="*70)
    print(f"✅ 모든 시각화 완료! 결과 저장 위치: {output_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="학습된 PanoLLaVA 모델의 Vision Encoder 시각화 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 사용
  python scripts/visualize_trained_model.py \\
      --checkpoint runs/SQ3_1_latent768_PE_e2p_vision_mlp/last.ckpt \\
      --image data/quic360/test/image.jpg \\
      --crop_strategy anyres_e2p
        """
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="학습된 체크포인트 경로 (.ckpt)")
    parser.add_argument("--image", type=str, required=True,
                       help="입력 이미지 경로")
    parser.add_argument("--crop_strategy", type=str, default="e2p",
                       help="이미지 crop 전략 (e2p, anyres_e2p, cubemap, etc.)")
    parser.add_argument("--image_size", type=int, default=224,
                       help="입력 이미지 크기")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="출력 디렉토리 (기본: visualizations/{checkpoint_name})")
    parser.add_argument("--n_components", type=int, default=3,
                       help="PCA 주성분 개수 (RGB 시각화용)")
    parser.add_argument("--no_cls_token", action="store_true",
                       help="CLS 토큰을 제거하지 않음")
    parser.add_argument("--bg_removal", type=str, default="threshold",
                       choices=["threshold", "remove_first_pc", "outlier_removal", "none"],
                       help="배경 제거 방법")
    parser.add_argument("--no_similarity", action="store_true",
                       help="유사도 분석을 수행하지 않음")
    parser.add_argument("--device", type=str, default="auto",
                       help="사용할 디바이스 (auto, cuda, cpu)")

    args = parser.parse_args()

    # 디바이스 설정
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # 체크포인트 이름 추출
    checkpoint_name = Path(args.checkpoint).parent.name

    # 출력 디렉토리 설정
    if args.output_dir is None:
        args.output_dir = f"results/visualizations/{checkpoint_name}"

    print("="*70)
    print("🚀 학습된 모델 Vision Encoder 시각화")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image: {args.image}")
    print(f"Crop Strategy: {args.crop_strategy}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print("="*70)

    # 모델 로드
    model, model_config = load_trained_model(
        checkpoint_path=args.checkpoint,
        device=device
    )

    # Feature 추출
    vision_features, resampled_features, pixel_values, view_metadata = extract_vision_features_from_model(
        model=model,
        image_path=args.image,
        crop_strategy=args.crop_strategy,
        image_size=args.image_size,
        device=device
    )
    
    # 시각화
    bg_removal = None if args.bg_removal == "none" else args.bg_removal
    visualize_features(
        vision_features=vision_features,
        resampled_features=resampled_features,
        pixel_values=pixel_values,
        output_dir=args.output_dir,
        n_components=args.n_components,
        remove_cls_token=not args.no_cls_token,
        bg_removal_method=bg_removal or "threshold",
        analyze_similarity=not args.no_similarity,
        view_metadata=view_metadata,
        checkpoint_name=f"_{checkpoint_name}"
    )


if __name__ == "__main__":
    main()
