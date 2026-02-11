#!/usr/bin/env python
"""
PanoLLaVA 이미지 프로세서 시각화
여러 crop strategy (resize, cubemap, anyres-e2p)로 처리된 이미지를 시각화합니다.

사용법:
    python scripts/visualize_processors.py --image-path /path/to/pano.jpg
    python scripts/visualize_processors.py --image-path /path/to/pano.jpg --output /path/to/output.png
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import torch

# PanoLLaVA imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from panovlm.processors.image import PanoramaImageProcessor


class ProcessorVisualizer:
    """이미지 프로세서 시각화 클래스"""
    
    def __init__(self, image_path: str, image_size: Tuple[int, int] = (224, 224)):
        self.image_path = Path(image_path)
        self.image_size = image_size
        self.original_image = self._load_image()
        self.results = {}
        
        print(f"📷 원본 이미지 로드: {self.image_path}")
        print(f"   이미지 크기: {self.original_image.size}")
    
    def _load_image(self) -> Image.Image:
        """이미지 로드"""
        if not self.image_path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {self.image_path}")
        
        image = Image.open(self.image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def _process_strategy(self, strategy: str) -> dict:
        """특정 전략으로 이미지 처리"""
        print(f"\n🔄 처리 중: {strategy.upper()}")
        
        try:
            # AnyRes-E2P는 다른 파라미터 필요
            if strategy == 'anyres_e2p':
                processor = PanoramaImageProcessor(
                    image_size=self.image_size,
                    crop_strategy=strategy,
                    fov_deg=90.0,
                    overlap_ratio=0.5,
                    normalize=False,
                    # AnyRes E2P 파라미터
                    anyres_e2p_base_size=336,  # 전역 뷰 크기
                    anyres_e2p_tile_size=336,  # 타일 크기 (base_size와 동일)
                    anyres_e2p_vit_size=336,
                    anyres_e2p_closed_loop=True,
                    anyres_e2p_pitch_range=(-45.0, 45.0)
                )
            else:
                processor = PanoramaImageProcessor(
                    image_size=self.image_size,
                    crop_strategy=strategy,
                    fov_deg=90.0,
                    overlap_ratio=0.5,
                    normalize=False  # 시각화용으로 정규화하지 않음
                )
            
            # 이미지 처리 (return_metadata=True로 메타데이터도 함께 반환)
            images, metadata = processor(self.original_image, return_metadata=True)
            
            result = {
                'strategy': strategy,
                'processor': processor,
                'images': images,
                'metadata': metadata,
                'num_views': processor.num_views,
                'image_size': self.image_size,
            }
            
            print(f"  ✓ 성공: {processor.num_views}개 뷰 생성")
            print(f"    이미지 형태: {images.shape}")
            
            return result
        
        except Exception as e:
            print(f"  ❌ 실패: {e}")
            import traceback
            traceback.print_exc()
            return {'strategy': strategy, 'error': str(e)}
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Tensor를 PIL Image로 변환"""
        if isinstance(tensor, torch.Tensor):
            # (C, H, W) → (H, W, C)
            if tensor.dim() == 3:
                array = tensor.permute(1, 2, 0).numpy()
            else:
                array = tensor.numpy()
        else:
            array = tensor
        
        # Normalize to [0, 255]
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(array)
    
    def _get_tile_visualization(self, result: dict) -> Image.Image:
        """타일 배치 시각화 (anyres-e2p용)"""
        if result.get('error'):
            return None
        
        strategy = result['strategy']
        
        # anyres-e2p 전용 시각화
        if strategy == 'anyres_e2p' and hasattr(result['processor'], 'tile_metas'):
            try:
                # 타일 메타데이터 기반 시각화
                tile_metas = result['processor'].tile_metas
                if not tile_metas:
                    return None
                
                # 전역 뷰 + 타일 시각화
                images = result['images']
                num_views = result['num_views']
                
                # 간단한 격자 시각화
                cols = 4
                rows = (num_views + cols - 1) // cols
                
                fig = plt.figure(figsize=(cols * 2, rows * 2))
                for idx, img_tensor in enumerate(images):
                    ax = plt.subplot(rows, cols, idx + 1)
                    pil_img = self._tensor_to_pil(img_tensor)
                    ax.imshow(pil_img)
                    ax.set_title(f"View {idx}")
                    ax.axis('off')
                
                plt.tight_layout()
                
                # Figure를 PIL Image로 변환
                fig.canvas.draw()
                image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                
                return Image.fromarray(image_data)
            
            except Exception as e:
                print(f"    타일 시각화 오류: {e}")
                return None
        
        return None
    
    def process_all_strategies(self):
        """모든 전략으로 처리"""
        strategies = ['resize', 'cubemap', 'sliding_window', 'anyres_e2p']
        
        print("=" * 70)
        print("📊 이미지 프로세서 시각화 시작")
        print("=" * 70)
        
        for strategy in strategies:
            result = self._process_strategy(strategy)
            self.results[strategy] = result
    
    def create_visualization(self, output_path: Optional[str] = None) -> Path:
        """모든 전략의 결과를 시각화하여 저장"""
        print("\n" + "=" * 70)
        print("🎨 시각화 생성 중...")
        print("=" * 70)
        
        if not self.results:
            raise ValueError("먼저 process_all_strategies()를 실행하세요")
        
        # Figure 생성
        num_strategies = len(self.results)
        fig = plt.figure(figsize=(20, 6 * num_strategies))
        gs = gridspec.GridSpec(num_strategies + 1, 1, height_ratios=[1] + [1] * num_strategies)
        
        # 1. 원본 이미지
        ax_orig = fig.add_subplot(gs[0])
        ax_orig.imshow(self.original_image)
        ax_orig.set_title("원본 파노라마 이미지", fontsize=16, fontweight='bold')
        ax_orig.axis('off')
        
        # 2. 각 전략별 결과
        for idx, (strategy, result) in enumerate(self.results.items(), 1):
            ax = fig.add_subplot(gs[idx])
            
            if result.get('error'):
                # 에러 표시
                ax.text(0.5, 0.5, f"❌ {strategy}\n\n{result['error']}", 
                       ha='center', va='center', fontsize=14, color='red')
                ax.set_title(f"{strategy.upper()} - 처리 실패", fontsize=14, fontweight='bold', color='red')
            else:
                # 처리된 이미지 표시
                images = result['images']
                num_views = result['num_views']
                
                # 여러 뷰를 격자로 표시
                if isinstance(images, torch.Tensor):
                    if images.dim() == 4:  # (num_views, C, H, W)
                        num_cols = min(4, num_views)
                        num_rows = (num_views + num_cols - 1) // num_cols
                        
                        # 서브그리드 생성
                        inner_gs = gridspec.GridSpecFromSubplotSpec(
                            num_rows, num_cols, 
                            subplot_spec=gs[idx],
                            wspace=0.05, hspace=0.05
                        )
                        
                        for view_idx, img_tensor in enumerate(images):
                            inner_ax = fig.add_subplot(inner_gs[view_idx])
                            pil_img = self._tensor_to_pil(img_tensor)
                            inner_ax.imshow(pil_img)
                            inner_ax.set_title(f"View {view_idx}", fontsize=10)
                            inner_ax.axis('off')
                    else:  # (C, H, W)
                        pil_img = self._tensor_to_pil(images)
                        ax.imshow(pil_img)
                        ax.set_title(f"{strategy.upper()} - 단일 뷰", fontsize=14, fontweight='bold')
                
                ax.axis('off')
                
                # 정보 추가
                info_text = f"전략: {strategy}\n뷰 수: {num_views}\n크기: {result['image_size']}"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 저장 경로 결정
        if output_path is None:
            output_path = self.image_path.parent / f"{self.image_path.stem}_processor_viz.png"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 저장
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ 시각화 저장: {output_path}")
        
        plt.close()
        return output_path
    
    def print_summary(self):
        """처리 결과 요약 출력"""
        print("\n" + "=" * 70)
        print("📈 처리 결과 요약")
        print("=" * 70)
        
        for strategy, result in self.results.items():
            print(f"\n{strategy.upper()}:")
            if result.get('error'):
                print(f"  ❌ 오류: {result['error']}")
            else:
                print(f"  ✓ 상태: 성공")
                print(f"  - 뷰 수: {result['num_views']}")
                print(f"  - 이미지 크기: {result['image_size']}")
                print(f"  - 이미지 형태: {result['images'].shape if hasattr(result['images'], 'shape') else 'list'}")
        
        print("\n" + "=" * 70)
    
    def save_views_by_strategy(self, output_dir: Optional[str] = None) -> Path:
        """각 전략별로 뷰 이미지를 폴더에 저장"""
        print("\n" + "=" * 70)
        print("💾 각 전략별 이미지 저장 중...")
        print("=" * 70)
        
        if not self.results:
            raise ValueError("먼저 process_all_strategies()를 실행하세요")
        
        # 출력 디렉토리 결정
        if output_dir is None:
            base_output_dir = self.image_path.parent / f"{self.image_path.stem}_processor_views"
        else:
            base_output_dir = Path(output_dir)
        
        saved_stats = {}
        
        for strategy, result in self.results.items():
            if result.get('error'):
                print(f"\n❌ {strategy.upper()}: 건너뜀 (처리 실패)")
                continue
            
            # 전략별 폴더 생성
            strategy_dir = base_output_dir / strategy
            strategy_dir.mkdir(parents=True, exist_ok=True)
            
            images = result['images']
            num_views = result['num_views']
            
            print(f"\n📁 {strategy.upper()}")
            print(f"   저장 경로: {strategy_dir}")
            
            # 이미지 저장
            saved_count = 0
            if isinstance(images, torch.Tensor):
                if images.dim() == 4:  # (num_views, C, H, W)
                    for view_idx, img_tensor in enumerate(images):
                        pil_img = self._tensor_to_pil(img_tensor)
                        img_path = strategy_dir / f"{strategy}_view_{view_idx:03d}.png"
                        pil_img.save(img_path, quality=95)
                        saved_count += 1
                else:  # (C, H, W)
                    pil_img = self._tensor_to_pil(images)
                    img_path = strategy_dir / f"{strategy}_view_000.png"
                    pil_img.save(img_path, quality=95)
                    saved_count += 1
            
            print(f"   ✓ {saved_count}개 이미지 저장 완료")
            saved_stats[strategy] = {
                'count': saved_count,
                'path': str(strategy_dir),
                'num_views': num_views
            }
        
        print("\n" + "=" * 70)
        print("📊 저장 요약")
        print("=" * 70)
        for strategy, stats in saved_stats.items():
            print(f"{strategy.upper()}:")
            print(f"  - 이미지 수: {stats['count']}")
            print(f"  - 저장 위치: {stats['path']}")
        
        print(f"\n✅ 기본 경로: {base_output_dir}")
        return base_output_dir


def main():
    parser = argparse.ArgumentParser(
        description="PanoLLaVA 이미지 프로세서 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용 (시각화 및 이미지 저장)
  python scripts/visualize_processors.py --image-path data/sample.jpg
  
  # 출력 경로 지정
  python scripts/visualize_processors.py --image-path data/sample.jpg --output results/
  
  # 이미지 크기 지정
  python scripts/visualize_processors.py --image-path data/sample.jpg --size 336
  
  # 시각화만 생성 (이미지 저장 안함)
  python scripts/visualize_processors.py --image-path data/sample.jpg --viz-only
        """
    )
    
    parser.add_argument('--image-path', type=str, required=True,
                       help='입력 파노라마 이미지 경로')
    parser.add_argument('--output', type=str, default="vis_ex/",
                       help='출력 디렉토리 경로 (기본: 입력 파일 디렉토리)')
    parser.add_argument('--size', type=int, default=448,
                       help='이미지 처리 크기 (기본: 224)')
    parser.add_argument('--viz-only', action='store_true',
                       help='시각화만 생성하고 개별 이미지는 저장 안함')
    
    args = parser.parse_args()
    
    # 시각화 실행
    try:
        visualizer = ProcessorVisualizer(args.image_path, image_size=(args.size, args.size))
        visualizer.process_all_strategies()
        
        # 비교 시각화 저장
        viz_output = args.output if args.output else None
        viz_path = visualizer.create_visualization(viz_output)
        
        # 개별 이미지 저장 (--viz-only 아닐 때)
        if not args.viz_only:
            views_dir = visualizer.save_views_by_strategy(args.output)
            print(f"\n📁 이미지 저장 위치: {views_dir}")
        
        visualizer.print_summary()
        
        print(f"\n✅ 완료!")
        print(f"� 비교 시각화: {viz_path}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
