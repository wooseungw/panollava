import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
import torch
import lpips
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

# ──────────────────────────────────────────────────────────────────────────────
# 1) 유사도 측정 헬퍼 함수
# ──────────────────────────────────────────────────────────────────────────────

def token_cosine(A: np.ndarray, B: np.ndarray) -> float:
    """두 토큰 집합 간의 평균 코사인 유사도를 계산합니다."""
    num = np.sum(A * B, axis=-1)
    den = np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-8
    return float(np.mean(num / den))

def hungarian_cosine(A: np.ndarray, B: np.ndarray) -> float:
    """Hungarian 알고리즘을 사용하여 최적 매칭 후 코사인 유사도를 계산합니다."""
    sim_mat = (A @ B.T) / (
        (np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None]) + 1e-8
    )
    row, col = linear_sum_assignment(-sim_mat)
    return float(sim_mat[row, col].mean())

def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """선형 Centered Kernel Alignment (CKA) 유사도를 계산합니다."""
    gram_A = A @ A.T
    gram_B = B @ B.T
    cka = np.sum(gram_A * gram_B) / np.sqrt(np.sum(gram_A**2) * np.sum(gram_B**2))
    return float(cka)

def rgb_ssim(A: np.ndarray, B: np.ndarray) -> float:
    """두 RGB 이미지 간의 Structural Similarity (SSIM)를 계산합니다."""
    return float(ssim(A, B, channel_axis=-1, data_range=1.0))

def rgb_lpips(A: np.ndarray, B: np.ndarray, model: lpips.LPIPS) -> float:
    """두 RGB 이미지 간의 LPIPS를 계산합니다. 실패 시 SSIM으로 대체합니다."""
    a = torch.from_numpy(A).permute(2, 0, 1)[None].float() * 2 - 1
    b = torch.from_numpy(B).permute(2, 0, 1)[None].float() * 2 - 1
    try:
        with torch.no_grad():
            d = model(a, b)
        return -float(d.item())  # 거리를 음수 유사도로 변환
    except RuntimeError as err:
        print(f"⚠️ LPIPS 연산 실패, SSIM으로 대체: {err}")
        return rgb_ssim(A, B)

def warp_grid_sample(source_features: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
    """Backward warp using grid_sample for geometric alignment"""
    return F.grid_sample(source_features, target_coords, mode='bilinear', 
                        padding_mode='border', align_corners=False)

def create_erp_coords(height: int, width: int, yaw_deg: float = 0.0, pitch_deg: float = 0.0, 
                     fov_deg: float = 90.0) -> torch.Tensor:
    """Create ERP coordinate grid for warping between different views"""
    # Create normalized grid [-1, 1]
    y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
    
    # Convert to spherical coordinates
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg) 
    fov_rad = np.radians(fov_deg)
    
    # Convert perspective coordinates to spherical
    tan_half_fov = np.tan(fov_rad / 2)
    sphere_x = x * tan_half_fov
    sphere_y = y * tan_half_fov
    sphere_z = torch.ones_like(x)
    
    # Apply rotation
    # Rotation around Y axis (yaw)
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    rotated_x = sphere_x * cos_yaw + sphere_z * sin_yaw
    rotated_z = -sphere_x * sin_yaw + sphere_z * cos_yaw
    
    # Rotation around X axis (pitch)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
    final_y = sphere_y * cos_pitch - rotated_z * sin_pitch
    final_z = sphere_y * sin_pitch + rotated_z * cos_pitch
    
    # Convert to ERP coordinates
    longitude = torch.atan2(rotated_x, final_z)  # [-π, π]
    latitude = torch.atan2(final_y, torch.sqrt(rotated_x**2 + final_z**2))  # [-π/2, π/2]
    
    # Normalize to [-1, 1] for grid_sample
    erp_x = longitude / np.pi  # [-1, 1]
    erp_y = latitude / (np.pi / 2)  # [-1, 1]
    
    return torch.stack([erp_x, erp_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

def compute_warp_aligned_similarity(features_a: np.ndarray, features_b: np.ndarray,
                                  metadata_a: dict, metadata_b: dict, 
                                  patch_size: int = 16) -> dict:
    """Compute similarity after warping features for geometric alignment"""
    # Convert to torch tensors
    feat_a = torch.from_numpy(features_a).float()
    feat_b = torch.from_numpy(features_b).float()
    
    # Reshape to spatial format [1, C, H, W] where H*W = num_patches
    num_patches = feat_a.shape[0]
    spatial_size = int(np.sqrt(num_patches))
    channels = feat_a.shape[1]
    
    feat_a = feat_a.T.view(1, channels, spatial_size, spatial_size)
    feat_b = feat_b.T.view(1, channels, spatial_size, spatial_size)
    
    # Create warp coordinates from b to a coordinate system
    yaw_diff = metadata_a['yaw'] - metadata_b['yaw']
    pitch_diff = metadata_a.get('pitch', 0) - metadata_b.get('pitch', 0)
    fov_a = metadata_a.get('effective_fov', metadata_a.get('original_fov', 90))
    
    target_coords = create_erp_coords(spatial_size, spatial_size, yaw_diff, pitch_diff, fov_a)
    
    # Warp features_b to align with features_a coordinate system
    feat_b_warped = warp_grid_sample(feat_b, target_coords)
    
    # Convert back to patch format
    feat_a_flat = feat_a.view(channels, -1).T  # [num_patches, channels]
    feat_b_warped_flat = feat_b_warped.view(channels, -1).T
    
    # Create overlap mask (where both views have valid data)
    # For simplicity, assume central region is valid
    mask_size = spatial_size // 4  # Central 50% region
    overlap_mask = torch.zeros(spatial_size, spatial_size)
    start_idx = spatial_size // 2 - mask_size // 2
    end_idx = start_idx + mask_size
    overlap_mask[start_idx:end_idx, start_idx:end_idx] = 1
    overlap_mask = overlap_mask.view(-1) > 0
    
    # Apply ERP latitude weighting (cos(φ))
    y_coords = torch.linspace(-np.pi/2, np.pi/2, spatial_size)
    cos_weights = torch.cos(y_coords).repeat(spatial_size, 1)
    cos_weights = cos_weights.view(-1)
    
    # Combined mask with ERP weighting
    final_weights = cos_weights * overlap_mask.float()
    valid_indices = final_weights > 0
    
    if valid_indices.sum() == 0:
        return {'ocs': 0.0, 'residual_mean': 1.0, 'valid_ratio': 0.0}
    
    # Compute weighted cosine similarity (OCS)
    feat_a_valid = feat_a_flat[valid_indices]
    feat_b_valid = feat_b_warped_flat[valid_indices]
    weights_valid = final_weights[valid_indices]
    
    # Normalize features
    feat_a_norm = F.normalize(feat_a_valid, dim=1)
    feat_b_norm = F.normalize(feat_b_valid, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(feat_a_norm * feat_b_norm, dim=1)
    
    # Weighted mean
    ocs = torch.sum(cosine_sim * weights_valid) / torch.sum(weights_valid)
    
    # Compute residual (1 - cosine similarity)
    residual = 1 - cosine_sim
    residual_mean = torch.sum(residual * weights_valid) / torch.sum(weights_valid)
    
    return {
        'ocs': float(ocs),
        'residual_mean': float(residual_mean),
        'valid_ratio': float(valid_indices.sum()) / float(len(valid_indices))
    }

# ──────────────────────────────────────────────────────────────────────────────
# 2) DINO 시각화 및 분석 클래스
# ──────────────────────────────────────────────────────────────────────────────

class DinoVisualizer:
    """
    DINOv2와 같은 Vision Transformer의 hidden states를 시각화하고 분석하는 클래스.
    """
    def __init__(self, hidden_states_list: List[np.ndarray], remove_cls_token: bool = True):
        """
        Args:
            hidden_states_list: 각 원소가 [seq_len, hidden_dim] 또는 
                                [batch_size, seq_len, hidden_dim] 형태인 리스트.
            remove_cls_token: CLS 토큰을 제거할지 여부.
        """
        self.hidden_states_list = hidden_states_list
        self.remove_cls_token = remove_cls_token
        self.processed_tokens = self._preprocess_tokens()
        
        self.pca_model: Optional[PCA] = None
        self.pca_rgb_images: Optional[List[np.ndarray]] = None
        self._lpips_model: Optional[lpips.LPIPS] = None

    @property
    def lpips_model(self) -> lpips.LPIPS:
        """LPIPS 모델을 지연 로딩합니다."""
        if self._lpips_model is None:
            print("LPIPS 모델을 로딩합니다...")
            self._lpips_model = lpips.LPIPS(net='alex').to('cpu')
        return self._lpips_model

    def _preprocess_tokens(self) -> List[np.ndarray]:
        """Hidden states를 정제하여 패치 토큰만 추출합니다."""
        processed = []
        for hs in self.hidden_states_list:
            if hs.ndim == 3:
                hs = hs[0]  # 배치 차원 제거
            
            patch_tokens = hs[1:] if self.remove_cls_token and len(hs) > 1 else hs
            processed.append(patch_tokens)
        return processed

    def fit_pca(self, n_components: int = 3, use_background_removal: bool = True, 
               bg_threshold: float = 0.6, use_global_scaling: bool = True):
        """
        모든 이미지의 패치 토큰에 대해 공통 PCA 모델을 학습하고,
        각 이미지를 RGB로 변환합니다. (개선: 공통 스케일링 추가)
        
        Args:
            n_components: PCA 주성분 개수.
            use_background_removal: 배경 제거 기법 사용 여부.
            bg_threshold: 배경/전경을 나누는 임계값.
            use_global_scaling: 모든 이미지에 대해 공통 스케일링 사용 여부.
        """
        combined_tokens = np.vstack(self.processed_tokens)
        
        # 배경 제거 (선택적)
        if use_background_removal:
            pca_bg = PCA(n_components=1)
            bg_component = pca_bg.fit_transform(combined_tokens)
            foreground_mask = bg_component.flatten() > bg_threshold
            
            fittable_tokens = combined_tokens[foreground_mask] if np.sum(foreground_mask) > 0 else combined_tokens
        else:
            fittable_tokens = combined_tokens
            
        # 공통 PCA 모델 학습
        self.pca_model = PCA(n_components=n_components)
        self.pca_model.fit(fittable_tokens)
        
        # 모든 이미지를 PCA로 변환
        all_semantic_features = []
        for patch_tokens in self.processed_tokens:
            semantic_features = self.pca_model.transform(patch_tokens)
            all_semantic_features.append(semantic_features)
        
        # 공통 스케일링을 위한 글로벌 min/max 계산
        if use_global_scaling:
            all_features_combined = np.vstack(all_semantic_features)
            global_min = np.percentile(all_features_combined, 2, axis=0)  # 2nd percentile
            global_max = np.percentile(all_features_combined, 98, axis=0)  # 98th percentile
        
        # 각 이미지를 RGB로 변환 (공통 스케일 적용)
        self.pca_rgb_images = []
        for semantic_features in all_semantic_features:
            rgb_features = np.zeros_like(semantic_features)
            
            for i in range(n_components):
                component = semantic_features[:, i]
                
                if use_global_scaling:
                    # 공통 스케일 사용
                    min_val, max_val = global_min[i], global_max[i]
                else:
                    # 개별 이미지별 스케일 사용
                    min_val, max_val = component.min(), component.max()
                
                if max_val != min_val:
                    rgb_features[:, i] = np.clip(
                        (component - min_val) / (max_val - min_val), 0, 1
                    )
                else:
                    rgb_features[:, i] = 0.5
            
            num_patches = len(semantic_features)
            patch_size = int(np.sqrt(num_patches))
            pca_rgb = rgb_features.reshape(patch_size, patch_size, n_components)
            self.pca_rgb_images.append(pca_rgb)
            
        print(f"PCA 모델 학습 및 RGB 변환 완료. (공통 스케일링: {use_global_scaling})")
        
        # 스케일링 정보 저장
        self.global_scaling = use_global_scaling
        if use_global_scaling:
            self.global_scale_info = {
                'min_vals': global_min,
                'max_vals': global_max
            }

    def get_hidden_similarity(self, pairs: Optional[List[Tuple[int, int]]] = None, 
                            view_metadata: Optional[List[dict]] = None) -> Dict[str, List[float]]:
        """두 hidden states 쌍 간의 유사도를 계산합니다. (개선: warp-aligned 비교 추가)"""
        if pairs is None:
            pairs = [(i, (i + 1) % len(self.processed_tokens)) for i in range(len(self.processed_tokens))]

        results = {"mse": [], "cosine": [], "cka": [], "hungarian": []}
        if view_metadata:
            results["warp_ocs"] = []
            results["warp_residual"] = []
            
        print("▶ Hidden-space 유사도 (MSE, Cosine, CKA, Hungarian, Warp-OCS):")
        for i, j in pairs:
            A = self.processed_tokens[i]
            B = self.processed_tokens[j]
            
            mse = np.mean((A - B)**2)
            cosine = token_cosine(A, B)
            cka = linear_cka(A, B)
            hg = hungarian_cosine(A, B)
            
            results["mse"].append(mse)
            results["cosine"].append(cosine)
            results["cka"].append(cka)
            results["hungarian"].append(hg)
            
            # Warp-aligned similarity if metadata available
            warp_info = ""
            if view_metadata and i < len(view_metadata) and j < len(view_metadata):
                try:
                    warp_result = compute_warp_aligned_similarity(
                        A, B, view_metadata[i], view_metadata[j]
                    )
                    results["warp_ocs"].append(warp_result['ocs'])
                    results["warp_residual"].append(warp_result['residual_mean'])
                    warp_info = f", OCS={warp_result['ocs']:.4f}"
                except Exception as e:
                    print(f"  Warning: Warp alignment failed for pair ({i}, {j}): {e}")
                    if "warp_ocs" in results:
                        results["warp_ocs"].append(0.0)
                        results["warp_residual"].append(1.0)
            
            print(f"  Pair ({i}, {j}): MSE={mse:.4f}, Cos={cosine:.4f}, CKA={cka:.4f}, Hung={hg:.4f}{warp_info}")
        return results

    def get_pca_similarity(self, pairs: Optional[List[Tuple[int, int]]] = None) -> Dict[str, List[float]]:
        """두 PCA-RGB 이미지 쌍 간의 유사도를 계산합니다."""
        if self.pca_rgb_images is None:
            raise RuntimeError("PCA를 먼저 수행해야 합니다. `fit_pca()`를 호출하세요.")
        if pairs is None:
            pairs = [(i, (i + 1) % len(self.pca_rgb_images)) for i in range(len(self.pca_rgb_images))]

        results = {"mse": [], "ssim": [], "lpips": []}
        print("▶ PCA-RGB 유사도 (MSE, SSIM, LPIPS):")
        for i, j in pairs:
            A = self.pca_rgb_images[i]
            B = self.pca_rgb_images[j]
            
            mse = np.mean((A - B)**2)
            ssim_val = rgb_ssim(A, B)
            lpips_val = rgb_lpips(A, B, self.lpips_model)
            
            results["mse"].append(mse)
            results["ssim"].append(ssim_val)
            results["lpips"].append(lpips_val)
            print(f"  Pair ({i}, {j}): MSE={mse:.4f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
        return results

    def plot_pca_results(self, titles: Optional[List[str]] = None, save_path: Optional[str] = None, figsize: Optional[Tuple[int, int]] = None):
        """PCA 결과를 시각화합니다."""
        if self.pca_rgb_images is None:
            raise RuntimeError("PCA를 먼저 수행해야 합니다. `fit_pca()`를 호출하세요.")
        
        num_images = len(self.pca_rgb_images)
        if titles is None:
            titles = [f'Image {i+1}' for i in range(num_images)]
        if figsize is None:
            figsize = (4 * num_images, 5)
            
        fig, axes = plt.subplots(1, num_images, figsize=figsize)
        if num_images == 1:
            axes = [axes]
            
        for i, (pca_rgb, title) in enumerate(zip(self.pca_rgb_images, titles)):
            axes[i].imshow(pca_rgb)
            axes[i].set_title(f'{title}\nPCA Visualization', fontsize=12)
            axes[i].axis('off')
        
        fig.suptitle('DINOv2 PCA Visualization (First 3 Components as RGB)', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"시각화 저장: {save_path}")
        
        plt.show()
        
        # PCA 분석 정보 출력
        if self.pca_model:
            explained_variance = self.pca_model.explained_variance_ratio_
            print("\n=== PCA 분석 결과 ===")
            for i, r in enumerate(explained_variance):
                print(f"주성분 {i+1} 설명 분산: {r:.2%}")
            print(f"총 설명 분산 (상위 {len(explained_variance)}개): {np.sum(explained_variance):.2%}")

    def plot_patch_cosine_histograms(self, pairs: Optional[List[Tuple[int, int]]] = None, bins: int = 20):
        """패치별 코사인 유사도 분포를 히스토그램으로 시각화합니다."""
        if self.pca_rgb_images is None:
            raise RuntimeError("PCA를 먼저 수행해야 합니다. `fit_pca()`를 호출하세요.")
        if pairs is None:
            pairs = [(i, (i + 1) % len(self.processed_tokens)) for i in range(len(self.processed_tokens))]
        
        for i, j in pairs:
            A = self.processed_tokens[i]
            B = self.processed_tokens[j]
            cos_patch = np.sum(A * B, axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-8)
            
            plt.figure(figsize=(6, 4))
            plt.hist(cos_patch, bins=bins, alpha=0.8)
            plt.title(f'Patch-wise Cosine Distribution (Image {i} vs {j})')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 3) 실행 예제
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # --- 가상 데이터 생성 ---
    # 실제 사용 시에는 모델에서 얻은 hidden_states를 사용하세요。
    # 예: last_hidden_states = model_outputs.hidden_states[-1]
    print("실행 예제를 위해 가상 hidden states를 생성합니다.")
    
    # 3개의 이미지를 시뮬레이션
    # 이미지 1: 기본 패턴
    base_pattern = np.random.rand(256, 768)
    # 이미지 2: 기본 패턴과 유사하지만 약간의 노이즈 추가
    noisy_pattern = base_pattern + 0.1 * np.random.randn(256, 768)
    # 이미지 3: 다른 패턴
    random_pattern = np.random.rand(256, 768)
    
    # CLS 토큰 추가
    cls_token1 = np.random.rand(1, 768)
    cls_token2 = np.random.rand(1, 768)
    cls_token3 = np.random.rand(1, 768)
    
    hidden_states_list_demo = [
        np.vstack([cls_token1, base_pattern]),
        np.vstack([cls_token2, noisy_pattern]),
        np.vstack([cls_token3, random_pattern]),
    ]
    
    # --- DinoVisualizer 사용 ---
    
    # 1. 클래스 인스턴스 생성
    visualizer = DinoVisualizer(hidden_states_list=hidden_states_list_demo)
    
    # 2. PCA 학습 및 변환
    visualizer.fit_pca()
    
    # 3. Hidden-space 유사도 계산 (모든 순환 쌍)
    visualizer.get_hidden_similarity()
    
    # 4. PCA-RGB 공간 유사도 계산 (특정 쌍 지정)
    visualizer.get_pca_similarity(pairs=[(0, 1), (0, 2)])
    
    # 5. PCA 결과 시각화
    visualizer.plot_pca_results(
        titles=['Base Image', 'Noisy Image', 'Random Image'],
        save_path='dino_pca_visualization.png'
    )
    
    # 6. 패치 코사인 유사도 히스토그램 (개선된 버전)
    visualizer.plot_patch_cosine_histograms(pairs=[(0, 1)])
    
    # 7. 추가 분석 기능들 (예시)
    # 가상 metadata로 테스트
    demo_metadata = [
        {'yaw': 0.0, 'pitch': 0.0, 'effective_fov': 90.0, 'original_fov': 90.0},
        {'yaw': 45.0, 'pitch': 0.0, 'effective_fov': 90.0, 'original_fov': 90.0},
        {'yaw': 90.0, 'pitch': 0.0, 'effective_fov': 90.0, 'original_fov': 90.0}
    ]
    
    print("\\n추가 분석 기능 시연:")
    visualizer.plot_phi_bin_analysis(pairs=[(0, 1)], view_metadata=demo_metadata)
    visualizer.plot_seam_analysis(pairs=[(0, 1)], seam_width=2)