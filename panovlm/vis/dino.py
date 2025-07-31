import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
import torch
import lpips
from scipy.optimize import linear_sum_assignment
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

    def fit_pca(self, n_components: int = 3, use_background_removal: bool = True, bg_threshold: float = 0.6):
        """
        모든 이미지의 패치 토큰에 대해 공통 PCA 모델을 학습하고,
        각 이미지를 RGB로 변환합니다.
        
        Args:
            n_components: PCA 주성분 개수.
            use_background_removal: 배경 제거 기법 사용 여부.
            bg_threshold: 배경/전경을 나누는 임계값.
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
        
        # 각 이미지를 PCA로 변환 및 RGB 정규화
        self.pca_rgb_images = []
        for patch_tokens in self.processed_tokens:
            semantic_features = self.pca_model.transform(patch_tokens)
            
            rgb_features = np.zeros_like(semantic_features)
            for i in range(n_components):
                component = semantic_features[:, i]
                min_val, max_val = component.min(), component.max()
                if max_val != min_val:
                    rgb_features[:, i] = (component - min_val) / (max_val - min_val)
                else:
                    rgb_features[:, i] = 0.5
            
            num_patches = len(patch_tokens)
            patch_size = int(np.sqrt(num_patches))
            pca_rgb = rgb_features.reshape(patch_size, patch_size, n_components)
            self.pca_rgb_images.append(pca_rgb)
            
        print("PCA 모델 학습 및 RGB 변환 완료.")

    def get_hidden_similarity(self, pairs: Optional[List[Tuple[int, int]]] = None) -> Dict[str, List[float]]:
        """두 hidden states 쌍 간의 유사도를 계산합니다."""
        if pairs is None:
            pairs = [(i, (i + 1) % len(self.processed_tokens)) for i in range(len(self.processed_tokens))]

        results = {"mse": [], "cosine": [], "cka": [], "hungarian": []}
        print("▶ Hidden-space 유사도 (MSE, Cosine, CKA, Hungarian):")
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
            print(f"  Pair ({i}, {j}): MSE={mse:.4f}, Cosine={cosine:.4f}, CKA={cka:.4f}, Hung={hg:.4f}")
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
    
    # 6. 패치 코사인 유사도 히스토그램
    visualizer.plot_patch_cosine_histograms(pairs=[(0, 1)])