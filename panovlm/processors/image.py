from typing import Tuple, Union, List, Optional, Dict, Any
import math
import requests
import numpy as np
from io import BytesIO
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
from torchvision import transforms
from py360convert import e2p, e2c

class PanoramaImageProcessor:
    """파노라마 → 멀티뷰 텐서 변환 (GPU autograd 호환)"""
    # Constants
    DEFAULT_IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet/SigLIP standard
    DEFAULT_IMAGE_STD = [0.229, 0.224, 0.225]
    POLAR_MARGIN_RATIO = 12  # Exclude polar distortion areas (1/12 of height)
    CENTRAL_CROP_RATIO = 0.9  # E2P central crop ratio
    HORIZON_MARGIN_RATIO = 4  # Horizon area for AnyRes (1/4 of height)
    ASPECT_RATIO_TOLERANCE = 0.1  # Threshold for aspect ratio correction
    E2P_CROP_RATIO = 0.8  # Central crop for E2P in AnyRes
    CUBEMAP_FACE_SIZE = 256  # Fixed size for cubemap extraction
    
    # Supported crop strategies
    VALID_STRATEGIES = {"sliding_window", "e2p", "cubemap", "anyres", "anyres_max", "resize"}
    
    def __init__(self,
                 image_size: Tuple[int, int] = (224, 224),
                 crop_strategy: str = "e2p",       # sliding_window | e2p | cubemap | anyres | anyres_max | resize
                 fov_deg: float = 90.0,
                 overlap_ratio: float = 0.5,
                 normalize: bool = True,
                 # 정규화 파라미터
                 image_mean: Optional[List[float]] = None,          # 이미지 정규화 평균값
                 image_std: Optional[List[float]] = None,           # 이미지 정규화 표준편차
                 # AnyRes 관련 파라미터
                 anyres_patch_size: int = 336,     # 각 패치의 크기
                 anyres_max_patches: int = 12,     # 최대 패치 수
                 anyres_image_grid_pinpoints: Optional[List[Tuple[int, int]]] = None,
                 # Vision processor 옵션
                 use_vision_processor: bool = False,
                 vision_model_name: Optional[str] = None):
        self._validate_parameters(crop_strategy, image_size, fov_deg, overlap_ratio)
        
        self.image_size = image_size
        self.crop_strategy = crop_strategy
        self.fov_deg = fov_deg
        self.overlap_ratio = overlap_ratio
        
        self._init_anyres_params(anyres_patch_size, anyres_max_patches, anyres_image_grid_pinpoints)
        self._init_normalization_params(image_mean, image_std)
        self._init_vision_processor(use_vision_processor, vision_model_name)
        self._init_transforms(normalize)
        
        self.num_views = self._calculate_num_views()
        self.view_metadata = []

    def _validate_parameters(self, crop_strategy: str, image_size: Tuple[int, int], 
                           fov_deg: float, overlap_ratio: float) -> None:
        """Validate initialization parameters."""
        if crop_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid crop_strategy: {crop_strategy}. Must be one of {self.VALID_STRATEGIES}")
        if not (0 < fov_deg <= 180):
            raise ValueError(f"fov_deg must be between 0 and 180, got {fov_deg}")
        if not (0 <= overlap_ratio < 1):
            raise ValueError(f"overlap_ratio must be between 0 and 1, got {overlap_ratio}")
        if len(image_size) != 2 or any(s <= 0 for s in image_size):
            raise ValueError(f"image_size must be a tuple of two positive integers, got {image_size}")
    
    def _init_anyres_params(self, anyres_patch_size: int, anyres_max_patches: int, 
                          anyres_image_grid_pinpoints: Optional[List[Tuple[int, int]]]) -> None:
        """Initialize AnyRes-related parameters."""
        self.anyres_patch_size = anyres_patch_size
        self.anyres_max_patches = anyres_max_patches
        
        if anyres_image_grid_pinpoints is None:
            self.anyres_image_grid_pinpoints = [
                (336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008),
                (672, 1008), (1008, 672), (336, 1344), (1344, 336), (1008, 1008)
            ]
        else:
            self.anyres_image_grid_pinpoints = anyres_image_grid_pinpoints
    
    def _init_normalization_params(self, image_mean: Optional[List[float]], image_std: Optional[List[float]]) -> None:
        """Initialize normalization parameters."""
        self.image_mean = image_mean or self.DEFAULT_IMAGE_MEAN
        self.image_std = image_std or self.DEFAULT_IMAGE_STD
    
    def _init_vision_processor(self, use_vision_processor: bool, vision_model_name: Optional[str]) -> None:
        """Initialize vision processor settings."""
        self.use_vision_processor = use_vision_processor
        self.vision_model_name = vision_model_name
        self._vision_processor = None  # Lazy loading
    
    def _init_transforms(self, normalize: bool) -> None:
        """Initialize tensor transformation pipeline."""
        transforms_list = [transforms.ToTensor()]
        if normalize and not self.use_vision_processor:
            transforms_list.append(transforms.Normalize(self.image_mean, self.image_std))
        transforms_list.append(transforms.Lambda(lambda t: t.contiguous()))
        self.to_tensor = transforms.Compose(transforms_list)
    
    def _calculate_num_views(self) -> int:
        """Calculate number of views based on crop strategy."""
        if self.crop_strategy in {"sliding_window", "e2p"}:
            stride = self.fov_deg * (1 - self.overlap_ratio)
            return math.ceil(360 / stride)
        elif self.crop_strategy == "cubemap":
            return 4
        elif self.crop_strategy == "resize":
            return 1
        elif self.crop_strategy in {"anyres", "anyres_max"}:
            return 1 + self.anyres_max_patches
        else:
            return 1
    
    def _validate_input_image(self, img: Image.Image) -> None:
        """Validate input image for panorama processing."""
        width, height = img.size
        
        if width < 100 or height < 50:
            raise ValueError(f"Input image is too small ({width}x{height}). Minimum size is 100x50.")
        
        # For panorama images, we expect width to be significantly larger than height
        # Typical panorama aspect ratios are 2:1 or wider
        if self.crop_strategy in {"e2p", "sliding_window", "cubemap"} and width < height:
            raise ValueError(f"Input image ({width}x{height}) appears to be portrait orientation. "
                           "Panorama processing expects landscape orientation (width > height).")
        
        # Warn if aspect ratio is unusual for panorama
        aspect_ratio = width / height
        if self.crop_strategy in {"e2p", "sliding_window"} and aspect_ratio < 1.5:
            import warnings
            warnings.warn(f"Input image has unusual aspect ratio {aspect_ratio:.2f} for panorama processing. "
                         "Consider using 'resize' strategy for non-panoramic images.")

    def _calculate_effective_fov(self, original_fov: float, crop_ratio: float) -> float:
        """Calculate effective FOV after central cropping."""
        effective_fov = 2 * np.arctan(crop_ratio * np.tan(np.radians(original_fov / 2)))
        return np.degrees(effective_fov)
    
    def _generate_e2p_views(self, img: Image.Image) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
        """Generate E2P views and metadata (shared logic)."""
        stride = self.fov_deg * (1 - self.overlap_ratio)
        yaws = ((np.arange(self.num_views) * stride) % 360)
        yaws = np.where(yaws > 180, yaws - 360, yaws)
        
        keep = self.CENTRAL_CROP_RATIO
        pil_views = []
        metadata = []
        original_fov = self.fov_deg
        
        # Cache numpy array to avoid repeated conversion
        img_array = np.array(img)
        
        for yaw in yaws:
            try:
                npv = e2p(img_array, fov_deg=self.fov_deg, u_deg=float(yaw), 
                         v_deg=0, out_hw=self.image_size, mode="bilinear")
            except Exception as e:
                raise ValueError(f"Failed to perform E2P projection for yaw {yaw}°: {e}") from e
            
            h = npv.shape[0]
            cut = int(h * (1 - keep) / 2)
            npv_cropped = npv[cut:h-cut]
            
            effective_fov = self._calculate_effective_fov(original_fov, keep)
            view_meta = {
                'yaw': float(yaw),
                'pitch': 0.0,
                'original_fov': original_fov,
                'effective_fov': effective_fov,
                'crop_ratio': keep,
                'view_index': len(pil_views)
            }
            metadata.append(view_meta)
            
            pil = Image.fromarray(npv_cropped.astype(np.uint8))
            pil_views.append(pil)
            
        return pil_views, metadata
    
    def _generate_sliding_views(self, img: Image.Image) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
        """Generate sliding window views and metadata (shared logic)."""
        W, H = img.size
        vw = int(W * self.fov_deg / 360)
        stride = int(vw * (1 - self.overlap_ratio))
        pil_views = []
        metadata = []
        
        # Pre-compute values to avoid repeated calculations
        polar_margin = H // self.POLAR_MARGIN_RATIO
        crop_top = max(0, polar_margin)
        crop_bottom = min(H, H - polar_margin)
        effective_height = crop_bottom - crop_top
        target_ratio = self.image_size[1] / self.image_size[0]  # w/h
        yaw_step = 360.0 / W  # Degrees per pixel
        
        for i in range(self.num_views):
            s, e = i * stride, i * stride + vw
            
            if e <= W:
                base_patch = img.crop((s, 0, e, H))
            else:
                base_patch = Image.new("RGB", (vw, H))
                if s < W:
                    left_part = img.crop((s, 0, W, H))
                    base_patch.paste(left_part, (0, 0))
                right_width = e - W
                if right_width > 0:
                    right_part = img.crop((0, 0, right_width, H))
                    base_patch.paste(right_part, (W - s, 0))
            
            smart_patch = base_patch.crop((0, crop_top, vw, crop_bottom))
            
            # Aspect ratio adjustment (using pre-computed target_ratio)
            current_ratio = vw / effective_height
            
            if abs(current_ratio - target_ratio) > self.ASPECT_RATIO_TOLERANCE:
                if current_ratio > target_ratio:  # too wide
                    new_width = int(effective_height * target_ratio)
                    crop_left = (vw - new_width) // 2
                    smart_patch = smart_patch.crop((crop_left, 0, crop_left + new_width, effective_height))
                else:  # too tall
                    new_height = int(vw / target_ratio)
                    if new_height < effective_height:
                        crop_top_local = (effective_height - new_height) // 2
                        smart_patch = smart_patch.crop((0, crop_top_local, vw, crop_top_local + new_height))
            
            pil_views.append(smart_patch)
            
            # Metadata (using pre-computed yaw_step)
            yaw = (i * stride * yaw_step) % 360.0
            if yaw > 180:
                yaw -= 360
            view_meta = {
                'yaw': yaw,
                'pitch': 0.0,
                'original_fov': self.fov_deg,
                'effective_fov': self.fov_deg,
                'crop_ratio': 1.0,
                'view_index': len(pil_views) - 1
            }
            metadata.append(view_meta)
        
        return pil_views, metadata

    @property
    def vision_processor(self):
        """Lazy loading of vision processor"""
        if self.use_vision_processor and self._vision_processor is None:
            try:
                from transformers import AutoProcessor
            except ImportError as e:
                raise ImportError("transformers library is required for vision processor functionality. "
                                "Install with: pip install transformers") from e
            
            
            try:
                self._vision_processor = AutoProcessor.from_pretrained(self.vision_model_name, use_fast=True)
            except Exception as e:
                raise ValueError(f"Failed to load vision processor '{self.vision_model_name}': {e}") from e
        return self._vision_processor

    # -- public --------------------------------------------------
    def __call__(self, x: Union[str, Image.Image, np.ndarray], return_metadata: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, Any]]]]:
        """
        Args:
            x: 입력 이미지
            return_metadata: True이면 (views, metadata) 튜플 반환, False이면 views만 반환
        """
        pil = self._to_pil(x)
        self._validate_input_image(pil)
        self.view_metadata = []  # 메타데이터 초기화
        
        if self.use_vision_processor:
            # Vision processor 사용: PIL 이미지들만 생성하고 한번에 처리
            pil_views = self._extract_pil_views(pil)
            views = self._process_with_vision_processor(pil_views)
        else:
            # 기존 방식: 개별적으로 tensor 변환
            match self.crop_strategy:
                case "sliding_window": views = self._sliding(pil)
                case "e2p":           views = self._e2p(pil)
                case "cubemap":       views = self._cubemap4(pil)
                case "resize":        views = self._resize(pil)
                case "anyres":        views = self._anyres(pil)
                case "anyres_max":    views = self._anyres_max(pil)
                case _:                raise ValueError(self.crop_strategy)
        
        if return_metadata:
            return views, getattr(self, 'view_metadata', [])
        return views
    
    def _extract_pil_views(self, pil: Image.Image) -> List[Image.Image]:
        """PIL 이미지들만 추출 (텐서 변환 없이)"""
        match self.crop_strategy:
            case "sliding_window": return self._sliding_pil(pil)
            case "e2p":           return self._e2p_pil(pil)
            case "cubemap":       return self._cubemap4_pil(pil)
            case "resize":        return self._resize_pil(pil)
            case "anyres":        return self._anyres_pil(pil)
            case "anyres_max":    return self._anyres_max_pil(pil)
            case _:                raise ValueError(self.crop_strategy)
    
    def _process_with_vision_processor(self, pil_views: List[Image.Image]) -> torch.Tensor:
        """Vision processor로 batch processing"""
        if not pil_views:
            # AutoProcessor의 실제 이미지 크기 가져오기
            processor_size = self.vision_processor.image_processor.size
            if isinstance(processor_size, dict):
                height, width = processor_size['height'], processor_size['width']
            else:
                height = width = processor_size
            return torch.zeros(0, 3, height, width)
        
        # 모든 뷰를 한번에 처리 (매우 빠름) - AutoProcessor가 자동으로 적절한 크기로 리사이징
        pixel_values = self.vision_processor(
            images=pil_views, 
            return_tensors="pt"
        )["pixel_values"]
        
        return pixel_values

    # -- helpers -------------------------------------------------
    @staticmethod
    def _to_pil(x: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Convert various input types to PIL Image with proper error handling."""
        try:
            if isinstance(x, Image.Image):
                return x.convert("RGB")
            elif isinstance(x, np.ndarray):
                if x.ndim not in (2, 3):
                    raise ValueError(f"Expected 2D or 3D numpy array, got {x.ndim}D")
                if x.ndim == 3 and x.shape[2] not in (1, 3, 4):
                    raise ValueError(f"Expected 1, 3, or 4 channels, got {x.shape[2]}")
                return Image.fromarray(x).convert("RGB")
            elif isinstance(x, str):
                if x.startswith(("http://", "https://")):
                    try:
                        r = requests.get(x, timeout=10)
                        r.raise_for_status()
                        return Image.open(BytesIO(r.content)).convert("RGB")
                    except requests.RequestException as e:
                        raise ValueError(f"Failed to download image from URL: {e}")
                else:
                    try:
                        return Image.open(x).convert("RGB")
                    except (FileNotFoundError, OSError) as e:
                        raise ValueError(f"Failed to open image file '{x}': {e}")
            else:
                raise TypeError(f"Unsupported input type: {type(x)}. Expected Image, numpy array, or string path/URL.")
        except Exception as e:
            if isinstance(e, (ValueError, TypeError)):
                raise
            raise ValueError(f"Failed to convert input to PIL Image: {e}")

    def _sliding(self, img: Image.Image) -> torch.Tensor:
        """파노라마 최적화 슬라이딩: 중요 영역 중심 + 스마트 오버랩"""
        pil_views, metadata = self._generate_sliding_views(img)
        self.view_metadata = metadata
        
        # Pre-allocate tensor for better performance
        views = torch.empty(len(pil_views), 3, *self.image_size, dtype=torch.float32)
        
        for i, pil in enumerate(pil_views):
            final_patch = pil.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
            views[i] = self.to_tensor(final_patch)
            
        return views  # (V,C,H,W)

    def _e2p(self, img: Image.Image) -> torch.Tensor:
        """E2P 변환 with proper FOV tracking for geometric alignment"""
        pil_views, metadata = self._generate_e2p_views(img)
        self.view_metadata = metadata
        
        # Pre-allocate tensor for better performance
        views = torch.empty(len(pil_views), 3, *self.image_size, dtype=torch.float32)
        
        for i, pil in enumerate(pil_views):
            resized_pil = pil.resize(self.image_size[::-1])
            views[i] = self.to_tensor(resized_pil)
            
        return views

    def _cubemap4(self, img: Image.Image) -> torch.Tensor:
        """Generate cubemap views with error handling."""
        try:
            face_w = self.image_size[1]
            faces = e2c(np.array(img), face_w=face_w, cube_format="dict")
            order = ["F", "R", "B", "L"]  # Pre-defined order
            
            # Pre-allocate tensor for better performance
            views = torch.empty(4, 3, *self.image_size, dtype=torch.float32)
            
            for i, face_key in enumerate(order):
                face_img = Image.fromarray(faces[face_key].astype(np.uint8)).resize(self.image_size[::-1])
                views[i] = self.to_tensor(face_img)
                
            return views  # (V,C,H,W)
        except Exception as e:
            raise ValueError(f"Failed to generate cubemap views: {e}") from e
    
    def _resize(self, img:Image.Image) -> torch.Tensor:
        """이미지를 단순히 리사이즈하여 하나의 뷰로 반환"""
        resized_img = img.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
        
        return self.to_tensor(resized_img).unsqueeze(0)
    
    def _anyres(self, img: Image.Image) -> torch.Tensor:
        """
        파노라마 최적화 AnyRes: 글로벌 뷰 + 중요 영역 중심의 로컬 패치
        """
        views = []
        orig_width, orig_height = img.size
        
        # 1. 글로벌 뷰: 전체 파노라마 맥락
        global_view = img.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
        views.append(self.to_tensor(global_view))
        
        # 2. 파노라마 중심 영역 추출 (수평선 근처 - 가장 중요한 정보)
        center_height = orig_height // 2
        horizon_margin = orig_height // self.HORIZON_MARGIN_RATIO  # 중앙 50% 영역
        
        horizon_top = max(0, center_height - horizon_margin)
        horizon_bottom = min(orig_height, center_height + horizon_margin)
        
        # 3. 수평 방향 중요 영역들 (파노라마 특성 반영)
        num_horizontal_patches = min(6, (self.num_views - 1) // 2)  # 글로벌 뷰 제외
        patch_width = orig_width / num_horizontal_patches
        
        # 수평선 근처 패치들 (고해상도 세부사항)
        for i in range(num_horizontal_patches):
            if len(views) >= self.num_views:
                break
                
            left = int(i * patch_width)
            right = int(min((i + 1) * patch_width, orig_width))
            
            # 수평선 근처 패치 (핵심 정보)
            horizon_patch = img.crop((left, horizon_top, right, horizon_bottom))
            horizon_patch = horizon_patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
            views.append(self.to_tensor(horizon_patch))
        
        # 4. 전체 높이 패치들 (추가 컨텍스트)
        remaining_slots = self.num_views - len(views)
        if remaining_slots > 0:
            vertical_patches = min(remaining_slots, num_horizontal_patches)
            patch_width_full = orig_width / vertical_patches
            
            for i in range(vertical_patches):
                if len(views) >= self.num_views:
                    break
                    
                left = int(i * patch_width_full)
                right = int(min((i + 1) * patch_width_full, orig_width))
                
                # 전체 높이 패치 (상하 컨텍스트 포함)
                full_patch = img.crop((left, 0, right, orig_height))
                full_patch = full_patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
                views.append(self.to_tensor(full_patch))
        
        # 5. E2P 변환으로 추가 세부사항 (남은 슬롯이 있으면)
        remaining_slots = self.num_views - len(views)
        if remaining_slots > 0:
            yaw_angles = np.linspace(0, 315, remaining_slots, endpoint=False)  # 45도 간격
            
            for yaw in yaw_angles:
                if len(views) >= self.num_views:
                    break
                    
                try:
                    # E2P로 원근 보정된 패치 생성
                    npv = e2p(np.array(img), fov_deg=self.fov_deg, u_deg=float(yaw), 
                             v_deg=0, out_hw=self.image_size, mode="bilinear")
                    
                    # 중앙 부분만 사용 (왜곡 최소화)
                    h = npv.shape[0]
                    crop_ratio = self.E2P_CROP_RATIO  # 중앙 80% 사용
                    crop_margin = int(h * (1 - crop_ratio) / 2)
                    npv = npv[crop_margin:h-crop_margin]
                    
                    pil = Image.fromarray(npv.astype(np.uint8))
                    pil = pil.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
                    views.append(self.to_tensor(pil))
                    
                except Exception:
                    # E2P 실패 시 중앙 크롭으로 대체
                    center_crop = self._get_center_crop(img, yaw)
                    views.append(self.to_tensor(center_crop))
        
        # 6. 부족한 뷰는 패딩
        while len(views) < self.num_views:
            if views:
                views.append(views[-1].clone())
            else:
                empty_img = Image.new('RGB', self.image_size[::-1], (0, 0, 0))
                views.append(self.to_tensor(empty_img))
        
        return torch.stack(views[:self.num_views], dim=0)
    
    def _get_center_crop(self, img: Image.Image, yaw_deg: float) -> Image.Image:
        """중앙 크롭 헬퍼 함수 (E2P 실패 시 대안)"""
        orig_width, orig_height = img.size
        
        # yaw 각도에 따른 중심점 계산
        center_x = int((yaw_deg / 360.0) * orig_width) % orig_width
        center_y = orig_height // 2
        
        # 크롭 영역 계산
        crop_size = min(orig_width // 4, orig_height // 2)
        left = max(0, center_x - crop_size // 2)
        right = min(orig_width, center_x + crop_size // 2)
        top = max(0, center_y - crop_size // 2)
        bottom = min(orig_height, center_y + crop_size // 2)
        
        # 경계 처리 (파노라마는 좌우가 연결됨)
        if right - left < crop_size:
            if left == 0:
                right = crop_size
            elif right == orig_width:
                left = orig_width - crop_size
        
        patch = img.crop((left, top, right, bottom))
        return patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
    
    def _anyres_max(self, img: Image.Image) -> torch.Tensor:
        """
        AnyRes Max 방식: 더 공격적인 패치 분할
        최대 해상도를 활용한 더 세밀한 분할
        """
        views = []
        
        # 1. 전체 이미지
        global_view = img.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
        views.append(self.to_tensor(global_view))
        
        # 2. 파노라마 특성을 고려한 세밀한 분할
        orig_width, orig_height = img.size
        
        # 파노라마의 특성상 수평 방향으로 더 많이 분할
        horizontal_splits = min(6, self.anyres_max_patches // 2)
        vertical_splits = min(3, self.anyres_max_patches // horizontal_splits)
        
        patch_width = orig_width // horizontal_splits
        patch_height = orig_height // vertical_splits
        
        # 3. 균등 분할로 패치 생성
        for row in range(vertical_splits):
            for col in range(horizontal_splits):
                if len(views) >= self.num_views:
                    break
                
                left = col * patch_width
                top = row * patch_height
                right = min(left + patch_width, orig_width)
                bottom = min(top + patch_height, orig_height)
                
                patch = img.crop((left, top, right, bottom))
                patch = patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
                views.append(self.to_tensor(patch))
            
            if len(views) >= self.num_views:
                break
        
        # 4. E2P 방식으로 추가 뷰 생성 (남은 슬롯이 있으면)
        if len(views) < self.num_views:
            remaining_slots = self.num_views - len(views)
            yaw_step = 360 / remaining_slots
            
            for i in range(remaining_slots):
                yaw = i * yaw_step
                if yaw > 180:
                    yaw -= 360
                
                try:
                    npv = e2p(np.array(img), fov_deg=self.fov_deg, u_deg=float(yaw), 
                             v_deg=0, out_hw=self.image_size, mode="bilinear")
                    pil = Image.fromarray(npv.astype(np.uint8))
                    views.append(self.to_tensor(pil))
                except:
                    # E2P 변환 실패 시 중앙 크롭 사용
                    center_x = orig_width // 2
                    crop_size = min(orig_width, orig_height) // 2
                    left = max(0, center_x - crop_size // 2)
                    right = min(orig_width, center_x + crop_size // 2)
                    top = max(0, orig_height // 2 - crop_size // 2)
                    bottom = min(orig_height, orig_height // 2 + crop_size // 2)
                    
                    patch = img.crop((left, top, right, bottom))
                    patch = patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
                    views.append(self.to_tensor(patch))
        
        # 5. 정확한 뷰 수로 맞춤
        while len(views) < self.num_views:
            views.append(views[-1].clone())
        
        return torch.stack(views[:self.num_views], dim=0)
    
    # -- PIL-only versions (for SigLIP processor) ------------------
    def _e2p_pil(self, img: Image.Image) -> List[Image.Image]:
        """E2P 변환 PIL only version"""
        pil_views, metadata = self._generate_e2p_views(img)
        self.view_metadata = metadata
        return pil_views
    
    def _sliding_pil(self, img: Image.Image) -> List[Image.Image]:
        """Sliding window PIL only version"""
        pil_views, metadata = self._generate_sliding_views(img)
        self.view_metadata = metadata
        return pil_views
    
    def _resize_pil(self, img: Image.Image) -> List[Image.Image]:
        """Resize PIL only version - AutoProcessor will handle resizing"""
        return [img]
    
    def _cubemap4_pil(self, img: Image.Image) -> List[Image.Image]:
        """Cubemap PIL only version - AutoProcessor will handle resizing"""
        # Use a reasonable face size for cubemap extraction
        face_w = self.CUBEMAP_FACE_SIZE  # Fixed size for cubemap faces, AutoProcessor will resize as needed
        faces = e2c(np.array(img), face_w=face_w, cube_format="dict")
        order = [faces[k] for k in ("F", "R", "B", "L")]
        pil_views = [Image.fromarray(f.astype(np.uint8)) for f in order]
        return pil_views
    
    def _anyres_pil(self, img: Image.Image) -> List[Image.Image]:
        """AnyRes PIL only version - AutoProcessor will handle resizing"""
        pil_views = []
        orig_width, orig_height = img.size
        
        # Global view - AutoProcessor will handle resizing
        pil_views.append(img)
        
        # Horizon patches
        center_height = orig_height // 2
        horizon_margin = orig_height // self.HORIZON_MARGIN_RATIO
        horizon_top = max(0, center_height - horizon_margin)
        horizon_bottom = min(orig_height, center_height + horizon_margin)
        
        num_horizontal_patches = min(6, (self.num_views - 1) // 2)
        patch_width = orig_width / num_horizontal_patches
        
        for i in range(num_horizontal_patches):
            if len(pil_views) >= self.num_views:
                break
            left = int(i * patch_width)
            right = int(min((i + 1) * patch_width, orig_width))
            horizon_patch = img.crop((left, horizon_top, right, horizon_bottom))
            pil_views.append(horizon_patch)
        
        return pil_views[:self.num_views]
    
    def _anyres_max_pil(self, img: Image.Image) -> List[Image.Image]:
        """AnyRes Max PIL only version - AutoProcessor will handle resizing"""
        pil_views = []
        orig_width, orig_height = img.size
        
        # Global view - AutoProcessor will handle resizing
        pil_views.append(img)
        
        # Grid patches
        horizontal_splits = min(6, self.anyres_max_patches // 2)
        vertical_splits = min(3, self.anyres_max_patches // horizontal_splits)
        
        patch_width = orig_width // horizontal_splits
        patch_height = orig_height // vertical_splits
        
        for row in range(vertical_splits):
            for col in range(horizontal_splits):
                if len(pil_views) >= self.num_views:
                    break
                
                left = col * patch_width
                top = row * patch_height
                right = min(left + patch_width, orig_width)
                bottom = min(top + patch_height, orig_height)
                
                patch = img.crop((left, top, right, bottom))
                pil_views.append(patch)
            
            if len(pil_views) >= self.num_views:
                break
        
        return pil_views[:self.num_views]
    
    def _select_best_resolution(self, orig_width: int, orig_height: int) -> Optional[Tuple[int, int]]:
        """
        원본 이미지 크기에 가장 적합한 그리드 해상도 선택
        LLaVA-NeXT의 해상도 선택 로직 참고
        """
        if not self.anyres_image_grid_pinpoints:
            return None
        
        # original_area = orig_width * orig_height  # For potential future use
        best_fit = None
        min_wasted_ratio = float('inf')
        
        for grid_w, grid_h in self.anyres_image_grid_pinpoints:
            # 현재 그리드에서 패치 수 계산
            patches_per_row = grid_w // self.anyres_patch_size
            patches_per_col = grid_h // self.anyres_patch_size
            total_patches = patches_per_row * patches_per_col
            
            # 최대 패치 수 제한 확인
            if total_patches > self.anyres_max_patches:
                continue
            
            # 종횡비 매칭 점수 계산
            orig_ratio = orig_width / orig_height
            grid_ratio = grid_w / grid_h
            ratio_diff = abs(orig_ratio - grid_ratio)
            
            # 해상도 효율성 계산
            scale_w = grid_w / orig_width
            scale_h = grid_h / orig_height
            scale = min(scale_w, scale_h)
            
            effective_area = (orig_width * scale) * (orig_height * scale)
            grid_area = grid_w * grid_h
            wasted_ratio = (grid_area - effective_area) / grid_area + ratio_diff * 0.1
            
            if wasted_ratio < min_wasted_ratio:
                min_wasted_ratio = wasted_ratio
                best_fit = (grid_w, grid_h)
        
        return best_fit