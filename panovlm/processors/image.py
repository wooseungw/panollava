from typing import Tuple, Union
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
    def __init__(self,
                 image_size: Tuple[int, int] = (224, 224),
                 crop_strategy: str = "e2p",       # sliding_window | e2p | cubemap | anyres | anyres_max | resize
                 fov_deg: float = 90.0,
                 overlap_ratio: float = 0.5,
                 normalize: bool = True,
                 # AnyRes 관련 파라미터
                 anyres_patch_size: int = 336,     # 각 패치의 크기
                 anyres_max_patches: int = 12,     # 최대 패치 수
                 anyres_image_grid_pinpoints: list = None):
        self.image_size, self.crop_strategy = image_size, crop_strategy
        self.fov_deg, self.overlap_ratio = fov_deg, overlap_ratio
        
        # AnyRes 파라미터 초기화
        self.anyres_patch_size = anyres_patch_size
        self.anyres_max_patches = anyres_max_patches
        
        # 기본 grid pinpoints (LLaVA-NeXT 스타일)
        if anyres_image_grid_pinpoints is None:
            self.anyres_image_grid_pinpoints = [
                (336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008),
                (672, 1008), (1008, 672), (336, 1344), (1344, 336), (1008, 1008)
            ]
        else:
            self.anyres_image_grid_pinpoints = anyres_image_grid_pinpoints
        
        # 뷰 수 계산
        if crop_strategy in {"sliding_window","e2p"}:
            stride = fov_deg * (1-overlap_ratio)
            self.num_views = math.ceil(360/stride)
        elif crop_strategy == "cubemap":
            self.num_views = 4
        elif crop_strategy == "resize":
            self.num_views = 1
        elif crop_strategy in {"anyres", "anyres_max"}:
            # AnyRes: 전체 이미지 1개 + 최대 패치 수
            self.num_views = 1 + anyres_max_patches
        else:
            self.num_views = 1
        tf = [transforms.ToTensor()]
        if normalize:
            tf.append(transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]))
        tf.append(transforms.Lambda(lambda t: t.contiguous()))
        self.to_tensor = transforms.Compose(tf)

    # -- public --------------------------------------------------
    def __call__(self, x: Union[str, Image.Image, np.ndarray], return_metadata: bool = False):
        """
        Args:
            x: 입력 이미지
            return_metadata: True이면 (views, metadata) 튜플 반환, False이면 views만 반환
        """
        pil = self._to_pil(x)
        self.view_metadata = []  # 메타데이터 초기화
        
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

    # -- helpers -------------------------------------------------
    @staticmethod
    def _to_pil(x):
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, np.ndarray):
            return Image.fromarray(x).convert("RGB")
        if isinstance(x, str):
            if x.startswith("http"):
                r = requests.get(x, timeout=10); r.raise_for_status()
                return Image.open(BytesIO(r.content)).convert("RGB")
            return Image.open(x).convert("RGB")
        raise TypeError(type(x))

    def _sliding(self, img: Image.Image) -> torch.Tensor:
        """
        파노라마 최적화 슬라이딩: 중요 영역 중심 + 스마트 오버랩
        """
        W, H = img.size
        vw = int(W * self.fov_deg / 360)
        stride = int(vw * (1 - self.overlap_ratio))
        views = []
        
        # 메타데이터 초기화
        if not hasattr(self, 'view_metadata'):
            self.view_metadata = []
        
        # 파노라마 중심 영역 계산 (극값 픽셀만 제외)
        center_h = H // 2
        # 극점 영역 제외: 상하 각각 약 15도 정도 (전체 높이의 1/12씩)
        polar_margin = H // 12  # 극점 왜곡 영역만 제외
        
        crop_top = max(0, polar_margin)  # 상단 극점 제외
        crop_bottom = min(H, H - polar_margin)  # 하단 극점 제외
        effective_height = crop_bottom - crop_top
        
        for i in range(self.num_views):
            # 1. 기본 수평 슬라이딩 좌표
            s, e = i * stride, i * stride + vw
            
            # 2. 파노라마 좌우 연결 처리 (구면 특성 반영)
            if e <= W:
                # 일반적인 경우: 경계 내부
                base_patch = img.crop((s, 0, e, H))
            else:
                # 경계 넘어가는 경우: 좌우 연결 (360도 연속성)
                base_patch = Image.new("RGB", (vw, H))
                
                # 좌측 부분 (현재 위치에서 이미지 끝까지)
                if s < W:
                    left_part = img.crop((s, 0, W, H))
                    base_patch.paste(left_part, (0, 0))
                    
                # 우측 부분 (이미지 시작부터 필요한 만큼)
                right_width = e - W
                if right_width > 0:
                    right_part = img.crop((0, 0, right_width, H))
                    base_patch.paste(right_part, (W - s, 0))
            
            # 3. 수직 스마트 크롭 (파노라마 왜곡 최소화)
            # 상하 극점 부근은 심하게 왜곡되므로 중앙 영역 집중
            smart_patch = base_patch.crop((0, crop_top, vw, crop_bottom))
            
            # 4. 적응적 리사이징 
            # 종횡비 보정으로 자연스러운 시점 유지
            target_ratio = self.image_size[1] / self.image_size[0]  # w/h
            current_ratio = vw / effective_height
            
            if abs(current_ratio - target_ratio) > 0.1:  # 종횡비 차이가 클 때
                # 중앙 크롭으로 종횡비 맞춤
                if current_ratio > target_ratio:  # 너무 넓음
                    new_width = int(effective_height * target_ratio)
                    crop_left = (vw - new_width) // 2
                    smart_patch = smart_patch.crop((crop_left, 0, crop_left + new_width, effective_height))
                else:  # 너무 높음
                    new_height = int(vw / target_ratio)
                    if new_height < effective_height:
                        crop_top_local = (effective_height - new_height) // 2
                        smart_patch = smart_patch.crop((0, crop_top_local, vw, crop_top_local + new_height))
            
            # 5. 최종 리사이징 및 텐서 변환
            final_patch = smart_patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
            views.append(self.to_tensor(final_patch))
            
            # 메타데이터 추가
            yaw = (i * stride * 360.0 / W) % 360.0
            if yaw > 180:
                yaw -= 360
                
            view_meta = {
                'yaw': yaw,
                'pitch': 0.0,
                'original_fov': self.fov_deg,
                'effective_fov': self.fov_deg,  # sliding에서는 FOV 변화 없음
                'crop_ratio': 1.0,
                'view_index': len(views) - 1
            }
            self.view_metadata.append(view_meta)
        
        return torch.stack(views, dim=0)  # (V,C,H,W)

    def _e2p(self, img: Image.Image) -> torch.Tensor:
        """E2P 변환 with proper FOV tracking for geometric alignment"""
        # Note: th, tw variables kept for potential future use
        # th, tw = self.image_size
        stride = self.fov_deg * (1 - self.overlap_ratio)
        yaws = ((np.arange(self.num_views) * stride) % 360)
        yaws = np.where(yaws > 180, yaws - 360, yaws)
        
        keep = 0.5  # Central crop ratio
        views = []
        
        # Store original FOV for metadata
        original_fov = self.fov_deg
        
        for yaw in yaws:
            # Generate E2P view
            npv = e2p(np.array(img), fov_deg=self.fov_deg, u_deg=float(yaw), 
                     v_deg=0, out_hw=self.image_size, mode="bilinear")
            
            # Apply central crop and calculate effective FOV
            h = npv.shape[0]
            cut = int(h * (1 - keep) / 2)
            npv_cropped = npv[cut:h-cut]
            
            # Calculate effective vertical FOV after cropping
            # FOV' = 2 * arctan(keep * tan(FOV/2))
            effective_fov = 2 * np.arctan(keep * np.tan(np.radians(original_fov / 2)))
            effective_fov = np.degrees(effective_fov)
            
            # Store metadata for warp alignment (stored as attributes)
            if not hasattr(self, 'view_metadata'):
                self.view_metadata = []
            
            view_meta = {
                'yaw': float(yaw),
                'pitch': 0.0,
                'original_fov': original_fov,
                'effective_fov': effective_fov,
                'crop_ratio': keep,
                'view_index': len(views)
            }
            self.view_metadata.append(view_meta)
            
            pil = Image.fromarray(npv_cropped.astype(np.uint8)).resize(self.image_size[::-1])
            views.append(self.to_tensor(pil))
            
        return torch.stack(views, dim=0)

    def _cubemap4(self,img:Image.Image)->torch.Tensor:
        face_w=self.image_size[1]
        faces=e2c(np.array(img),face_w=face_w,cube_format="dict")
        order=[faces[k] for k in ("F","R","B","L")]
        views=[self.to_tensor(Image.fromarray(f.astype(np.uint8)).resize(self.image_size[::-1])) for f in order]
        return torch.stack(views,dim=0)  # (V,C,H,W)
    
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
        horizon_margin = orig_height // 4  # 중앙 50% 영역
        
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
                    crop_ratio = 0.8  # 중앙 80% 사용
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
    
    def _select_best_resolution(self, orig_width: int, orig_height: int) -> Tuple[int, int]:
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