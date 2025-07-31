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
                 crop_strategy: str = "e2p",       # sliding_window | e2p | cubemap | anyres | anyres_max
                 fov_deg: float = 90.0,
                 overlap_ratio: float = 0.5,
                 normalize: bool = False,
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
    def __call__(self, x: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        pil = self._to_pil(x)
        match self.crop_strategy:
            case "sliding_window": return self._sliding(pil)
            case "e2p":           return self._e2p(pil)
            case "cubemap":       return self._cubemap4(pil)
            case "resize":        return self._resize(pil)
            case "anyres":        return self._anyres(pil)
            case "anyres_max":    return self._anyres_max(pil)
            case _:                raise ValueError(self.crop_strategy)

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

    def _sliding(self, img:Image.Image) -> torch.Tensor:
        W, H = img.size
        vw = int(W * self.fov_deg / 360)
        stride = int(vw * (1-self.overlap_ratio))
        views = []
        for i in range(self.num_views):
            s, e = i*stride, i*stride+vw
            patch = (img.crop((s,0,e,H)) if e<=W else
                     Image.new("RGB", (vw,H)))
            if e>W:
                patch.paste(img.crop((s,0,W,H)), (0,0))
                patch.paste(img.crop((0,0,e%W,H)), (W-s,0))
            patch = patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
            views.append(self.to_tensor(patch))
        return torch.stack(views,dim=0)   # (V,C,H,W)

    def _e2p(self,img:Image.Image)->torch.Tensor:
        th, tw = self.image_size; stride=self.fov_deg*(1-self.overlap_ratio)
        yaws=((np.arange(self.num_views)*stride)%360); yaws=np.where(yaws>180,yaws-360,yaws)
        keep=0.5; views=[]
        for yaw in yaws:
            npv=e2p(np.array(img),fov_deg=self.fov_deg,u_deg=float(yaw),v_deg=0,out_hw=self.image_size,mode="bilinear")
            h=npv.shape[0]; cut=int(h*(1-keep)/2); npv=npv[cut:h-cut]
            pil=Image.fromarray(npv.astype(np.uint8)).resize(self.image_size[::-1])
            views.append(self.to_tensor(pil))
        return torch.stack(views,dim=0)

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
        AnyRes 방식: 전체 이미지 + 적응적 패치 분할
        LLaVA-NeXT 스타일의 AnyRes 구현
        """
        views = []
        
        # 1. 전체 이미지를 기본 크기로 리사이즈
        global_view = img.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
        views.append(self.to_tensor(global_view))
        
        # 2. 최적 그리드 크기 선택
        orig_width, orig_height = img.size
        best_grid = self._select_best_resolution(orig_width, orig_height)
        
        if best_grid is not None:
            grid_w, grid_h = best_grid
            
            # 3. 이미지를 선택된 그리드 크기로 리사이즈
            resized_img = img.resize((grid_w, grid_h), Image.Resampling.LANCZOS)
            
            # 4. 그리드로 분할
            patch_w = grid_w // (grid_w // self.anyres_patch_size)
            patch_h = grid_h // (grid_h // self.anyres_patch_size)
            
            patches_per_row = grid_w // patch_w
            patches_per_col = grid_h // patch_h
            
            # 5. 각 패치 추출
            for row in range(patches_per_col):
                for col in range(patches_per_row):
                    if len(views) >= self.num_views:
                        break
                    
                    left = col * patch_w
                    top = row * patch_h
                    right = min(left + patch_w, grid_w)
                    bottom = min(top + patch_h, grid_h)
                    
                    patch = resized_img.crop((left, top, right, bottom))
                    patch = patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
                    views.append(self.to_tensor(patch))
                
                if len(views) >= self.num_views:
                    break
        
        # 6. 부족한 뷰는 패딩으로 채움
        while len(views) < self.num_views:
            # 마지막 유효한 뷰를 복제하거나 빈 이미지 사용
            if views:
                views.append(views[-1].clone())
            else:
                empty_img = Image.new('RGB', self.image_size[::-1], (0, 0, 0))
                views.append(self.to_tensor(empty_img))
        
        return torch.stack(views[:self.num_views], dim=0)
    
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
        
        original_area = orig_width * orig_height
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