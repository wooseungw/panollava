"""Image processing logic for Panorama VLM."""

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
    """Panorama -> Multi-view tensor conversion."""
    
    VALID_STRATEGIES = {"sliding_window", "e2p", "cubemap", "anyres", "resize", "anyres_e2p"}
    DEFAULT_IMAGE_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_IMAGE_STD = [0.229, 0.224, 0.225]

    def __init__(self,
                 image_size: Tuple[int, int] = (224, 224),
                 crop_strategy: str = "sliding_window",
                 fov_deg: float = 90.0,
                 overlap_ratio: float = 0.5,
                 normalize: bool = True,
                 image_mean: Optional[List[float]] = None,
                 image_std: Optional[List[float]] = None,
                 anyres_max_patches: int = 12):
        
        self.image_size = image_size
        self.crop_strategy = crop_strategy
        self.fov_deg = fov_deg
        self.overlap_ratio = overlap_ratio
        self.anyres_max_patches = anyres_max_patches
        self.image_mean = image_mean or self.DEFAULT_IMAGE_MEAN
        self.image_std = image_std or self.DEFAULT_IMAGE_STD
        
        if crop_strategy not in self.VALID_STRATEGIES:
             raise ValueError(f"Invalid crop strategy: {crop_strategy}")

        self.to_tensor = self._build_transforms(normalize)
        self.num_views = self._calculate_num_views()
        self.view_metadata = []

    def _build_transforms(self, normalize: bool):
        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize(self.image_mean, self.image_std))
        return transforms.Compose(t)

    def _calculate_num_views(self) -> int:
        if self.crop_strategy in {"sliding_window", "e2p"}:
            stride = self.fov_deg * (1 - self.overlap_ratio)
            return math.ceil(360 / stride)
        elif self.crop_strategy == "cubemap":
            return 4
        elif self.crop_strategy == "resize":
            return 1
        elif self.crop_strategy == "anyres":
            return 1 + self.anyres_max_patches
        elif self.crop_strategy == "anyres_e2p":
            return 1 + self.anyres_max_patches
        return 1

    def __call__(self, x: Union[str, Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Input: Single Image or List of Images.
        Output: [B, V, C, H, W] tensor.
        """
        # Internal helper to process a single image
        def process_one(img_input):
            pil = self._to_pil(img_input)
            match self.crop_strategy:
                case "sliding_window": return self._sliding(pil)
                case "e2p":           return self._e2p(pil)
                case "cubemap":       return self._cubemap4(pil)
                case "resize":        return self._resize(pil)
                case "anyres":        return self._anyres(pil)
                case "anyres_e2p":    return self._anyres_e2p(pil)
                case _:                raise ValueError(f"Unknown strategy {self.crop_strategy}")

        # Check if input is a list of images (batch)
        if isinstance(x, list):
             tensors = [process_one(img) for img in x]
             return torch.stack(tensors) # [B, V, C, H, W]
        else:
             # Single image, add batch dim
             return process_one(x).unsqueeze(0) # [1, V, C, H, W]

    def _to_pil(self, x) -> Image.Image:
        if isinstance(x, Image.Image): return x.convert("RGB")
        if isinstance(x, str):
            if x.startswith("http"):
                resp = requests.get(x)
                return Image.open(BytesIO(resp.content)).convert("RGB")
            return Image.open(x).convert("RGB")
        raise ValueError(f"Unsupported image type: {type(x)}")

    def _resize(self, img: Image.Image) -> torch.Tensor:
        resized = img.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
        return self.to_tensor(resized).unsqueeze(0) # [1, C, H, W]

    def _sliding(self, img: Image.Image) -> torch.Tensor:
        W, H = img.size
        views = []
        stride = int((self.fov_deg / 360) * W * (1 - self.overlap_ratio))
        vw = int((self.fov_deg / 360) * W)
        
        for i in range(self.num_views):
            s = i * stride
            # Very simple sliding window for brevity (full logic in pano_llava code if needed)
            if s + vw <= W:
                patch = img.crop((s, 0, s + vw, H))
            else:
                # Wrap around
                patch = Image.new("RGB", (vw, H))
                w1 = W - s
                patch.paste(img.crop((s, 0, W, H)), (0, 0))
                patch.paste(img.crop((0, 0, vw - w1, H)), (w1, 0))
            
            patch = patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
            views.append(self.to_tensor(patch))
        
        # Ensure we match expected num_views (pad if needed)
        while len(views) < self.num_views:
             views.append(views[-1])
        return torch.stack(views[:self.num_views])

    def _e2p(self, img: Image.Image) -> torch.Tensor:
        img_arr = np.array(img)
        stride = self.fov_deg * (1 - self.overlap_ratio)
        views = []
        for i in range(self.num_views):
            yaw = (i * stride) % 360
            if yaw > 180: yaw -= 360
            try:
                npv = e2p(img_arr, fov_deg=self.fov_deg, u_deg=float(yaw), v_deg=0, 
                          out_hw=self.image_size, mode="bilinear")
                views.append(self.to_tensor(Image.fromarray(npv.astype(np.uint8))))
            except Exception:
                # Fallback to black image
                views.append(torch.zeros(3, *self.image_size))
        return torch.stack(views)

    def _cubemap4(self, img: Image.Image) -> torch.Tensor:
        img_arr = np.array(img)
        try:
            faces = e2c(img_arr, face_w=self.image_size[1], cube_format="dict")
            order = ["F", "R", "B", "L"]
            views = [self.to_tensor(Image.fromarray(faces[k].astype(np.uint8))) for k in order]
            return torch.stack(views)
        except Exception:
            return torch.zeros(4, 3, *self.image_size)

    def _anyres(self, img: Image.Image) -> torch.Tensor:
        # Simplified AnyRes: Global + Grid patches
        views = []
        # 1. Global
        views.append(self.to_tensor(img.resize(self.image_size[::-1], Image.Resampling.LANCZOS)))
        
        # 2. Grid (Simple regular grid for now)
        W, H = img.size
        # Try to fit standard grid
        # This is a simplified placeholder for the complex AnyRes logic
        # For full implementation, one would adapt the logic from processing_panovlm.py completely
        # Here we just implement basic grid split
        n_patches = self.anyres_max_patches
        cols = int(np.sqrt(n_patches * W / H))
        rows = math.ceil(n_patches / cols)
        
        pw, ph = W // cols, H // rows
        for r in range(rows):
            for c in range(cols):
                if len(views) >= 1 + n_patches: break
                patch = img.crop((c*pw, r*ph, (c+1)*pw, (r+1)*ph))
                views.append(self.to_tensor(patch.resize(self.image_size[::-1], Image.Resampling.LANCZOS)))
                
        # Pad
        target = 1 + n_patches
        while len(views) < target:
            views.append(torch.zeros(3, *self.image_size))
            
        return torch.stack(views[:target])

    def _anyres_e2p(self, img: Image.Image) -> torch.Tensor:
        """
        AnyRes with E2P projection logic.
        1. Global view (resized equirectangular)
        2. Local views (E2P projections distributed by anyres_max_patches)
        """
        views = []
        
        # 1. Global View (Resized Equirectangular)
        # Using _resize logic directly
        resized = img.resize(self.image_size[::-1], Image.Resampling.LANCZOS)
        views.append(self.to_tensor(resized))
        
        # 2. Local Views (E2P)
        img_arr = np.array(img)
        n_patches = self.anyres_max_patches
        stride = 360.0 / n_patches
        
        for i in range(n_patches):
            yaw = (i * stride) % 360
            if yaw > 180: yaw -= 360
            
            try:
                # Use fov_deg from config for local patches
                npv = e2p(img_arr, fov_deg=self.fov_deg, u_deg=float(yaw), v_deg=0, 
                          out_hw=self.image_size, mode="bilinear")
                views.append(self.to_tensor(Image.fromarray(npv.astype(np.uint8))))
            except Exception:
                # Fallback
                views.append(torch.zeros(3, *self.image_size))
                
        return torch.stack(views)
