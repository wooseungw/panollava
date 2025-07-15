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
                 crop_strategy: str = "e2p",       # sliding_window | e2p | cubemap
                 fov_deg: float = 90.0,
                 overlap_ratio: float = 0.5,
                 normalize: bool = False):
        self.image_size, self.crop_strategy = image_size, crop_strategy
        self.fov_deg, self.overlap_ratio = fov_deg, overlap_ratio
        if crop_strategy in {"sliding_window","e2p"}:
            stride = fov_deg * (1-overlap_ratio)
            self.num_views = math.ceil(360/stride)
        elif crop_strategy == "cubemap":
            self.num_views = 4
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