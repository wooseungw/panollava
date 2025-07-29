import re
import ast
from typing import Tuple, Union, List
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
    """파노라마 → 멀티뷰 텐서 변환 (GPU autograd 호환)
    args:
        image_size: (H, W) 크기, 기본 (224, 224)
        crop_strategy: "sliding_window", "e2p", "cubemap", "resize", "anyres", "anyres_max"
            - sliding_window: 슬라이딩
            - e2p: E2P 방식 (Equiangular to Perspective)
            - cubemap: 4-face 큐브맵
            - resize: 단순 리사이즈 (1뷰)
            - anyres: LLaVA-NeXT 스타일 AnyRes (전체 이미지 + 적응적 패치 분할)
            - anyres_max: 더 공격적인 AnyRes (최대 패치 수)
        overlap_ratio: 슬라이딩 윈도우의 겹침 비율 (0.0~1.0)
        normalize: True면 [0,1] 범위로 정규화 (ImageNet 평균/표준편차 적용)
        anyres_patch_size: AnyRes 패치 크기 (기본 336)
        anyres_max_patches: AnyRes 최대 패치 수 (기본 12)
        anyres_image_grid_pinpoints: LLaVA-NeXT 스타일 그리드 핀 또는 문자열 형태
    """
    def __init__(self,
                 image_size: Tuple[int, int] = (224, 224),
                 crop_strategy: str = "e2p",       # sliding_window | e2p | cubemap | anyres | anyres_max
                 fov_deg: float = 90.0,
                 overlap_ratio: float = 0.5,
                 normalize: bool = False,
                 # AnyRes 관련 파라미터
                 anyres_patch_size: int = 336,     # 각 패치의 크기
                 anyres_max_patches: int = 4,     # 최대 패치 수
                 anyres_image_grid_pinpoints = None):
        self.image_size, self.crop_strategy = image_size, crop_strategy
        self.fov_deg, self.overlap_ratio = fov_deg, overlap_ratio
        
        # AnyRes 파라미터 초기화
        self.anyres_patch_size = anyres_patch_size
        self.anyres_max_patches = anyres_max_patches
        
        # 기본 grid pinpoints (LLaVA-NeXT 스타일) 또는 문자열 형태 지원
        if anyres_image_grid_pinpoints is None:
            self.anyres_image_grid_pinpoints = [
                (336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008),
                (672, 1008), (1008, 672), (336, 1344), (1344, 336), (1008, 1008)
            ]
        else:
            self.anyres_image_grid_pinpoints = anyres_image_grid_pinpoints
        
        # processor.size 속성 시뮬레이션 (anyres 알고리즘 호환성)
        self.size = {"shortest_edge": min(image_size)}
        self.crop_size = {"height": anyres_patch_size, "width": anyres_patch_size}
        
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
        LLaVA-NeXT 스타일의 AnyRes 구현 - 원본 알고리즘과 동일
        """
        return self.process_anyres_image(img, self, self.anyres_image_grid_pinpoints)
    
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

    # ===== LLaVA-NeXT AnyRes 알고리즘 구현 =====
    
    def process_anyres_image(self, image: Image.Image, processor, grid_pinpoints) -> torch.Tensor:
        """
        Process an image with variable resolutions.
        원본 LLaVA-NeXT 알고리즘과 동일한 구현

        Args:
            image (PIL.Image.Image): The input image to be processed.
            processor: The image processor object.
            grid_pinpoints: A string representation or list of possible resolutions.

        Returns:
            torch.Tensor: A tensor containing the processed image patches.
        """
        # Convert grid_pinpoints from string to list
        if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
            try:
                patch_size = processor.size["shortest_edge"]
            except Exception as e:
                patch_size = self.anyres_patch_size
            assert patch_size in [224, 336, 384], "patch_size should be in [224, 336, 384, 448, 512]"
            # Use regex to extract the range from the input string
            matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
            range_start = tuple(map(int, matches[0]))
            range_end = tuple(map(int, matches[-1]))
            # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
            grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
            # Multiply all elements by patch_size
            grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

        if type(grid_pinpoints) is list:
            possible_resolutions = grid_pinpoints
        else:
            possible_resolutions = ast.literal_eval(grid_pinpoints)
        
        best_resolution = self.select_best_resolution(image.size, possible_resolutions)
        image_padded = self.resize_and_pad_image(image, best_resolution)

        patches = self.divide_to_patches(image_padded, processor.crop_size["height"])

        # FIXME: this seems to be a bug that it resizes instead of pad.
        # but to keep it consistent with previous, i will keep it as it is
        # TODO: uncomment below to ablate with the padding
        if isinstance(processor.size, dict):
            shortest_edge = processor.size["shortest_edge"]
        else:
            shortest_edge = min(processor.size)
        image_original_resize = image.resize((shortest_edge, shortest_edge))
        # image_padded_square = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
        # image_original_resize = image_padded_square.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

        image_patches = [image_original_resize] + patches
        image_patches = [self.preprocess(image_patch) for image_patch in image_patches]
        
        # 최대 뷰 수 제한
        if len(image_patches) > self.num_views:
            image_patches = image_patches[:self.num_views]
        
        # 부족한 경우 패딩
        while len(image_patches) < self.num_views:
            if image_patches:
                image_patches.append(image_patches[-1].clone())
            else:
                # 빈 이미지로 패딩
                empty_img = Image.new('RGB', (shortest_edge, shortest_edge), (0, 0, 0))
                image_patches.append(self.preprocess(empty_img))
        
        return torch.stack(image_patches, dim=0)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """이미지 전처리 (tensor 변환)"""
        return self.to_tensor(image)

    def select_best_resolution(self, original_size: Tuple[int, int], possible_resolutions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Selects the best resolution from a list of possible resolutions based on the original size.

        Args:
            original_size (tuple): The original size of the image in the format (width, height).
            possible_resolutions (list): A list of possible resolutions in the format [(width, height), ...].

        Returns:
            tuple: The best fit resolution in the format (width, height).
        """
        original_width, original_height = original_size
        best_fit = None
        max_effective_resolution = 0
        min_wasted_resolution = float('inf')

        for width, height in possible_resolutions:
            # Calculate the downscaled size to fit within the possible resolution
            scale = min(width / original_width, height / original_height)
            downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

            # Calculate effective and wasted resolutions
            effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
            wasted_resolution = (width * height) - effective_resolution

            # Choose the best fit based on the criteria
            if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
                max_effective_resolution = effective_resolution
                min_wasted_resolution = wasted_resolution
                best_fit = (width, height)

        return best_fit

    def resize_and_pad_image(self, image: Image.Image, target_resolution: Tuple[int, int]) -> Image.Image:
        """
        Resize and pad an image to a target resolution while maintaining aspect ratio.

        Args:
            image (PIL.Image.Image): The input image.
            target_resolution (tuple): The target resolution (width, height) to resize and pad the image to.

        Returns:
            PIL.Image.Image: The resized and padded image.
        """
        original_width, original_height = image.size
        target_width, target_height = target_resolution

        # Determine which dimension (width or height) to fill
        scale_w = target_width / original_width
        scale_h = target_height / original_height

        if scale_w < scale_h:
            # Width will be filled completely
            new_width = target_width
            new_height = min(math.ceil(original_height * scale_w), target_height)
        else:
            # Height will be filled completely
            new_height = target_height
            new_width = min(math.ceil(original_width * scale_h), target_width)

        # Resize the image
        resized_image = image.resize((new_width, new_height))

        new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_image.paste(resized_image, (paste_x, paste_y))

        return new_image

    def divide_to_patches(self, image: Image.Image, patch_size: int) -> List[Image.Image]:
        """
        Divides an image into patches of a specified size.

        Args:
            image (PIL.Image.Image): The input image.
            patch_size (int): The size of each patch.

        Returns:
            list: A list of PIL.Image.Image objects representing the patches.
        """
        patches = []
        width, height = image.size
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                box = (j, i, j + patch_size, i + patch_size)
                patch = image.crop(box)
                patches.append(patch)

        return patches