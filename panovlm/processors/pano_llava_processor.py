from typing import Union
from PIL import Image
from transformers import BatchEncoding
from .image import PanoramaImageProcessor
from .vision import VisionProcessorWrapper

class PanoLLaVAProcessor:
    """이미지+텍스트 → BatchEncoding (UniversalTextFormatter와 통합)"""
    def __init__(self, img_proc: PanoramaImageProcessor, 
                 vis_proc: VisionProcessorWrapper | None = None,
                 max_length: int = 512):
        self.img_proc = img_proc
        self.vis_proc = vis_proc
        self.max_length = max_length
        
    def __call__(self, pil: Image.Image, formatted_text: str, tokenizer, flatten=True):
        """
        Args:
            pil: 입력 이미지
            formatted_text: UniversalTextFormatter가 포맷한 텍스트
            tokenizer: HuggingFace 토크나이저
            flatten: 이미지 차원을 평평하게 할지 여부
        """
        # 이미지 처리
        pv5d = self.img_proc(pil)  # (V,C,H,W)
        pv = pv5d.reshape(-1, *pv5d.shape[-3:]) if flatten else pv5d
        
        # 텍스트 토크나이징 (UniversalTextFormatter로 이미 포맷됨)
        enc = tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 결과 조합
        data = {"pixel_values": pv, **enc}
        return BatchEncoding(data,tensor_type="pt")