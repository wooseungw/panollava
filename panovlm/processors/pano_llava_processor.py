from typing import Union
from PIL import Image
from transformers import BatchEncoding
from .image import PanoramaImageProcessor
from .text import TextTokenizer
from .vision import VisionProcessorWrapper
from .builder import ConversationPromptBuilder

class PanoLLaVAProcessor:
    """이미지+대화 텍스트 → BatchEncoding"""
    def __init__(self,img_proc:PanoramaImageProcessor, 
                 txt_tok:TextTokenizer, 
                 vis_proc:VisionProcessorWrapper|None=None,
                 max_length:int=512,  # 최대 시퀀스 길이 제한
                 ):
        self.img_proc, self.txt_tok, self.vis_proc = img_proc, txt_tok, vis_proc
        self.max_length = max_length
        
    def __call__(self, pil:Image.Image, builder:ConversationPromptBuilder, flatten=True):
        pv5d=self.img_proc(pil)              # (V,C,H,W)
        pv=pv5d.reshape(-1,*pv5d.shape[-3:]) if flatten else pv5d
        enc=builder.tokenized()
        
        # 텍스트 길이 제한 적용
        if self.max_length is not None:
            for key in ['input_ids', 'attention_mask', 'labels']:
                if key in enc and enc[key].shape[-1] > self.max_length:
                    enc[key] = enc[key][:, :self.max_length]
        
        data={"pixel_values":pv,**enc}
        return BatchEncoding(data,tensor_type="pt")