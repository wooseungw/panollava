from typing import Union
from PIL import Image
from transformers import BatchEncoding
from .image import PanoramaImageProcessor
from .text import TextTokenizer
from .builder import ConversationPromptBuilder
from .vision import VisionProcessorWrapper

class PanoLLaVAProcessor:
    """이미지+대화 텍스트 → BatchEncoding"""
    def __init__(self,img_proc:PanoramaImageProcessor, 
                 txt_tok:TextTokenizer, 
                 vis_proc:VisionProcessorWrapper|None=None,
                 ):
        self.img_proc, self.txt_tok, self.vis_proc = img_proc, txt_tok, vis_proc
    def __call__(self, pil:Image.Image, builder:ConversationPromptBuilder, flatten=True):
        pv5d=self.img_proc(pil)              # (V,C,H,W)
        pv=pv5d.reshape(-1,*pv5d.shape[-3:]) if flatten else pv5d
        enc=builder.tokenized()
        data={"pixel_values":pv,**enc}
        return BatchEncoding(data,tensor_type="pt")