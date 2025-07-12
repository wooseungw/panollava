from transformers import AutoTokenizer

class TextTokenizer:
    def __init__(self, model_name:str):
        self.tok=AutoTokenizer.from_pretrained(model_name,use_fast=None)
        if self.tok.pad_token is None:
            self.tok.pad_token=self.tok.eos_token or self.tok.bos_token
    def __call__(self,texts,**kw):
        return self.tok(texts,return_tensors="pt",**kw)