from transformers import AutoProcessor

class VisionProcessorWrapper:
    def __init__(self, model_name:str | None = "google/siglip-so400m-patch14-384"):
        self.proc=AutoProcessor.from_pretrained(model_name)
    def __call__(self,pil_list):
        return self.proc(images=pil_list,return_tensors="pt")["pixel_values"]
