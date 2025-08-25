from transformers import AutoProcessor

class VisionProcessorWrapper:
    def __init__(self, model_name:str | None = "google/siglip-base-patch16-224"):
        self.proc=AutoProcessor.from_pretrained(model_name)
    def __call__(self,pil_list):
        return self.proc(images=pil_list,return_tensors="pt")["pixel_values"]