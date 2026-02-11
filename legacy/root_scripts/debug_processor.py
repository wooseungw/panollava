from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor

model_id = "OpenGVLab/InternVL2_5-1B"
try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"Processor type: {type(processor)}")
    print(f"Has tokenizer: {hasattr(processor, 'tokenizer')}")
    print(f"Has image_processor: {hasattr(processor, 'image_processor')}")
except Exception as e:
    print(f"AutoProcessor failed: {e}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"Tokenizer type: {type(tokenizer)}")
except Exception as e:
    print(f"AutoTokenizer failed: {e}")

try:
    image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"ImageProcessor type: {type(image_processor)}")
except Exception as e:
    print(f"AutoImageProcessor failed: {e}")
