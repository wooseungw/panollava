from transformers import AutoTokenizer

class TextTokenizer:
    def __init__(self, model_name, max_len=512):  # 기본값을 512로 줄임
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        if self.tok.pad_token is None:
            self.tok.pad_token=self.tok.eos_token or self.tok.bos_token
    def __call__(self, text):
        return self.tok(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )