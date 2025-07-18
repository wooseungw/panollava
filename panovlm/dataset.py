from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BatchEncoding, default_data_collator, AutoTokenizer
from .processors.pano_llava_processor import PanoLLaVAProcessor
from .processors.builder import ConversationPromptBuilder
from .processors.image import PanoramaImageProcessor
from .processors.text import TextTokenizer
from .processors.vision import VisionProcessorWrapper

class PanoDataset(Dataset):
    """CSV (url,query,annotation) -> BatchEncoding via `PanoLLaVAProcessor`.
    학습용: user ↔ assistant 대화 + 파노라마 이미지를 한 행(row)으로 취급.
    `vis_proc`가 None 이면 CLIP 입력 없이 pixel_values 만 반환.
    """
    def __init__(
        self,
        csv_path: str,
        processor: PanoLLaVAProcessor,
        tokenizer: AutoTokenizer,                       # AutoTokenizer (builder용)
        system_msg: str | None = "You are a helpful assistant.",
        flatten: bool = True,
    ):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        
        self.proc = processor
        self.tokenizer = tokenizer
        self.system_msg = system_msg
        self.flatten = flatten

    def __len__(self):
        return len(self.df)

class ChatPanoTestDataset(PanoDataset):
    def __init__(self, csv_path, processor, tokenizer, system_msg = "You are a helpful assistant.", flatten = True):
        super().__init__(csv_path, processor, tokenizer, system_msg, flatten)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pil = Image.open(row.url).convert("RGB")
        builder = ConversationPromptBuilder(self.tokenizer, system_msg=self.system_msg)
        builder.push("user", str(row.query))
        # Processor 호출: annotation 없이 user 쿼리만
        batch = self.proc(pil, builder, flatten=self.flatten)
        batch["input_ids"] = batch["input_ids"].squeeze(0)
        batch["attention_mask"] = batch["attention_mask"].squeeze(0)
        batch["input_text"] = self.tokenizer.decode(batch["input_ids"].tolist(), skip_special_tokens=True)
        batch["image_path"] = row.url
        return batch

# ===================== dataset.chat_pano ========================
class ChatPanoDataset(PanoDataset):
    def __init__(self, csv_path, processor, tokenizer, system_msg = "You are a helpful assistant.", flatten = True):
        super().__init__(csv_path, processor, tokenizer, system_msg, flatten)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # --- 이미지 로드
        pil = Image.open(row.url).convert("RGB")
        # --- 프롬프트 빌드
        builder = ConversationPromptBuilder(self.tokenizer, system_msg=self.system_msg)
        builder.push("user",      str(row.query))    
        builder.push("assistant", str(row.annotation))
        # --- Processor 호출
        batch = self.proc(pil, builder, flatten=self.flatten)  # pixel_values + input_ids ...
        # --- assistant 토큰만 loss
        lbl = batch["input_ids"].clone()
        user_len = self.tokenizer(builder.formatted().split("Assistant:")[0], add_special_tokens=False)["input_ids"].__len__()
        lbl[0, :user_len] = -100
        batch["labels"] = lbl
        # collate 이후 3차원을 2차원으로 평탄화
        batch["input_ids"]     = batch["input_ids"].squeeze(0)       # (B,L)
        batch["attention_mask"] = batch["attention_mask"].squeeze(0)
        batch["labels"]        = batch["labels"].squeeze(0)
        # input_ids를 string으로 디코딩하여 추가
        batch["input_text"] = self.tokenizer.decode(batch["input_ids"].tolist(), skip_special_tokens=True)
        batch["image_path"] = row.url  # 이미지 경로 추가
        return batch
