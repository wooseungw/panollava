from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BatchEncoding, default_data_collator, AutoTokenizer
from .processors.pano_llava_processor import PanoLLaVAProcessor
from .processors.builder import ConversationPromptBuilder
from .processors.image import PanoramaImageProcessor
from .processors.text import TextTokenizer
from .processors.vision import VisionProcessorWrapper


# ===================== dataset.chat_pano ========================
class ChatPanoDataset(Dataset):
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

# ===================== Lightning DataModule ====================
class ChatPanoDataModule:
    """PyTorch Lightning friendly DataModule."""
    def __init__(self, csv_train:str, csv_val:str, image_root:str,
                 batch_size:int=2, num_workers:int=4, **proc_kw):
        self.save_hyperparameters("csv_train","csv_val","image_root","batch_size","num_workers")
        # Processor 공유
        img_proc = PanoramaImageProcessor(**proc_kw.get("image", {}))
        txt_tok  = TextTokenizer(proc_kw.get("tokenizer", "Qwen/Qwen3-0.6B"))
        vis_proc = None
        self.processor = PanoLLaVAProcessor(img_proc, txt_tok, vis_proc)
        self.tokenizer = txt_tok.tok

    def setup(self, stage:str|None=None):
        self.train_ds = ChatPanoDataset(self.hparams.csv_train, self.hparams.image_root,
                                        self.processor, self.tokenizer)
        self.val_ds   = ChatPanoDataset(self.hparams.csv_val,   self.hparams.image_root,
                                        self.processor, self.tokenizer)
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers,
                          collate_fn=default_data_collator, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers,
                          collate_fn=default_data_collator, pin_memory=True)
    @staticmethod
    def custom_collate_fn(batch):
        # 텐서/배열은 default_data_collator로 처리
        tensor_batch = default_data_collator([{k:v for k,v in item.items() if not isinstance(v,str)} for item in batch])
        # string은 리스트로 따로 모음
        tensor_batch["input_text"] = [item["input_text"] for item in batch]
        return tensor_batch

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers,
                          collate_fn=self.custom_collate_fn, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers,
                          collate_fn=self.custom_collate_fn, pin_memory=True)

# ===================== HF Trainer DataLoader builder ==========
def build_hf_dataloaders(csv_train, csv_val, image_root,
                          batch_size=2, num_workers=4, **proc_kw):
    img_proc = PanoramaImageProcessor(**proc_kw.get("image", {}))
    txt_tok  = TextTokenizer(proc_kw.get("tokenizer", "Qwen/Qwen3-0.6B"),max_len=64)
    processor= PanoLLaVAProcessor(img_proc, txt_tok)
    token    = txt_tok.tok

    train_ds = ChatPanoDataset(csv_train, processor, token)
    val_ds   = ChatPanoDataset(csv_val, processor, token)

    def custom_collate_fn(batch):
        tensor_batch = default_data_collator([{k:v for k,v in item.items() if not isinstance(v,str)} for item in batch])
        tensor_batch["input_text"] = [item["input_text"] for item in batch]
        return tensor_batch

    return {
        "train_dataset": train_ds,
        "eval_dataset":  val_ds,
        "data_collator": custom_collate_fn,
    }