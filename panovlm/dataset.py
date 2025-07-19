from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BatchEncoding, default_data_collator, AutoTokenizer
from typing import Optional
from .processors.pano_llava_processor import PanoLLaVAProcessor
from .processors.builder import ConversationPromptBuilder
from .processors.image import PanoramaImageProcessor
from .processors.text import TextTokenizer
from .processors.vision import VisionProcessorWrapper
import torch

from .utils import memory_monitor, get_gpu_memory_info, auto_adjust_batch_size
import lightning as pl  # Lightning v2 호환성을 위해 변경
import psutil
import logging
logger = logging.getLogger(__name__)
import os

class ChatPanoTestDataset(Dataset):
    """generate 테스트용: 이미지와 쿼리(텍스트)만 받아서 모델 입력에 맞게 반환."""
    def __init__(
        self,
        csv_path: str,
        processor: PanoLLaVAProcessor,
        tokenizer: AutoTokenizer,
        system_msg: str | None = "You are a helpful assistant.",
        
    ):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.proc = processor
        self.tokenizer = tokenizer
        self.system_msg = system_msg
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pil = Image.open(row.url).convert("RGB")
        builder = ConversationPromptBuilder(self.tokenizer, system_msg=self.system_msg)
        builder.push("user", str(row.query))
        # Processor 호출: annotation 없이 user 쿼리만
        batch = self.proc(pil, builder)
        batch["input_ids"] = batch["input_ids"].squeeze(0)
        batch["attention_mask"] = batch["attention_mask"].squeeze(0)
        batch["input_text"] = self.tokenizer.decode(batch["input_ids"].tolist(), skip_special_tokens=True)
        batch["image_path"] = str(row.url)  # 문자열로 변환
        return batch

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
        
    ):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        
        self.proc = processor
        self.tokenizer = tokenizer
        self.system_msg = system_msg


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        """VLM 표준 패턴을 따르는 데이터 로딩"""
        try:
            row = self.df.iloc[idx]
            
            # --- 이미지 로드 및 검증
            try:
                pil = Image.open(row.url).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {row.url}: {e}")
                # 폴백: 다음 인덱스 시도
                return self.__getitem__((idx + 1) % len(self))
            
            # --- 대화 템플릿 구성 (LLaVA 스타일)
            builder = ConversationPromptBuilder(self.tokenizer, system_msg=self.system_msg)
            builder.push("user", str(row.query))
            
            # 학습 모드에서만 assistant 응답 추가
            
            builder.push("assistant", str(row.annotation))
            
            # --- 멀티모달 처리
            batch = self.proc(pil, builder)
            
            # --- 라벨 마스킹 (표준 VLM 패턴)
            IGNORE_INDEX = -100
            labels = batch["input_ids"].clone()
            
            # 디버깅용 정보 출력
            formatted_text = builder.formatted()
            print(f"[Dataset Debug] Formatted text: {formatted_text[:200]}...")
            print(f"[Dataset Debug] Input IDs shape: {batch['input_ids'].shape}")
            print(f"[Dataset Debug] Labels shape before masking: {labels.shape}")
            
            # 사용자 입력 부분만 마스킹, assistant 응답은 학습에 사용
            if "assistant" in formatted_text.lower():
                # 대소문자 구분 없이 assistant 찾기
                if "Assistant:" in formatted_text:
                    split_token = "Assistant:"
                elif "assistant:" in formatted_text:
                    split_token = "assistant:"
                else:
                    # 다른 형태의 assistant 토큰 찾기
                    import re
                    match = re.search(r'assistant\s*:', formatted_text, re.IGNORECASE)
                    if match:
                        split_token = match.group()
                    else:
                        split_token = None
                
                if split_token:
                    user_part = formatted_text.split(split_token)[0] + split_token
                    user_tokens = self.tokenizer(user_part, add_special_tokens=False)["input_ids"]
                    user_len = len(user_tokens)
                    
                    # 배치 차원 고려한 마스킹
                    if labels.dim() > 1:
                        labels[0, :user_len] = IGNORE_INDEX
                    else:
                        labels[:user_len] = IGNORE_INDEX
                    
                    # Assistant 응답 부분은 학습에 사용 (ignore하지 않음)
                    valid_labels = (labels != IGNORE_INDEX).sum()
                    print(f"[Dataset Debug] User part length: {user_len}, Valid labels: {valid_labels.item()}")
                else:
                    # split_token을 찾지 못한 경우 전체 마스킹
                    labels.fill_(IGNORE_INDEX)
                    print(f"[Dataset Debug] No split token found, masking all labels")
            else:
                # Assistant 응답이 없으면 전체 마스킹
                labels.fill_(IGNORE_INDEX)
                print(f"[Dataset Debug] No assistant response found, masking all labels")
            
            batch["labels"] = labels
            
            # --- 차원 정규화 (배치 차원 제거)
            for key in ["input_ids", "attention_mask", "labels"]:
                if key in batch and batch[key].dim() > 1:
                    batch[key] = batch[key].squeeze(0)
            
            # --- 메타데이터 추가 (디버깅 및 분석용)
            batch.update({
                "input_text": builder.formatted(),  # 원본 대화 텍스트
                "image_path": str(row.url),         # 이미지 경로 (문자열로 변환)
                "sample_id": idx,                   # 샘플 ID
            })
            # print(f"Processed sample {idx}: {batch['input_text'][:50]}...")  # 디버깅용 출력 (주석 처리)
            return batch
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # 안전한 폴백: 다음 유효한 샘플 반환
            return self.__getitem__((idx + 1) % len(self))

# Safe collate function for DataLoader
def safe_collate_fn(batch):
    """배치 처리 시 안전한 collate 함수"""
    from transformers import default_data_collator
    
    def to_dict(x):
        if hasattr(x, 'to_dict'):
            return x.to_dict()
        elif isinstance(x, dict):
            return x
        else:
            return dict(x)
    
    return default_data_collator([to_dict(sample) for sample in batch])

def custom_collate_fn(batch):
    """배치 처리를 위한 커스텀 collate 함수"""
    # 텐서 키와 문자열 키 분리
    tensor_keys = ['input_ids', 'attention_mask', 'labels', 'pixel_values']
    string_keys = ['image_path', 'input_text']
    
    tensor_batch = {}
    
    # 텐서 데이터 처리
    for key in tensor_keys:
        if key in batch[0]:
            try:
                tensor_batch[key] = default_data_collator([{key: item[key]} for item in batch])[key]
            except Exception as e:
                print(f"Error collating tensor key {key}: {e}")
                # Fallback: 리스트로 보관
                tensor_batch[key] = [item[key] for item in batch if key in item]
    
    # 문자열/기타 데이터 처리
    for key in string_keys:
        if key in batch[0]:
            tensor_batch[key] = [item[key] for item in batch if key in item]
    
    # 추가 키들 처리 (sample_id 등)
    for key in batch[0].keys():
        if key not in tensor_keys and key not in string_keys:
            try:
                # 숫자 타입이면 텐서로 변환 시도
                if isinstance(batch[0][key], (int, float)):
                    tensor_batch[key] = torch.tensor([item[key] for item in batch if key in item])
                else:
                    # 문자열이나 기타 타입은 리스트로 보관
                    tensor_batch[key] = [item[key] for item in batch if key in item]
            except:
                # 에러 발생 시 리스트로 보관
                tensor_batch[key] = [item[key] for item in batch if key in item]
    
    return tensor_batch
# =============================================================================
# 1. DataModule
# =============================================================================
class VLMDataModule(pl.LightningDataModule):
    def __init__(self, csv_train, csv_val, batch_size=4, num_workers=4,
                 image_size=(224,224), crop_strategy="e2p",
                 tokenizer_name="Qwen/Qwen3-0.6B", max_txt_len=512, 
                 collate_fn=custom_collate_fn,
                 eval_mode=False):
        # Lightning v2에서 권장하는 명시적 super 호출
        super(VLMDataModule, self).__init__()
        
        # 모든 하이퍼파라미터 저장 (Lightning v2 호환성)
        self.save_hyperparameters(ignore=['collate_fn'])  # collate_fn은 pickle 불가능하므로 제외
        
        # collate_fn은 별도로 저장
        self.collate_fn = collate_fn
        
        # 메모리 기반 배치 크기 자동 조정
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        self.hparams.batch_size = auto_adjust_batch_size(batch_size, available_memory)
        
        # 배치 크기 정보 출력
        logger.info(f"=== BATCH SIZE CONFIGURATION ===")
        logger.info(f"Original batch size: {batch_size}")
        logger.info(f"Adjusted batch size: {self.hparams.batch_size}")
        logger.info(f"Available RAM: {available_memory:.1f}GB")
        
        # GPU 메모리 정보 출력
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            logger.info(f"GPU Memory: {gpu_info['free']:.1f}GB free / {gpu_info['total']:.1f}GB total")
        logger.info(f"================================")
        
        if self.hparams.batch_size != batch_size:
            logger.warning(f"BATCH SIZE ADJUSTED: {batch_size} -> {self.hparams.batch_size} (Available memory: {available_memory:.1f}GB)")
        
        try:
            img_proc = PanoramaImageProcessor(image_size=image_size,
                                              crop_strategy=crop_strategy)
            txt_tok  = TextTokenizer(tokenizer_name, max_len=max_txt_len,)
            self.processor = PanoLLaVAProcessor(img_proc, txt_tok, max_length=512)
            self.tokenizer = txt_tok.tok
            logger.info(f"Data processors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data processors: {e}")
            raise
    
    def prepare_data(self):
        """
        Lightning DataModule의 prepare_data 메서드
        데이터 다운로드나 전처리 작업을 수행
        이 메서드는 GPU 0에서만 실행되고, 분산 훈련에서 한 번만 호출됨
        """
        # CSV 파일 존재 확인
        if not Path(self.hparams.csv_train).exists():
            raise FileNotFoundError(f"Training CSV not found: {self.hparams.csv_train}")
        if not Path(self.hparams.csv_val).exists():
            raise FileNotFoundError(f"Validation CSV not found: {self.hparams.csv_val}")
        
        logger.info("Data files validated successfully")
    
    def setup(self, stage: Optional[str] = None):
        """
        Lightning DataModule의 setup 메서드
        stage: 'fit', 'validate', 'test', 'predict' 중 하나 (또는 None)
        """
        try:
            with memory_monitor():
                if self.hparams.eval_mode:
                    # Evaluation 모드: validation 데이터만 로드하고, generation 모드로 설정
                    self.val_ds = ChatPanoTestDataset(self.hparams.csv_val,
                                                  self.processor, self.tokenizer)
                    logger.info(f"Evaluation dataset loaded - Val: {len(self.val_ds)}")
                    # Training dataset은 None으로 설정 (evaluation에서는 사용하지 않음)
                    self.train_ds = None
                else:
                    # Training 모드: 정상적인 학습 데이터셋 로드
                    self.train_ds = ChatPanoDataset(self.hparams.csv_train,
                                                    self.processor, self.tokenizer)
                    self.val_ds   = ChatPanoDataset(self.hparams.csv_val,
                                                    self.processor, self.tokenizer)
                    logger.info(f"Training datasets loaded - Train: {len(self.train_ds)}, Val: {len(self.val_ds)}")
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise

    def train_dataloader(self):
        if self.train_ds is None:
            # eval_mode에서는 train_dataloader가 호출되지 않아야 하지만, 안전장치
            raise RuntimeError("Training dataset not available in evaluation mode")
        
        return DataLoader(
            self.train_ds, 
            batch_size=self.hparams.batch_size,
            shuffle=True, 
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,  # self.hparams.collate_fn 대신 self.collate_fn 사용
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            prefetch_factor=2 if self.hparams.num_workers > 0 else None
        )

    def val_dataloader(self):
        if self.val_ds is None:
            raise RuntimeError("Validation dataset not available")
        
        return DataLoader(
            self.val_ds, 
            batch_size=self.hparams.batch_size,
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,  # self.hparams.collate_fn 대신 self.collate_fn 사용
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            prefetch_factor=2 if self.hparams.num_workers > 0 else None
        )