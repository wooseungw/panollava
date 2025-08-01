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
        """VLM 표준 패턴을 따르는 데이터 로딩 - 수정된 마스킹 로직"""
        try:
            row = self.df.iloc[idx]
            
            # --- 이미지 로드 및 검증
            try:
                pil = Image.open(row.url).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {row.url}: {e}")
                return self.__getitem__((idx + 1) % len(self))
            
            # --- 대화 템플릿 구성
            builder = ConversationPromptBuilder(self.tokenizer, system_msg=self.system_msg)
            builder.push("user", str(row.query))
            builder.push("assistant", str(row.annotation))
            
            # --- 멀티모달 처리
            batch = self.proc(pil, builder)
            
            # --- 올바른 라벨 마스킹
            IGNORE_INDEX = -100
            labels = batch["input_ids"].clone()
            
            # 디버깅용 정보 출력
            formatted_text = builder.formatted()
            if idx == 0:
                print(f"[Dataset Debug] Pixel values shape: {batch['pixel_values'].shape}")
                print(f"[Dataset Debug] Formatted text: {formatted_text[:200]}...")
                print(f"[Dataset Debug] Input IDs shape: {batch['input_ids'].shape}")
                print(f"[Dataset Debug] Labels shape before masking: {labels.shape}")
                print(f"[Dataset Debug] Image path: {row.url}")
            
            # 다양한 LLM 모델의 assistant 토큰 패턴 지원
            assistant_patterns = [
                "<|im_start|>assistant",  # Qwen
                "### Assistant:",         # LLaMA/Vicuna
                "ASSISTANT:",            # 일반
                "<|assistant|>",         # ChatML
                "### Response:",         # Alpaca
                "[/INST]",              # Mistral
                "Assistant:",           # 기본
                "assistant:",           # 소문자
            ]
            
            assistant_start_pos = -1
            found_pattern = None
            
            # Assistant 패턴 찾기
            for pattern in assistant_patterns:
                pos = formatted_text.find(pattern)
                if pos != -1:
                    assistant_start_pos = pos
                    found_pattern = pattern
                    break
            
            if assistant_start_pos != -1 and found_pattern:
                # 1단계: 전체를 먼저 ignore로 마스킹
                labels.fill_(IGNORE_INDEX)
                
                # 2단계: Assistant 응답 부분만 실제 토큰 ID로 복원
                assistant_start_text = formatted_text[assistant_start_pos:]
                
                # Assistant 토큰 이후의 실제 응답 시작점 찾기
                response_start = assistant_start_text.find(found_pattern) + len(found_pattern)
                
                # 응답 끝점 찾기 (다음 특수 토큰 또는 텍스트 끝)
                end_markers = ["<|im_end|>", "</s>", "[/INST]", "\n\n", "<|endoftext|>"]
                response_end = len(assistant_start_text)
                
                for marker in end_markers:
                    marker_pos = assistant_start_text.find(marker, response_start)
                    if marker_pos != -1:
                        response_end = marker_pos
                        break
                
                # 실제 응답 텍스트 추출
                actual_response = assistant_start_text[response_start:response_end].strip()
                
                if actual_response:
                    # 전체 텍스트에서 Assistant 응답의 실제 시작 위치
                    full_response_start = assistant_start_pos + response_start
                    full_response_text = formatted_text[:full_response_start] + actual_response
                    
                    try:
                        # 응답 시작 지점까지의 토큰 수 계산
                        prefix_tokens = self.tokenizer(
                            formatted_text[:full_response_start], 
                            add_special_tokens=False
                        )["input_ids"]
                        prefix_len = len(prefix_tokens)
                        
                        # 응답 부분의 토큰 수 계산
                        response_tokens = self.tokenizer(
                            actual_response, 
                            add_special_tokens=False
                        )["input_ids"]
                        response_len = len(response_tokens)
                        
                        # Assistant 응답 부분만 실제 토큰 ID로 복원
                        end_pos = min(prefix_len + response_len, labels.size(-1))
                        
                        if labels.dim() > 1:
                            # 원본 input_ids에서 해당 부분 복사
                            labels[0, prefix_len:end_pos] = batch["input_ids"][0, prefix_len:end_pos]
                        else:
                            labels[prefix_len:end_pos] = batch["input_ids"][prefix_len:end_pos]
                        
                        # 검증: 유효한 레이블 개수 확인
                        valid_labels = (labels != IGNORE_INDEX).sum()
                        total_labels = labels.numel()
                        
                        if idx == 0:
                            print(f"[Dataset Debug] Found pattern: '{found_pattern}' at position: {assistant_start_pos}")
                            print(f"[Dataset Debug] Response text: '{actual_response[:50]}...'")
                            print(f"[Dataset Debug] Prefix length: {prefix_len}, Response length: {response_len}")
                            print(f"[Dataset Debug] Valid labels: {valid_labels.item()}/{total_labels} ({valid_labels.item()/total_labels*100:.1f}%)")
                            
                    except Exception as e:
                        logger.warning(f"Failed to process assistant response: {e}")
                        # 실패 시 전체 마스킹
                        labels.fill_(IGNORE_INDEX)
                        if idx == 0:
                            print(f"[Dataset Debug] Response processing failed, masking all labels")
                else:
                    # 빈 응답인 경우
                    labels.fill_(IGNORE_INDEX)
                    if idx == 0:
                        print(f"[Dataset Debug] Empty assistant response, masking all labels")
            else:
                # Assistant 패턴을 찾지 못한 경우
                labels.fill_(IGNORE_INDEX)
                if idx == 0:
                    print(f"[Dataset Debug] No assistant pattern found, masking all labels")
                    print(f"[Dataset Debug] Searched patterns: {assistant_patterns[:3]}...")
            
            batch["labels"] = labels
            
            # --- 차원 정규화 (배치 차원 제거)
            for key in ["input_ids", "attention_mask", "labels"]:
                if key in batch and batch[key].dim() > 1:
                    batch[key] = batch[key].squeeze(0)
            
            # --- 메타데이터 추가
            batch.update({
                "input_text": builder.formatted(),
                "image_path": str(row.url),
                "sample_id": idx,
            })
            
            return batch
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
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
                 eval_mode=False,
                 system_msg=None):
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
            self.processor = PanoLLaVAProcessor(img_proc, txt_tok, max_length=max_txt_len)
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
                                                  self.processor, self.tokenizer,
                                                  system_msg=self.hparams.system_msg)
                    logger.info(f"Evaluation dataset loaded - Val: {len(self.val_ds)}")
                    # Training dataset은 None으로 설정 (evaluation에서는 사용하지 않음)
                    self.train_ds = None
                else:
                    # Training 모드: 정상적인 학습 데이터셋 로드
                    self.train_ds = ChatPanoDataset(self.hparams.csv_train,
                                                    self.processor, self.tokenizer,
                                                    system_msg=self.hparams.system_msg)
                    self.val_ds   = ChatPanoDataset(self.hparams.csv_val,
                                                    self.processor, self.tokenizer,
                                                    system_msg=self.hparams.system_msg)
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