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

# dataset.py에 추가할 평가용 데이터셋 클래스

class ChatPanoEvalDataset(Dataset):
    """평가 전용 데이터셋 - reference와 prediction을 명확히 분리"""
    
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
        """평가용 데이터 로딩 - reference와 query를 명확히 분리"""
        try:
            row = self.df.iloc[idx]
            
            # --- 이미지 로드 및 검증
            try:
                pil = Image.open(row.url).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {row.url}: {e}")
                return self.__getitem__((idx + 1) % len(self))
            
            # --- 쿼리만으로 대화 템플릿 구성 (답변은 모델이 생성)
            builder = ConversationPromptBuilder(self.tokenizer, system_msg=self.system_msg)
            builder.push("user", str(row.query))
            # 평가시에는 assistant 응답을 추가하지 않음
            
            # --- 멀티모달 처리 (쿼리만)
            batch = self.proc(pil, builder)
            
            # --- 차원 정규화 (배치 차원 제거)
            for key in ["input_ids", "attention_mask"]:
                if key in batch and batch[key].dim() > 1:
                    batch[key] = batch[key].squeeze(0)
            
            # --- Ground truth와 메타데이터 추가
            batch.update({
                "reference": str(row.annotation),  # 정답 텍스트
                "query": str(row.query),           # 질문
                "input_text": builder.formatted(), # 전체 입력 텍스트
                "image_path": str(row.url),        # 이미지 경로
                "sample_id": idx,
            })
            
            return batch
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

# 평가용 collate 함수
def eval_collate_fn(batch):
    """평가용 배치 처리 함수"""
    from transformers import default_data_collator
    
    # 텐서 키와 문자열 키 분리
    tensor_keys = ['input_ids', 'attention_mask', 'pixel_values']
    string_keys = ['reference', 'query', 'input_text', 'image_path']
    
    tensor_batch = {}
    
    # 텐서 데이터 처리
    for key in tensor_keys:
        if key in batch[0]:
            try:
                tensor_batch[key] = default_data_collator([{key: item[key]} for item in batch])[key]
            except Exception as e:
                print(f"Error collating tensor key {key}: {e}")
                tensor_batch[key] = [item[key] for item in batch if key in item]
    
    # 문자열/기타 데이터 처리
    for key in string_keys:
        if key in batch[0]:
            tensor_batch[key] = [item[key] for item in batch if key in item]
    
    # 추가 키들 처리 (sample_id 등)
    for key in batch[0].keys():
        if key not in tensor_keys and key not in string_keys:
            try:
                if isinstance(batch[0][key], (int, float)):
                    tensor_batch[key] = torch.tensor([item[key] for item in batch if key in item])
                else:
                    tensor_batch[key] = [item[key] for item in batch if key in item]
            except:
                tensor_batch[key] = [item[key] for item in batch if key in item]
    
    return tensor_batch

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
# dataset.py의 VLMDataModule 클래스 수정
class VLMDataModule(pl.LightningDataModule):
    def __init__(self, csv_train, csv_val, batch_size=4, num_workers=4,
                 image_size=(224,224), crop_strategy="e2p",
                 tokenizer_name="Qwen/Qwen3-0.6B", max_txt_len=512, 
                 collate_fn=custom_collate_fn,
                 eval_mode=False):  # 평가 모드 추가
        # Lightning v2에서 권장하는 명시적 super 호출
        super(VLMDataModule, self).__init__()
        
        # 모든 하이퍼파라미터 저장 (Lightning v2 호환성)
        self.save_hyperparameters(ignore=['collate_fn'])
        
        # 평가 모드에 따라 다른 collate_fn 사용
        if eval_mode:
            self.collate_fn = eval_collate_fn
        else:
            self.collate_fn = collate_fn
        
        # 메모리 기반 배치 크기 자동 조정
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        self.hparams.batch_size = auto_adjust_batch_size(batch_size, available_memory)
        
        # ...existing code...
    
    def setup(self, stage: Optional[str] = None):
        """데이터셋 초기화"""
        processor, tokenizer = self._init_processor_and_tokenizer()
        self.tokenizer = tokenizer  # 평가에서 사용할 수 있도록 저장
        
        if self.hparams.eval_mode:
            # 평가 모드: ChatPanoEvalDataset 사용
            self.train_ds = ChatPanoEvalDataset(
                self.hparams.csv_train, processor, tokenizer, 
                system_msg="You are a helpful assistant."
            )
            self.val_ds = ChatPanoEvalDataset(
                self.hparams.csv_val, processor, tokenizer,
                system_msg="You are a helpful assistant."
            )
            logger.info(f"✓ Evaluation datasets loaded")
        else:
            # 학습 모드: ChatPanoDataset 사용
            self.train_ds = ChatPanoDataset(
                self.hparams.csv_train, processor, tokenizer,
                system_msg="You are a helpful assistant."
            )
            self.val_ds = ChatPanoDataset(
                self.hparams.csv_val, processor, tokenizer,
                system_msg="You are a helpful assistant."
            )
            logger.info(f"✓ Training datasets loaded")
        
        logger.info(f"Training samples: {len(self.train_ds)}")
        logger.info(f"Validation samples: {len(self.val_ds)}")