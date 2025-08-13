from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BatchEncoding, default_data_collator, AutoTokenizer
from typing import Optional
import pandas as pd
from .processors.pano_llava_processor import PanoLLaVAProcessor
from .processors.universal_text_formatter import UniversalTextFormatter
from .processors.image import PanoramaImageProcessor
from .processors.vision import VisionProcessorWrapper
import torch

from .utils import memory_monitor, get_gpu_memory_info, auto_adjust_batch_size
import lightning as pl  # Lightning v2 호환성을 위해 변경
import psutil
import logging
logger = logging.getLogger(__name__)
import os

class BaseChatPanoDataset(Dataset):
    """파노라마 채팅 데이터셋의 기본 클래스 - 공통 기능을 통합"""
    def __init__(
        self,
        csv_path: str,
        processor: PanoLLaVAProcessor,
        tokenizer: AutoTokenizer,
        system_msg: str | None = "You are an expert assistant specialized in analyzing panoramic images. Please provide detailed, accurate, and helpful responses about what you observe in the panoramic view shortly.",
    ):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.proc = processor
        self.tokenizer = tokenizer
        self.system_msg = system_msg
        self.user_template = "In this panoramic image, please provide a concise but detailed description of \"<subject>\"."
        # CSV 컬럼 확인
        self.has_annotation = 'annotation' in self.df.columns
        
        # UniversalTextFormatter 초기화 (ConversationPromptBuilder 대신 사용)
        self.text_formatter = UniversalTextFormatter(
            tokenizer_name_or_path=tokenizer.name_or_path,
            system_msg=system_msg
        )
        
        logger.info(f"Dataset loaded: {len(self.df)} samples")
        logger.info(f"CSV columns: {list(self.df.columns)}")
        logger.info(f"Has annotation: {self.has_annotation}")
        logger.info(f"Using UniversalTextFormatter for {self.text_formatter.model_family} model")

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_path: str, idx: int) -> Image.Image:
        """이미지 로드 및 에러 처리"""
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # 순환 참조 방지를 위한 재시도
            return self._load_image(self.df.iloc[(idx + 1) % len(self)].url, (idx + 1) % len(self))

    def _format_user_query(self, query: str) -> str:
        """사용자 질문을 템플릿으로 포맷팅"""
        return self.user_template.replace("<subject>", query)

    def _create_formatted_conversation(self, user_query: str, assistant_response: str = None, add_generation_prompt: bool = False) -> str:
        """대화 포맷팅 - UniversalTextFormatter 사용"""
        return self.text_formatter.format_conversation(
            user_msg=user_query,
            assistant_msg=assistant_response,
            add_generation_prompt=add_generation_prompt,
            tokenizer=self.tokenizer
        )

    def _process_batch_with_text(self, pil: Image.Image, user_query: str, assistant_response: str = None) -> dict:
        """프로세서를 통한 배치 처리 - UniversalTextFormatter 직접 사용"""
        # 1. 이미지 처리
        pv5d = self.proc.img_proc(pil)  # (V,C,H,W)
        pv = pv5d.reshape(-1, *pv5d.shape[-3:])  # flatten
        
        # 2. 텍스트 처리 (UniversalTextFormatter 사용)
        if assistant_response is not None:
            # 훈련용: 전체 대화
            text_result = self.text_formatter.tokenize_for_training(
                user_msg=user_query,
                assistant_msg=assistant_response,
                tokenizer=self.tokenizer,
                max_length=self.proc.max_length
            )
        else:
            # 생성용: 사용자 쿼리만
            text_result = self.text_formatter.tokenize_for_generation(
                user_msg=user_query,
                tokenizer=self.tokenizer,
                max_length=self.proc.max_length
            )
        
        # 3. 결과 조합
        batch = {
            "pixel_values": pv,
            "input_ids": text_result["input_ids"],
            "attention_mask": text_result["attention_mask"],
            "formatted_text": text_result["formatted_text"]
        }
        
        if "labels" in text_result:
            batch["labels"] = text_result["labels"]
        
        return batch

    def _add_metadata(self, batch: dict, row: pd.Series, idx: int):
        """메타데이터 추가"""
        batch.update({
            "input_text": batch.get("formatted_text", ""),
            "image_path": str(row.url),
            "sample_id": idx,
        })

    # -------- 공통 전처리/검증 유틸 (Train/Test 공용) --------
    def _apply_common_text_preprocessing(
        self,
        batch: dict,
        idx: int,
        add_eos: bool = False,
        max_length: int | None = None,
        has_labels: bool = False,
    ) -> dict:
        """공통 텍스트 전처리: 패딩 마스크, 길이 제한, (선택) EOS 추가.
        생성(Test)에는 add_eos=False 권장.
        """
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None and "input_ids" in batch:
            # 어텐션 마스크 생성/업데이트
            attention_mask = (batch["input_ids"] != pad_token_id).long()
            batch["attention_mask"] = attention_mask

        # 길이 제한
        if max_length is None:
            max_length = self.proc.max_length
        for key in ["input_ids", "attention_mask"] + (["labels"] if has_labels and "labels" in batch else []):
            if key in batch and hasattr(batch[key], 'size') and batch[key].size(-1) > max_length:
                batch[key] = batch[key][..., :max_length]
                if idx == 0:
                    print(f"[Dataset Debug] (common) Truncated {key} to max_length: {max_length}")

        # EOS 추가 (생성에는 권장하지 않음)
        if add_eos and "input_ids" in batch:
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None:
                last_token = batch["input_ids"][-1] if batch["input_ids"].dim() == 1 else batch["input_ids"][0, -1]
                if last_token != eos_token_id:
                    if batch["input_ids"].dim() == 1:
                        batch["input_ids"] = torch.cat([batch["input_ids"], torch.tensor([eos_token_id], dtype=batch["input_ids"].dtype)])
                        if "attention_mask" in batch:
                            batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.tensor([1], dtype=batch["attention_mask"].dtype)])
                        if has_labels and "labels" in batch:
                            batch["labels"] = torch.cat([batch["labels"], torch.tensor([eos_token_id], dtype=batch["labels"].dtype)])
                    else:
                        batch["input_ids"] = torch.cat([batch["input_ids"], torch.tensor([[eos_token_id]], dtype=batch["input_ids"].dtype)], dim=-1)
                        if "attention_mask" in batch:
                            batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.tensor([[1]], dtype=batch["attention_mask"].dtype)], dim=-1)
                        if has_labels and "labels" in batch:
                            batch["labels"] = torch.cat([batch["labels"], torch.tensor([[eos_token_id]], dtype=batch["labels"].dtype)], dim=-1)
                    if idx == 0:
                        print(f"[Dataset Debug] (common) Added EOS token at the end of sequence")

        return batch

    def _validate_common(self, batch: dict, idx: int, has_labels: bool = False) -> bool:
        """공통 일관성 검증 (생성/학습 겸용)."""
        try:
            if "input_ids" not in batch:
                return True
            seq_len = batch["input_ids"].size(-1)
            for key in ["attention_mask"] + (["labels"] if has_labels and "labels" in batch else []):
                if key in batch and batch[key].size(-1) != seq_len:
                    if idx == 0:
                        print(f"[Dataset Debug] (common) Error: {key} length mismatch with input_ids")
                    return False
            if "attention_mask" in batch and batch["attention_mask"].sum() == 0:
                if idx == 0:
                    print(f"[Dataset Debug] (common) Error: No valid attention positions")
                return False
            return True
        except Exception as e:
            if idx == 0:
                print(f"[Dataset Debug] (common) Validation error: {e}")
            return False

class ChatPanoTestDataset(BaseChatPanoDataset):
    pass  # Deprecated: 통합된 ChatPanoDataset(mode='eval')를 사용하세요.

# ===================== dataset.chat_pano ========================
class ChatPanoDataset(BaseChatPanoDataset):
    """CSV (url,query,annotation) -> BatchEncoding via `PanoLLaVAProcessor`.
    
    간소화된 VLM 데이터셋:
    1. UniversalTextFormatter 사용으로 다양한 LLM 모델 지원
    2. 표준 HuggingFace 라벨 처리 방식
    3. 단순하고 안정적인 텍스트 포맷팅
    4. 복잡한 패턴 매칭 제거
    """
    
    IGNORE_INDEX = -100

    def __init__(self,
                 csv_path: str,
                 processor: PanoLLaVAProcessor,
                 tokenizer: AutoTokenizer,
                 system_msg: str | None = "You are an expert assistant specialized in analyzing panoramic images. Please provide detailed, accurate, and helpful responses about what you observe in the panoramic view shortly.",
                 mode: str = "train",  # 'train' | 'eval'
                 include_reference: bool = True):
        super().__init__(csv_path, processor, tokenizer, system_msg)
        assert mode in ("train", "eval"), "mode must be 'train' or 'eval'"
        self.mode = mode
        self.include_reference = include_reference
        if self.mode == "eval" and self.include_reference and not self.has_annotation:
            logger.warning("include_reference=True but 'annotation' column not found in CSV")
        
        logger.info(f"ChatPanoDataset initialized in mode='{self.mode}', include_reference={self.include_reference}")

    def _create_labels_with_formatter(self, user_query: str, assistant_response: str, idx: int) -> dict:
        """UniversalTextFormatter를 사용한 간소화된 라벨 생성"""
        try:
            # UniversalTextFormatter로 토큰화 및 라벨 생성
            result = self.text_formatter.tokenize_for_training(
                user_msg=user_query,
                assistant_msg=assistant_response,
                tokenizer=self.tokenizer,
                max_length=self.proc.max_length
            )
            
            if idx == 0:
                valid_labels = (result["labels"] != self.IGNORE_INDEX).sum()
                total_labels = result["labels"].numel()
                print(f"[Dataset Debug] Formatted text: {result['formatted_text'][:]}")
                print(f"[Dataset Debug] Input IDs shape: {result['input_ids'].shape}")
                print(f"[Dataset Debug] Valid labels: {valid_labels.item()}/{total_labels} ({valid_labels.item()/total_labels*100:.1f}%)")
                print(f"[Dataset Debug] Using UniversalTextFormatter for {self.text_formatter.model_family}")
            
            return result
            
        except Exception as e:
            logger.error(f"UniversalTextFormatter failed: {e}, falling back to simple processing")
            # Fallback: 단순 처리
            return self._create_simple_labels(user_query, assistant_response)
    
    def _create_simple_labels(self, user_query: str, assistant_response: str) -> dict:
        """Fallback: 단순한 라벨 생성"""
        # 간단한 포맷: "User: {query}\n\nAssistant: {response}"
        formatted_text = f"User: {user_query}\n\nAssistant: {assistant_response}"
        
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.proc.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding.get("attention_mask", 
                                    (input_ids != self.tokenizer.pad_token_id).long()).squeeze(0)
        
        # 라벨: Assistant 응답 부분만 학습
        labels = input_ids.clone()
        
        # "Assistant: " 이후 부분만 학습 대상으로 설정
        try:
            assistant_start_tokens = self.tokenizer("Assistant: ", add_special_tokens=False)["input_ids"]
            # 간단한 패턴 매칭으로 Assistant 시작 위치 찾기
            for i in range(len(input_ids) - len(assistant_start_tokens) + 1):
                if input_ids[i:i+len(assistant_start_tokens)].tolist() == assistant_start_tokens:
                    labels[:i+len(assistant_start_tokens)] = self.IGNORE_INDEX
                    break
            else:
                # 패턴을 찾지 못한 경우 절반 지점부터 학습 (rough estimate)
                midpoint = len(input_ids) // 2
                labels[:midpoint] = self.IGNORE_INDEX
        except:
            # 에러 발생 시 절반 지점부터 학습
            midpoint = len(input_ids) // 2
            labels[:midpoint] = self.IGNORE_INDEX
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "formatted_text": formatted_text
        }


    def __getitem__(self, idx):
        """간소화된 VLM 데이터 로딩 - UniversalTextFormatter 직접 사용"""
        try:
            row = self.df.iloc[idx]
            pil = self._load_image(row.url, idx)
            
            user_query = self._format_user_query(str(row.query))
            
            if self.mode == "train":
                # 학습 모드: 전체 대화
                assistant_response = str(row.annotation)
                batch = self._process_batch_with_text(pil, user_query, assistant_response)
            else:
                # 평가/생성 모드: 사용자 쿼리만
                batch = self._process_batch_with_text(pil, user_query, None)
                
                # 참조 정답 추가 (있는 경우)
                if self.include_reference and self.has_annotation:
                    batch["reference"] = str(row.annotation)
            
            # 메타데이터 추가
            self._add_metadata(batch, row, idx)
            
            return batch
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # 다음 샘플로 fallback
            return self.__getitem__((idx + 1) % len(self))
    
    def _validate_simple_batch(self, batch: dict, idx: int) -> bool:
        """간소화된 배치 검증"""
        try:
            # 기본 텐서 존재 확인
            required_keys = ["pixel_values", "input_ids", "attention_mask"]
            for key in required_keys:
                if key not in batch:
                    if idx == 0:
                        print(f"[Dataset Debug] Missing key: {key}")
                    return False
            
            # 시퀀스 길이 일치 확인
            seq_len = batch["input_ids"].size(-1)
            if "attention_mask" in batch and batch["attention_mask"].size(-1) != seq_len:
                if idx == 0:
                    print(f"[Dataset Debug] Attention mask length mismatch")
                return False
            
            if "labels" in batch and batch["labels"] is not None:
                if batch["labels"].size(-1) != seq_len:
                    if idx == 0:
                        print(f"[Dataset Debug] Labels length mismatch")
                    return False
                
                # 유효한 라벨 확인
                valid_labels = (batch["labels"] != self.IGNORE_INDEX).sum()
                if valid_labels == 0:
                    if idx == 0:
                        print(f"[Dataset Debug] No valid labels found")
                    return False
            
            return True
            
        except Exception as e:
            if idx == 0:
                print(f"[Dataset Debug] Validation error: {e}")
            return False

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
    """배치 처리를 위한 커스텀 collate 함수 (패딩 지원)"""
    from torch.nn.utils.rnn import pad_sequence
    
    # 텐서 키와 문자열 키 분리
    tensor_keys = ['input_ids', 'attention_mask', 'labels']  
    image_keys = ['pixel_values']
    string_keys = ['image_path', 'input_text', 'reference']
    
    tensor_batch = {}
    
    # 텍스트 텐서 데이터 처리 (패딩 적용)
    for key in tensor_keys:
        if key in batch[0] and batch[0][key] is not None:
            tensors = [item[key] for item in batch if key in item and item[key] is not None]
            if tensors:
                try:
                    # 1차원 텐서들을 동일한 길이로 패딩
                    if all(len(t.shape) == 1 for t in tensors):
                        # padding_value는 토크나이저의 pad_token_id 또는 -100 (labels)
                        pad_value = -100 if key == 'labels' else 0
                        padded = pad_sequence(tensors, batch_first=True, padding_value=pad_value)
                        tensor_batch[key] = padded
                    else:
                        # 다차원 텐서는 기존 방식 사용
                        tensor_batch[key] = default_data_collator([{key: t} for t in tensors])[key]
                except Exception as e:
                    print(f"Error collating tensor key {key}: {e}")
                    # Fallback: 리스트로 보관
                    tensor_batch[key] = tensors
            else:
                tensor_batch[key] = None
    
    # 이미지 데이터 처리 (pixel_values)
    for key in image_keys:
        if key in batch[0] and batch[0][key] is not None:
            tensors = [item[key] for item in batch if key in item and item[key] is not None]
            if tensors:
                try:
                    # 이미지 데이터는 일반적으로 동일한 크기여야 함
                    tensor_batch[key] = torch.stack(tensors, dim=0)
                except Exception as e:
                    print(f"Error collating image key {key}: {e}")
                    tensor_batch[key] = tensors
            else:
                tensor_batch[key] = None
    
    # 문자열/기타 데이터 처리
    for key in string_keys:
        if key in batch[0]:
            tensor_batch[key] = [item[key] for item in batch if key in item]
    
    # 추가 키들 처리 (sample_id 등)
    all_processed_keys = tensor_keys + image_keys + string_keys
    for key in batch[0].keys():
        if key not in all_processed_keys:
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
                 tokenizer_name="Qwen/Qwen3-0.6B", max_text_length=256, 
                 collate_fn=custom_collate_fn,
                 eval_mode=False,
                 system_msg=None,
                 overlap_ratio=0.5):
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
                                              crop_strategy=crop_strategy,
                                              overlap_ratio=overlap_ratio)
            # TextTokenizer 대신 AutoTokenizer 직접 사용
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.bos_token
                
            self.processor = PanoLLaVAProcessor(img_proc, max_length=max_text_length)
            logger.info(f"Data processors initialized successfully")
            logger.info(f"Tokenizer max_length: {max_text_length}")
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
                    # Evaluation 모드: 통합 데이터셋을 eval 모드로 사용
                    self.val_ds = ChatPanoDataset(self.hparams.csv_val,
                                                  self.processor, self.tokenizer,
                                                  system_msg=self.hparams.system_msg,
                                                  mode="eval",
                                                  include_reference=True)
                    logger.info(f"Evaluation dataset loaded - Val: {len(self.val_ds)}")
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
