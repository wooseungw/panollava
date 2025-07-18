from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BatchEncoding, default_data_collator, AutoTokenizer
from .processors.pano_llava_processor import PanoLLaVAProcessor
from .processors.builder import ConversationPromptBuilder
from .processors.image import PanoramaImageProcessor
from .processors.text import TextTokenizer
from .processors.vision import VisionProcessorWrapper
import torch


class BasePanoDataset(Dataset):
    """파노라마 데이터셋의 베이스 클래스 - 공통 인자 및 메서드"""
    
    def __init__(
        self,
        csv_path: str,
        processor: PanoLLaVAProcessor,
        tokenizer: AutoTokenizer,
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
    
    def _load_image(self, image_path: str) -> Image.Image:
        """이미지 로드 공통 메서드"""
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 더미 이미지 반환 (검은색 224x224)
            return Image.new('RGB', (224, 224), color='black')
    
    def _normalize_batch_dimensions(self, batch: dict) -> dict:
        """배치 차원 정규화 공통 메서드"""
        for key in ["input_ids", "attention_mask"]:
            if key in batch:
                tensor = batch[key]
                if tensor.dim() == 3 and tensor.shape[0] == 1:  # (1, 1, L)
                    batch[key] = tensor.squeeze(0)  # (1, L)
                elif tensor.dim() == 2 and tensor.shape[0] == 1:  # (1, L)
                    # 이미 올바른 형태
                    pass
        return batch
    
    def _add_metadata(self, batch: dict, row) -> dict:
        """메타데이터 추가 공통 메서드"""
        # input_text 디코딩
        if batch["input_ids"].dim() == 2:
            input_ids_flat = batch["input_ids"][0]
        else:
            input_ids_flat = batch["input_ids"]
            
        batch["input_text"] = self.tokenizer.decode(
            input_ids_flat.tolist(), 
            skip_special_tokens=True
        )
        batch["image_path"] = str(row.url)
        batch["query"] = str(row.query)
        
        return batch
    
    def _get_dummy_sample(self) -> dict:
        """에러 발생 시 반환할 더미 샘플"""
        dummy_ids = torch.tensor([[0]], dtype=torch.long)
        return {
            "pixel_values": torch.zeros((1, 3, 224, 224)),
            "input_ids": dummy_ids,
            "attention_mask": torch.ones_like(dummy_ids),
            "labels": torch.full_like(dummy_ids, -100),
            "input_text": "",
            "image_path": "",
            "query": "",
            "gt_annotation": ""
        }


class ChatPanoDataset(BasePanoDataset):
    """학습용 파노라마 데이터셋 - user ↔ assistant 대화 + 파노라마 이미지"""
    
    def __init__(
        self,
        csv_path: str,
        processor: PanoLLaVAProcessor,
        tokenizer: AutoTokenizer,
        system_msg: str | None = "You are a helpful assistant.",
        flatten: bool = True,
        gen: bool = False  # assistant 생성 프롬프트 추가 여부
    ):
        super().__init__(csv_path, processor, tokenizer, system_msg, flatten)
        self.gen = gen
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            # --- 이미지 로드
            pil = self._load_image(row.url)
            
            # --- 프롬프트 빌드
            builder = ConversationPromptBuilder(self.tokenizer, system_msg=self.system_msg)
            builder.push("user", str(row.query))
            
            if self.gen:
                # 생성 모드: assistant 답변 없이 프롬프트만
                builder.push("assistant", "")  # 빈 assistant 시작
            else:
                # 학습 모드: 전체 대화 포함
                builder.push("assistant", str(row.annotation))
            
            # --- Processor 호출
            batch = self.proc(pil, builder, flatten=self.flatten)
            
            # --- 레이블 생성
            batch = self._generate_labels(batch, builder, row)
            
            # --- 배치 차원 정규화
            batch = self._normalize_batch_dimensions(batch)
            
            # --- 메타데이터 추가
            batch = self._add_metadata(batch, row)
            
            # Ground truth annotation (평가용)
            if not self.gen:
                batch["gt_annotation"] = str(row.annotation)
            else:
                batch["gt_annotation"] = ""
            
            return batch
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return self._get_dummy_sample()
    
    def _generate_labels(self, batch: dict, builder: ConversationPromptBuilder, row) -> dict:
        """레이블 생성 메서드"""
        if not self.gen:
            # 학습 모드에서만 레이블 생성
            input_ids = batch["input_ids"].clone()
            
            # User 부분만 따로 토크나이징하여 정확한 길이 계산
            try:
                # System + User 프롬프트만 추출
                user_builder = ConversationPromptBuilder(self.tokenizer, system_msg=self.system_msg)
                user_builder.push("user", str(row.query))
                user_prompt = user_builder.formatted()
                
                user_tokens = self.tokenizer(
                    user_prompt,
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"]
                
                user_len = user_tokens.shape[1]
                
                # labels 생성: user 부분은 -100으로 마스킹
                labels = input_ids.clone()
                if labels.dim() == 3:  # (1, 1, L)
                    labels = labels.squeeze(0)  # (1, L)
                
                if labels.shape[0] > 0 and user_len < labels.shape[1]:
                    labels[0, :user_len] = -100
                
                batch["labels"] = labels
                
            except Exception as e:
                print(f"Error generating labels: {e}")
                # Fallback: 모든 토큰을 학습 대상으로 설정
                labels = input_ids.clone()
                if labels.dim() == 3:
                    labels = labels.squeeze(0)
                batch["labels"] = labels
        else:
            # 생성 모드에서는 레이블 없음
            input_ids = batch["input_ids"]
            if input_ids.dim() == 3:
                input_ids = input_ids.squeeze(0)
            batch["labels"] = torch.full_like(input_ids, -100)
        
        return batch


class ChatPanoTestDataset(BasePanoDataset):
    """평가/테스트용 데이터셋 - 정답 없이 생성만"""
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # --- 이미지 로드
        pil = self._load_image(row.url)
        
        # --- 프롬프트 빌드 (user query만)
        builder = ConversationPromptBuilder(self.tokenizer, system_msg=self.system_msg)
        builder.push("user", str(row.query))
        
        # --- Processor 호출: annotation 없이 user 쿼리만
        batch = self.proc(pil, builder, flatten=self.flatten)
        
        # --- 배치 차원 정규화
        batch = self._normalize_batch_dimensions(batch)
        
        # --- 메타데이터 추가
        batch = self._add_metadata(batch, row)
        
        # Ground truth annotation (평가용 정답)
        batch["gt_annotation"] = str(row.annotation)
        
        return batch


# 편의를 위한 팩토리 함수들
def create_train_dataset(
    csv_path: str,
    processor: PanoLLaVAProcessor,
    tokenizer: AutoTokenizer,
    **kwargs
) -> ChatPanoDataset:
    """학습용 데이터셋 생성 팩토리 함수"""
    return ChatPanoDataset(csv_path, processor, tokenizer, **kwargs)


def create_test_dataset(
    csv_path: str,
    processor: PanoLLaVAProcessor,
    tokenizer: AutoTokenizer,
    **kwargs
) -> ChatPanoTestDataset:
    """테스트용 데이터셋 생성 팩토리 함수"""
    return ChatPanoTestDataset(csv_path, processor, tokenizer, **kwargs)


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