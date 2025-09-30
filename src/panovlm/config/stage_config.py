"""
스테이지별 설정 관리
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class LossConfig:
    """로스 함수 설정"""
    vicreg: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "weight": 0.0})
    language_modeling: Dict[str, Any] = field(default_factory=lambda: {"enabled": True})


@dataclass
class DatasetConfig:
    """데이터셋 설정"""
    train: List[str]
    val: List[str]
    max_text_length: str = "auto"


@dataclass
class TrainingStageConfig:
    """훈련 최적화 설정"""
    lr: float = 1e-5
    epochs: int = 1
    batch_size: int = 1
    accumulate_grad_batches: int = 2
    early_stopping_patience: Optional[int] = None


@dataclass
class StageConfig:
    """단일 스테이지 설정"""
    name: str
    data: DatasetConfig
    loss: LossConfig
    optimizer: TrainingStageConfig
    image_processing: Dict[str, Any]