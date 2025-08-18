#!/usr/bin/env python3
# coding: utf-8
"""
PanoramaVLM 모델 설정 관리자
===========================

모델 하이퍼파라미터와 설정을 JSON 파일로 저장/로딩하는 시스템입니다.
훈련 시 설정을 저장하고, 추론/평가 시 일관된 설정을 사용할 수 있도록 합니다.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import warnings


@dataclass
class ModelConfig:
    """PanoramaVLM 모델 설정"""
    
    # 모델 아키텍처
    vision_name: str = "google/siglip-base-patch16-224"
    language_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    resampler_type: str = "mlp"
    latent_dimension: int = 768
    
    # VICReg 관련 설정
    vicreg_loss_weight: float = 1.0
    vicreg_overlap_ratio: float = 0.5
    
    # 텍스트 처리 설정
    max_text_length: int = 512
    
    # 이미지 처리 설정
    image_size: tuple = field(default_factory=lambda: (224, 224))
    crop_strategy: str = "e2p"
    fov_deg: float = 90.0
    overlap_ratio: float = 0.5
    
    # 훈련 설정 (옵션)
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    num_epochs: Optional[int] = None
    stage: Optional[str] = None  # vision, resampler, finetune
    
    # LoRA 설정 (옵션)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[list] = None
    
    # 메타데이터
    created_at: Optional[str] = None
    created_by: str = "PanoramaVLM"
    version: str = "1.0.0"
    description: str = ""
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        
        # LoRA 타겟 모듈 기본값 설정
        if self.use_lora and self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """딕셔너리에서 생성"""
        # 알려진 필드만 추출
        try:
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        except AttributeError:
            # fallback: dataclass fields의 키만 사용
            valid_fields = set(cls.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        # tuple 타입 변환
        if 'image_size' in filtered_dict and isinstance(filtered_dict['image_size'], list):
            filtered_dict['image_size'] = tuple(filtered_dict['image_size'])
        
        return cls(**filtered_dict)
    
    def update(self, **kwargs) -> 'ModelConfig':
        """설정 업데이트 (새 인스턴스 반환)"""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """설정을 JSON 파일로 저장"""
        ConfigManager.save_config(self, file_path)
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'ModelConfig':
        """JSON 파일에서 설정 로딩"""
        return ConfigManager.load_config(file_path)
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """PanoramaVLM 모델 생성에 필요한 kwargs 반환"""
        return {
            'vision_name': self.vision_name,
            'language_model_name': self.language_model_name,
            'resampler_type': self.resampler_type,
            'latent_dimension': self.latent_dimension,
            'vicreg_loss_weight': self.vicreg_loss_weight,
            'vicreg_overlap_ratio': self.vicreg_overlap_ratio,
            'max_text_length': self.max_text_length,
        }
    
    def get_image_processor_kwargs(self) -> Dict[str, Any]:
        """이미지 프로세서 생성에 필요한 kwargs 반환"""
        return {
            'image_size': self.image_size,
            'crop_strategy': self.crop_strategy,
            'fov_deg': self.fov_deg,
            'overlap_ratio': self.overlap_ratio,
        }
    
    def get_lora_kwargs(self) -> Dict[str, Any]:
        """LoRA 설정에 필요한 kwargs 반환"""
        return {
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': self.lora_target_modules,
        }
    
    def validate(self) -> bool:
        """설정 유효성 검사"""
        try:
            # 필수 문자열 필드 확인
            assert self.vision_name.strip(), "vision_name은 비어있을 수 없습니다"
            assert self.language_model_name.strip(), "language_model_name은 비어있을 수 없습니다"
            assert self.resampler_type in ["mlp"], f"지원하지 않는 resampler_type: {self.resampler_type}"
            
            # 숫자 범위 확인
            assert self.latent_dimension > 0, "latent_dimension은 양수여야 합니다"
            assert self.vicreg_loss_weight >= 0, "vicreg_loss_weight는 0 이상이어야 합니다"
            assert 0 <= self.vicreg_overlap_ratio <= 1, "vicreg_overlap_ratio는 0-1 사이여야 합니다"
            assert self.max_text_length > 0, "max_text_length는 양수여야 합니다"
            
            # 이미지 설정 확인
            assert len(self.image_size) == 2, "image_size는 (height, width) 튜플이어야 합니다"
            assert all(s > 0 for s in self.image_size), "image_size의 모든 값은 양수여야 합니다"
            assert self.crop_strategy in ["resize","e2p","sliding_window", "cubemap","anyres"], f"지원하지 않는 crop_strategy: {self.crop_strategy}"
            assert 0 < self.fov_deg <= 180, "fov_deg는 0-180 사이여야 합니다"
            assert 0 <= self.overlap_ratio <= 1, "overlap_ratio는 0-1 사이여야 합니다"
            
            # LoRA 설정 확인
            if self.use_lora:
                assert self.lora_r > 0, "lora_r은 양수여야 합니다"
                assert self.lora_alpha > 0, "lora_alpha는 양수여야 합니다"
                assert 0 <= self.lora_dropout <= 1, "lora_dropout은 0-1 사이여야 합니다"
            
            return True
            
        except AssertionError as e:
            warnings.warn(f"설정 유효성 검사 실패: {e}")
            return False
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"ModelConfig(vision={self.vision_name}, language={self.language_model_name}, dim={self.latent_dimension})"

class Config:
    """Training configuration management"""
    
    STAGE_DEFAULTS = {
        "vision": {
            "epochs": 1, 
            "lr": 5e-6, 
            "batch_size": 16,   # 32에서 8로 감소
            "vicreg_loss_weight": 1.0, 
            "max_text_length": 32
        },
        "resampler": {
            "epochs": 1, 
            "lr": 2e-6, 
            "batch_size": 8,   # 16에서 4로 감소
            "vicreg_loss_weight": 0.0, 
            "max_text_length": 64
        },
        "finetune": {
            "epochs": 1, 
            "lr": 2e-6, 
            "batch_size": 8,   # 16에서 4로 감소
            "vicreg_loss_weight": 0.0, 
            "max_text_length": 128
        }
    }


class ConfigManager:
    """설정 파일 관리 유틸리티"""
    
    DEFAULT_CONFIG_NAME = "model_config.json"
    
    @staticmethod
    def save_config(config: ModelConfig, file_path: Union[str, Path]) -> None:
        """설정을 JSON 파일로 저장"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 설정 유효성 검사
        if not config.validate():
            warnings.warn("설정 유효성 검사에 실패했지만 저장을 계속합니다")
        
        config_dict = config.to_dict()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 설정 저장 완료: {file_path}")
            
        except Exception as e:
            raise RuntimeError(f"설정 저장 실패: {e}")
    
    @staticmethod
    def load_config(file_path: Union[str, Path]) -> ModelConfig:
        """JSON 파일에서 설정 로딩"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            config = ModelConfig.from_dict(config_dict)
            
            # 설정 유효성 검사
            if not config.validate():
                warnings.warn("로드된 설정이 유효성 검사에 실패했습니다")
            
            print(f"✅ 설정 로딩 완료: {file_path}")
            return config
            
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON 파싱 실패: {e}")
        except Exception as e:
            raise RuntimeError(f"설정 로딩 실패: {e}")
    
    @staticmethod
    def auto_detect_config(checkpoint_path: Union[str, Path]) -> Optional[ModelConfig]:
        """체크포인트 경로에서 설정 파일 자동 감지"""
        checkpoint_path = Path(checkpoint_path)
        
        # 체크포인트가 파일인 경우 디렉토리 추출
        if checkpoint_path.is_file():
            search_dir = checkpoint_path.parent
        else:
            search_dir = checkpoint_path
        
        # 설정 파일 후보들
        config_candidates = [
            search_dir / ConfigManager.DEFAULT_CONFIG_NAME,
            search_dir / "config.json",
            search_dir / "model_config.json",
            search_dir / "panovlm_config.json"
        ]
        
        for config_path in config_candidates:
            if config_path.exists():
                try:
                    return ConfigManager.load_config(config_path)
                except Exception as e:
                    warnings.warn(f"설정 파일 로딩 실패 ({config_path}): {e}")
        
        return None
    
    @staticmethod
    def create_default_config(**overrides) -> ModelConfig:
        """기본 설정 생성 (오버라이드 적용 가능)"""
        config = ModelConfig()
        if overrides:
            config = config.update(**overrides)
        return config
    
    @staticmethod
    def migrate_old_config(old_config_dict: Dict[str, Any]) -> ModelConfig:
        """구버전 설정을 새 형식으로 마이그레이션"""
        # 구버전 필드명 매핑
        field_mapping = {
            'vision_model': 'vision_name',
            'language_model': 'language_model_name',
            'resampler': 'resampler_type',
            'dim': 'latent_dimension',
            'vicreg_weight': 'vicreg_loss_weight',
            'vicreg_overlap': 'vicreg_overlap_ratio',
            'max_length': 'max_text_length',
        }
        
        # 필드명 변환
        migrated_dict = {}
        for old_key, value in old_config_dict.items():
            new_key = field_mapping.get(old_key, old_key)
            migrated_dict[new_key] = value
        
        return ModelConfig.from_dict(migrated_dict)
    


# 편의 함수들
def create_config(**kwargs) -> ModelConfig:
    """편의 함수: 설정 생성"""
    return ConfigManager.create_default_config(**kwargs)

def save_config(config: ModelConfig, file_path: Union[str, Path]) -> None:
    """편의 함수: 설정 저장"""
    ConfigManager.save_config(config, file_path)

def load_config(file_path: Union[str, Path]) -> ModelConfig:
    """편의 함수: 설정 로딩"""
    return ConfigManager.load_config(file_path)

def auto_detect_config(checkpoint_path: Union[str, Path]) -> Optional[ModelConfig]:
    """편의 함수: 설정 자동 감지"""
    return ConfigManager.auto_detect_config(checkpoint_path)
