"""
설정 파일 로드 및 관리 유틸리티
"""

import yaml
import os
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig


class ConfigManager:
    """설정 파일을 로드하고 관리하는 클래스"""
    
    def __init__(self, config_dir: str = "config"):
        """
        Args:
            config_dir: 설정 파일이 있는 디렉토리
        """
        self.config_dir = config_dir
        self._configs = {}
    
    def load_config(self, config_name: str) -> DictConfig:
        """
        설정 파일을 로드합니다.
        
        Args:
            config_name: 설정 파일명 (확장자 제외)
            
        Returns:
            DictConfig: 로드된 설정
        """
        if config_name in self._configs:
            return self._configs[config_name]
        
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        config = OmegaConf.create(config_dict)
        self._configs[config_name] = config
        
        return config
    
    def get_model_config(self) -> DictConfig:
        """모델 설정을 반환합니다."""
        return self.load_config("model_config")
    
    def get_training_config(self) -> DictConfig:
        """학습 설정을 반환합니다.""" 
        return self.load_config("training_config")
    
    def merge_configs(self, *config_names: str) -> DictConfig:
        """여러 설정 파일을 병합합니다."""
        merged = OmegaConf.create({})
        
        for config_name in config_names:
            config = self.load_config(config_name)
            merged = OmegaConf.merge(merged, config)
        
        return merged
    
    def override_config(self, config: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
        """설정을 오버라이드합니다."""
        config_copy = OmegaConf.copy(config)
        
        for key, value in overrides.items():
            OmegaConf.set(config_copy, key, value)
        
        return config_copy
    
    def save_config(self, config: DictConfig, filename: str):
        """설정을 파일로 저장합니다."""
        output_path = os.path.join(self.config_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(OmegaConf.to_yaml(config), f, allow_unicode=True)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일을 로드하는 간단한 함수"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml_config(config: Dict[str, Any], config_path: str):
    """설정을 YAML 파일로 저장하는 간단한 함수"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def get_default_config() -> Dict[str, Any]:
    """기본 설정을 반환합니다."""
    return {
        "model": {
            "image_encoder": {
                "name": "openai/clip-vit-base-patch32",
                "freeze": False
            },
            "llm": {
                "name": "microsoft/DialoGPT-medium",
                "freeze_layers": 0
            }
        },
        "panorama": {
            "resolution": [512, 1024],
            "projection": "equirectangular",
            "crop_strategy": "adaptive",
            "normalize": True
        },
        "text": {
            "max_length": 512,
            "padding": "max_length",
            "truncation": True
        },
        "processing": {
            "batch_size": 4,
            "num_workers": 2,
            "device": "auto"
        }
    }
