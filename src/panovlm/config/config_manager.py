"""
개선된 설정 관리 시스템 - YAML 기반 계층적 설정
"""
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import yaml
import json
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
import logging

from .stage_config import StageConfig, TrainingStageConfig, LossConfig, DatasetConfig

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """실험 설정을 담는 타입 안전한 클래스"""
    global_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    environment_config: Dict[str, Any] = field(default_factory=dict)
    lora_config: Dict[str, Any] = field(default_factory=dict)
    stage_name: Optional[str] = None
    stage_config: Optional[StageConfig] = None

class ConfigManager:
    """개선된 설정 관리자"""

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self._config: Optional[DictConfig] = None

    @classmethod
    def from_legacy_json(cls, json_path: Union[str, Path]) -> 'ConfigManager':
        """기존 JSON 설정을 YAML로 변환하여 로드"""
        json_path = Path(json_path)

        # JSON 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            json_config = json.load(f)

        # YAML 형태로 변환
        yaml_config = cls._convert_json_to_yaml_structure(json_config)

        # 임시 YAML 파일 생성
        temp_yaml_path = json_path.with_suffix('.converted.yaml')
        with open(temp_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False, indent=2)

        logger.info(f"Legacy JSON config converted to YAML: {temp_yaml_path}")
        return cls(temp_yaml_path)

    @staticmethod
    def _convert_json_to_yaml_structure(json_config: Dict[str, Any]) -> Dict[str, Any]:
        """기존 JSON 구조를 새로운 YAML 구조로 변환"""
        yaml_config = {
            'experiment': {
                'name': json_config.get('training', {}).get('prefix', 'converted_experiment'),
                'description': 'Converted from legacy JSON config',
                'tags': ['legacy', 'converted']
            },
            'model': {
                'vision': {
                    'name': json_config.get('models', {}).get('vision_name', 'google/siglip-base-patch16-224')
                },
                'language': {
                    'name': json_config.get('models', {}).get('language_model_name', 'Qwen/Qwen2-1.5B')
                },
                'resampler': {
                    'type': json_config.get('models', {}).get('resampler_type', 'mlp'),
                    'hidden_dim': json_config.get('models', {}).get('latent_dimension', 768),
                    'num_layers': json_config.get('models', {}).get('resampler_depth', 2)
                }
            },
            'training': {
                'stages': []
            }
        }

        # 기존 stage_configs를 새로운 stages로 변환
        stage_configs = json_config.get('training', {}).get('stage_configs', {})
        default_config = json_config.get('training', {})

        for stage_name, stage_config in stage_configs.items():
            yaml_stage = {
                'name': stage_name,
                'epochs': stage_config.get('epochs', default_config.get('epochs', 1)),
                'learning_rate': stage_config.get('lr', default_config.get('lr', 1e-4)),
                'batch_size': stage_config.get('batch_size', 4),
                'losses': [
                    {
                        'type': 'autoregressive',  # 기본값
                        'weight': 1.0
                    }
                ],
                'dataset': {
                    'type': 'panorama_chat',
                    'train_csv': json_config.get('paths', {}).get('csv_train', ''),
                    'val_csv': json_config.get('paths', {}).get('csv_val', '')
                },
                'model_config': {
                    'freeze': [],
                    'unfreeze': stage_config.get('unfreeze_modules', [])
                }
            }
            yaml_config['training']['stages'].append(yaml_stage)

        return yaml_config

    def load_config(self) -> ExperimentConfig:
        """YAML 설정을 로드하고 검증"""
        if self.config_path.suffix.lower() == '.json':
            # JSON 파일인 경우 변환 후 로드
            return self.from_legacy_json(self.config_path).load_config()

        with open(self.config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        # OmegaConf로 변환 (변수 보간 지원)
        self._config = OmegaConf.create(raw_config)

        # 기본값 병합
        self._apply_defaults()

        # 검증
        self._validate_config()

        logger.info("Configuration loaded successfully")
        return None

    def _apply_defaults(self):
        """기본 설정값 적용"""
        defaults = OmegaConf.create({
            'hardware': {
                'gpus': 1,
                'precision': '16-mixed',
                'strategy': 'auto',
                'accumulate_grad_batches': 1
            },
            'logging': {
                'wandb': {'enabled': False},
                'tensorboard': {'enabled': True, 'log_dir': 'logs'},
                'checkpoints': {
                    'save_every_n_epochs': 1,
                    'save_top_k': 3,
                    'monitor': 'val_loss',
                    'mode': 'min'
                }
            },
            'evaluation': {
                'metrics': ['bleu', 'rouge'],
                'run_validation': True,
                'run_test': False
            }
        })
        self._config = OmegaConf.merge(defaults, self._config)

    def _validate_config(self):
        """설정 검증"""
        required_keys = ['experiment', 'models', 'training', 'data', 'generation', 'environment', 'lora']
        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Required config key '{key}' not found")

        # 실험 설정 검증
        if 'name' not in self._config.experiment:
            raise ValueError("experiment.name is required")

        # 훈련 설정 검증
        if 'stage_configs' not in self._config.training:
            raise ValueError("training.stage_configs is required")

        # 스테이지 설정 검증
        stage_configs = self._config.training.stage_configs
        if not stage_configs:
            raise ValueError("At least one stage configuration is required")

        # 모델 설정 검증
        required_model_keys = ['vision_name', 'language_model_name', 'resampler_type']
        for key in required_model_keys:
            if key not in self._config.models:
                raise ValueError(f"models.{key} is required")

        logger.info(f"Config validation passed. Found {len(stage_configs)} stages.")

    def override_config(self, overrides: Dict[str, Any]) -> 'ConfigManager':
        """설정 오버라이드 (CLI에서 사용)"""
        if self._config is None:
            raise ValueError("Config must be loaded before applying overrides")

        override_conf = OmegaConf.create(overrides)
        self._config = OmegaConf.merge(self._config, override_conf)
        return self

    def get_stage_config(self, stage_name: str) -> Optional[StageConfig]:
        """특정 스테이지 설정 반환 (StageConfig 객체)"""
        if self._config is None:
            raise ValueError("Config must be loaded first")

        stage_configs = self._config.training.stage_configs
        if stage_name not in stage_configs:
            return None

        stage_data = OmegaConf.to_container(stage_configs[stage_name])

        # StageConfig 객체 생성
        return StageConfig(
            name=stage_name,
            data=DatasetConfig(**stage_data['data']),
            loss=LossConfig(**stage_data['loss']),
            optimizer=TrainingStageConfig(**stage_data['optimizer']),
            image_processing=stage_data.get('image_processing', OmegaConf.to_container(self._config.image_processing))
        )

    def save_config(self, output_path: Union[str, Path]):
        """현재 설정을 YAML 파일로 저장"""
        if self._config is None:
            raise ValueError("Config must be loaded first")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                OmegaConf.to_container(self._config),
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )

        logger.info(f"Config saved to: {output_path}")

    def get_available_stages(self) -> List[str]:
        """사용 가능한 스테이지 목록 반환"""
        if self._config is None:
            raise ValueError("Config must be loaded first")

        return list(self._config.training.stage_configs.keys())

    def get_default_stage(self) -> str:
        """기본 스테이지 반환"""
        if self._config is None:
            raise ValueError("Config must be loaded first")

        return self._config.training.default_stage

    def get_experiment_config(self) -> ExperimentConfig:
        """실험 설정 객체 반환"""
        if self._config is None:
            raise ValueError("Config must be loaded first")

        return ExperimentConfig(
            global_config=OmegaConf.to_container(self._config.training),
            model_config=OmegaConf.to_container(self._config.models),
            data_config=OmegaConf.to_container(self._config.data),
            generation_config=OmegaConf.to_container(self._config.generation),
            environment_config=OmegaConf.to_container(self._config.environment),
            lora_config=OmegaConf.to_container(self._config.lora)
        )