# coding: utf-8
"""
Configuration Management for Panorama VLM
==========================================

YAML 기반 설정 관리 시스템
- Base configuration과 stage별 override 지원
- 환경별 설정 오버라이드
- 설정 검증 및 타입 체크
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """LoRA 설정"""
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Optional[list] = None

@dataclass 
class ModelConfig:
    """모델 설정"""
    vision_model_name: str = "google/siglip-base-patch16-224"
    language_model_name: str = "Qwen/Qwen2-0.5B"
    resampler_type: str = "mlp"
    latent_dimension: int = 768
    lora: LoRAConfig = field(default_factory=LoRAConfig)

@dataclass
class ImageConfig:
    """이미지 처리 설정"""
    size: list = field(default_factory=lambda: [224, 224])
    crop_strategy: str = "e2p"

@dataclass
class DataConfig:
    """데이터 설정"""
    csv_train: str = "data/quic360/train.csv"
    csv_val: str = "data/quic360/valid.csv"
    batch_size: int = 4
    num_workers: int = 4
    max_txt_len: int = 512
    image: ImageConfig = field(default_factory=ImageConfig)

@dataclass
class SchedulerConfig:
    """스케줄러 설정"""
    type: str = "linear_with_warmup"
    warmup_steps: Optional[int] = None

@dataclass
class TrainingConfig:
    """학습 설정"""
    epochs: int = 3
    learning_rate: float = 5e-5
    optimizer: str = "adamw"
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    gradient_clip_val: float = 0.5
    precision: str = "16-mixed"

@dataclass
class VICRegConfig:
    """VICReg Loss 설정"""
    loss_weight: float = 1.0
    similarity_weight: float = 25.0
    variance_weight: float = 25.0
    covariance_weight: float = 1.0

@dataclass
class WandBConfig:
    """WandB 설정"""
    project: str = "panorama-vlm"
    name: Optional[str] = None
    dir: str = "./runs"

@dataclass
class CheckpointConfig:
    """체크포인트 설정"""
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 1
    save_last: bool = True
    every_n_epochs: int = 1

@dataclass
class EarlyStoppingConfig:
    """Early Stopping 설정"""
    patience: int = 3
    min_delta: float = 0.001

@dataclass
class LoggingConfig:
    """로깅 설정"""
    wandb: WandBConfig = field(default_factory=WandBConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

@dataclass
class MemoryConfig:
    """메모리 관리 설정"""
    auto_adjust_batch_size: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True

@dataclass
class HardwareConfig:
    """하드웨어 설정"""
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "16-mixed"
    deterministic: bool = False
    benchmark: bool = True
    memory: MemoryConfig = field(default_factory=MemoryConfig)

@dataclass
class ValidationConfig:
    """검증 설정"""
    check_interval: float = 0.25
    log_samples: bool = True
    num_samples: int = 5
    max_new_tokens: int = 32
    temperature: float = 0.7

@dataclass
class PanoVLMConfig:
    """Panorama VLM 전체 설정"""
    stage: str = "vision"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vicreg: VICRegConfig = field(default_factory=VICRegConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

class ConfigManager:
    """설정 관리자"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.base_config_path = self.config_dir / "model_config.yaml"  # base.yaml -> model_config.yaml
        
    def load_config(self, stage: str, override_file: Optional[str] = None) -> PanoVLMConfig:
        """설정 로드 및 병합
        
        Args:
            stage: 학습 단계 ('vision', 'resampler', 'finetune')
            override_file: 추가 설정 파일 경로
            
        Returns:
            PanoVLMConfig: 병합된 설정
        """
        # 1. Base 설정 로드 (model_config.yaml)
        base_config = self._load_yaml(self.base_config_path)
        logger.info(f"✓ Loaded base config from {self.base_config_path}")
        
        # 2. Stage별 설정 로드 및 병합
        stage_config_path = self.config_dir / f"{stage}_stage.yaml"  # stages/{stage}.yaml -> {stage}_stage.yaml
        if stage_config_path.exists():
            stage_config = self._load_yaml(stage_config_path)
            base_config = self._deep_merge(base_config, stage_config)
            logger.info(f"✓ Loaded stage config from {stage_config_path}")
        else:
            logger.warning(f"Stage config not found: {stage_config_path}")
        
        # 3. Override 설정 로드 및 병합 (선택사항)
        if override_file and Path(override_file).exists():
            override_config = self._load_yaml(override_file)
            base_config = self._deep_merge(base_config, override_config)
            logger.info(f"✓ Applied override config from {override_file}")
        
        # 4. 환경 변수 오버라이드
        base_config = self._apply_env_overrides(base_config)
        
        # 5. 설정 검증 및 dataclass 변환
        config = self._validate_and_convert(base_config)
        
        # 6. Stage 정보 설정
        config.stage = stage
        
        logger.info(f"✓ Configuration loaded for stage: {stage}")
        return config
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """YAML 파일 로드 (과학 표기법 지원)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 먼저 문자열로 읽어서 과학 표기법 처리
                content = f.read()
                
                # 과학 표기법을 소수점 표기법으로 변환 (더 안전한 처리를 위해)
                import re
                def convert_scientific(match):
                    return str(float(match.group(0)))
                
                # e나 E가 포함된 숫자를 찾아서 변환
                content = re.sub(r'\b\d+\.?\d*[eE][+-]?\d+\b', convert_scientific, content)
                
                return yaml.safe_load(content) or {}
        except Exception as e:
            logger.error(f"Failed to load YAML file {file_path}: {e}")
            return {}
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """딥 머지 (재귀적 딕셔너리 병합)"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict) -> Dict:
        """환경 변수를 통한 설정 오버라이드
        
        환경 변수 형식: PANO_VLM_SECTION_KEY=value
        예: PANO_VLM_TRAINING_LEARNING_RATE=1e-4
        """
        prefix = "PANO_VLM_"
        
        for env_var, value in os.environ.items():
            if not env_var.startswith(prefix):
                continue
            
            # 환경 변수 파싱: PANO_VLM_TRAINING_LEARNING_RATE -> ['training', 'learning_rate']
            keys = env_var[len(prefix):].lower().split('_')
            
            # 타입 변환 시도
            try:
                # 과학 표기법 처리
                if 'e' in value.lower():
                    value = float(value)
                # 소수점 있는 숫자
                elif '.' in value:
                    value = float(value)
                # 정수
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    value = int(value)
                # 불린 값
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
            except ValueError:
                pass  # 문자열로 유지
            
            # 중첩 딕셔너리에 값 설정
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
            
            logger.info(f"✓ Applied env override: {env_var} = {value}")
        
        return config
    
    def _convert_types(self, config_dict: Dict) -> Dict:
        """YAML에서 로드된 값들의 타입을 올바르게 변환"""
        def convert_value(value):
            if isinstance(value, str):
                # 과학 표기법 처리 (예: 1e-4, 5e-5)
                try:
                    if 'e' in value.lower():
                        return float(value)
                except (ValueError, AttributeError):
                    pass
                
                # 불린 값 처리
                if value.lower() in ('true', 'false'):
                    return value.lower() == 'true'
                
                # 숫자 처리
                try:
                    if '.' in value:
                        return float(value)
                    elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                        return int(value)
                except (ValueError, AttributeError):
                    pass
            
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            
            return value
        
        return convert_value(config_dict)
    
    def _validate_and_convert(self, config_dict: Dict) -> PanoVLMConfig:
        """설정 검증 및 dataclass 변환"""
        try:
            # 타입 변환 전처리
            config_dict = self._convert_types(config_dict)
            
            # 중첩 구조 변환
            model_config = ModelConfig(
                vision_model_name=config_dict.get('model', {}).get('vision_model_name', ModelConfig.vision_model_name),
                language_model_name=config_dict.get('model', {}).get('language_model_name', ModelConfig.language_model_name),
                resampler_type=config_dict.get('model', {}).get('resampler_type', ModelConfig.resampler_type),
                latent_dimension=config_dict.get('model', {}).get('latent_dimension', ModelConfig.latent_dimension),
                lora=LoRAConfig(**config_dict.get('model', {}).get('lora', {}))
            )
            
            image_config = ImageConfig(**config_dict.get('data', {}).get('image', {}))
            data_config = DataConfig(
                **{k: v for k, v in config_dict.get('data', {}).items() if k != 'image'},
                image=image_config
            )
            
            scheduler_config = SchedulerConfig(**config_dict.get('training', {}).get('scheduler', {}))
            training_config = TrainingConfig(
                **{k: v for k, v in config_dict.get('training', {}).items() if k != 'scheduler'},
                scheduler=scheduler_config
            )
            
            vicreg_config = VICRegConfig(**config_dict.get('vicreg', {}))
            
            wandb_config = WandBConfig(**config_dict.get('logging', {}).get('wandb', {}))
            checkpoint_config = CheckpointConfig(**config_dict.get('logging', {}).get('checkpoint', {}))
            early_stopping_config = EarlyStoppingConfig(**config_dict.get('logging', {}).get('early_stopping', {}))
            logging_config = LoggingConfig(
                wandb=wandb_config,
                checkpoint=checkpoint_config,
                early_stopping=early_stopping_config
            )
            
            memory_config = MemoryConfig(**config_dict.get('hardware', {}).get('memory', {}))
            hardware_config = HardwareConfig(
                **{k: v for k, v in config_dict.get('hardware', {}).items() if k != 'memory'},
                memory=memory_config
            )
            
            validation_config = ValidationConfig(**config_dict.get('validation', {}))
            
            config = PanoVLMConfig(
                stage=config_dict.get('stage', 'vision'),
                model=model_config,
                data=data_config,
                training=training_config,
                vicreg=vicreg_config,
                logging=logging_config,
                hardware=hardware_config,
                validation=validation_config
            )
            
            # 기본 검증
            self._validate_config(config)
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to validate/convert config: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    def _validate_config(self, config: PanoVLMConfig):
        """설정 값 검증"""
        # 필수 파일 존재 확인
        if config.stage in ['resampler', 'finetune']:
            if not Path(config.data.csv_train).exists():
                logger.warning(f"Training CSV not found: {config.data.csv_train}")
            if not Path(config.data.csv_val).exists():
                logger.warning(f"Validation CSV not found: {config.data.csv_val}")
        
        # 값 범위 검증 (타입 변환 후)
        try:
            lr = float(config.training.learning_rate)
            assert lr > 0, f"Learning rate must be positive, got {lr}"
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid learning rate: {config.training.learning_rate}, error: {e}")
        
        try:
            epochs = int(config.training.epochs)
            assert epochs > 0, f"Epochs must be positive, got {epochs}"
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid epochs: {config.training.epochs}, error: {e}")
        
        try:
            batch_size = int(config.data.batch_size)
            assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid batch size: {config.data.batch_size}, error: {e}")
        
        # LoRA 설정 검증
        if config.model.lora.enabled:
            try:
                r = int(config.model.lora.r)
                assert r > 0, f"LoRA rank must be positive, got {r}"
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid LoRA rank: {config.model.lora.r}, error: {e}")
            
            try:
                alpha = int(config.model.lora.alpha)
                assert alpha > 0, f"LoRA alpha must be positive, got {alpha}"
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid LoRA alpha: {config.model.lora.alpha}, error: {e}")
            
            try:
                dropout = float(config.model.lora.dropout)
                assert 0 <= dropout <= 1, f"LoRA dropout must be in [0, 1], got {dropout}"
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid LoRA dropout: {config.model.lora.dropout}, error: {e}")
        
        logger.info("✓ Configuration validation passed")
    
    def save_config(self, config: PanoVLMConfig, save_path: str):
        """설정을 YAML 파일로 저장"""
        import dataclasses
        
        def dataclass_to_dict(obj):
            """Dataclass를 딕셔너리로 변환"""
            if dataclasses.is_dataclass(obj):
                return {k: dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        
        config_dict = dataclass_to_dict(config)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logger.info(f"✓ Configuration saved to {save_path}")

# 전역 설정 관리자 인스턴스
config_manager = ConfigManager()

def load_stage_config(stage: str, override_file: Optional[str] = None) -> PanoVLMConfig:
    """Stage별 설정 로드 (편의 함수)"""
    return config_manager.load_config(stage, override_file)