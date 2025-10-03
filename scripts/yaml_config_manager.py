"""
YAML 기반 설정 관리자
공통 설정과 스테이지별 설정을 효율적으로 관리
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class StageConfig:
    """단일 스테이지 설정"""
    name: str
    epochs: int
    learning_rate: float
    batch_size: int
    accumulate_grad_batches: int
    losses: List[Dict[str, Any]]
    data: Dict[str, Any]
    image_processing: Dict[str, Any]
    model_config: Dict[str, Any]

class ConfigManager:
    """YAML 설정 관리자"""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config: Optional[Dict[str, Any]] = None
        
    def load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self._validate_config()
        self._merge_common_settings()
        return self.config
    
    def _validate_config(self):
        """설정 검증"""
        required_sections = ['experiment', 'model', 'training']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Required section '{section}' not found in config")
        
        # 스테이지 순서와 실제 스테이지 정의 확인
        stage_order = self.config['training'].get('stage_order', [])
        stages = self.config['training'].get('stages', {})
        
        for stage_name in stage_order:
            if stage_name not in stages:
                raise ValueError(f"Stage '{stage_name}' defined in stage_order but not in stages")
    
    def _merge_common_settings(self):
        """공통 설정을 각 스테이지에 병합"""
        common_model = self.config.get('model', {})
        stages = self.config['training']['stages']
        
        for stage_name, stage_config in stages.items():
            # 모델 설정 병합
            if 'model_config' in stage_config:
                # LoRA 설정 병합
                if 'lora' in stage_config['model_config']:
                    stage_lora = stage_config['model_config']['lora']
                    if isinstance(stage_lora, dict):
                        # 스테이지별 LoRA 설정이 dict인 경우 공통 설정과 병합
                        common_lora = common_model.get('lora', {})
                        merged_lora = {**common_lora, **stage_lora}
                        stage_config['model_config']['lora'] = merged_lora
                    elif stage_lora.get('enabled', False):
                        # enabled: true만 있는 경우 공통 설정 사용
                        stage_config['model_config']['lora'] = common_model.get('lora', {})
                        stage_config['model_config']['lora']['enabled'] = True
    
    def get_stage_config(self, stage_name: str) -> StageConfig:
        """특정 스테이지 설정 반환"""
        if not self.config:
            raise ValueError("Config must be loaded first")
        
        stages = self.config['training']['stages']
        if stage_name not in stages:
            raise ValueError(f"Stage '{stage_name}' not found")
        
        stage_data = stages[stage_name]
        
        return StageConfig(
            name=stage_name,
            epochs=stage_data['epochs'],
            learning_rate=stage_data['learning_rate'],
            batch_size=stage_data['batch_size'],
            accumulate_grad_batches=stage_data.get('accumulate_grad_batches', 1),
            losses=stage_data['losses'],
            data=stage_data['data'],
            image_processing=stage_data['image_processing'],
            model_config=stage_data['model_config']
        )
    
    def get_stage_order(self) -> List[str]:
        """스테이지 실행 순서 반환"""
        return self.config['training'].get('stage_order', [])
    
    def get_model_config(self) -> Dict[str, Any]:
        """공통 모델 설정 반환"""
        return self.config.get('model', {})
    
    def get_global_config(self) -> Dict[str, Any]:
        """전역 설정 반환"""
        return self.config['training'].get('global', {})
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """실험 정보 반환"""
        return self.config.get('experiment', {})
    
    @classmethod
    def convert_from_json(cls, json_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None):
        """기존 JSON 설정을 YAML로 변환"""
        json_path = Path(json_path)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_config = json.load(f)
        
        # JSON -> YAML 변환 로직
        yaml_config = cls._convert_json_structure(json_config)
        
        if output_path is None:
            output_path = json_path.with_suffix('.yaml')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"Converted {json_path} -> {output_path}")
        return output_path
    
    @staticmethod
    def _convert_json_structure(json_config: Dict[str, Any]) -> Dict[str, Any]:
        """JSON 구조를 YAML 구조로 변환"""
        models = json_config.get('models', {})
        training = json_config.get('training', {})
        
        # 기본 YAML 구조 생성
        yaml_config = {
            'experiment': {
                'name': training.get('prefix', 'converted_experiment'),
                'description': 'Converted from JSON config',
                'tags': ['converted'],
                'version': '1.0'
            },
            'model': {
                'vision': {
                    'name': models.get('vision_name', 'google/siglip-base-patch16-224')
                },
                'language': {
                    'name': models.get('language_model_name', 'meta-llama/Llama-3.2-1B')
                },
                'resampler': {
                    'type': models.get('resampler_type', 'mlp'),
                    'latent_dimension': models.get('latent_dimension', 768)
                },
                'lora': json_config.get('lora', {})
            },
            'data': {
                'default_dataset': {
                    'train_csv': json_config.get('paths', {}).get('csv_train', 'data/quic360/train.csv'),
                    'val_csv': json_config.get('paths', {}).get('csv_val', 'data/quic360/valid.csv'),
                    'max_text_length': training.get('max_text_length', 'auto'),
                    'auto_max_text_length_cap': json_config.get('data', {}).get('auto_max_text_length_cap', 512)
                }
            },
            'image_processing': json_config.get('image_processing', {}),
            'training': {
                'global': {
                    'num_workers': training.get('num_workers', 16),
                    'wandb_project': training.get('wandb_project', 'panollava-training'),
                    'system_msg': training.get('system_msg', ''),
                    'empty_cache_each_step': training.get('empty_cache_each_step', 1)
                },
                'stage_order': training.get('stages', ['vision', 'resampler', 'finetune']),
                'stages': {}
            },
            'evaluation': json_config.get('generation', {}),
            'environment': json_config.get('environment', {}),
            'hardware': {
                'precision': '16-mixed'
            },
            'deepspeed': training.get('deepspeed', {'enabled': False})
        }
        
        # 스테이지 설정 변환
        stage_configs = training.get('stage_configs', {})
        for stage_name, stage_config in stage_configs.items():
            yaml_stage = {
                'epochs': stage_config.get('epochs', 1),
                'learning_rate': stage_config.get('lr', 1e-5),
                'batch_size': stage_config.get('batch_size', 1),
                'accumulate_grad_batches': stage_config.get('accumulate_grad_batches', 1),
                'losses': [],
                'data': stage_config.get('data', yaml_config['data']['default_dataset']),
                'image_processing': stage_config.get('image_processing', yaml_config['image_processing']),
                'model_config': {
                    'freeze_language_model': True,  # 기본값
                    'freeze_vision_encoder': False,
                    'use_vicreg_head': stage_config.get('vicreg_loss_weight', 0.0) > 0
                }
            }
            
            # Loss 설정 변환
            if stage_config.get('vicreg_loss_weight', 0.0) > 0:
                yaml_stage['losses'].append({
                    'type': 'vicreg',
                    'weight': stage_config['vicreg_loss_weight'],
                    'params': {
                        'similarity_weight': stage_config.get('vicreg_similarity_weight', 25.0),
                        'variance_weight': stage_config.get('vicreg_variance_weight', 25.0),
                        'covariance_weight': stage_config.get('vicreg_covariance_weight', 1.0)
                    }
                })
            else:
                yaml_stage['losses'].append({
                    'type': 'cross_entropy',
                    'weight': 1.0
                })
            
            yaml_config['training']['stages'][stage_name] = yaml_stage
        
        return yaml_config
    
    def preview_config(self):
        """설정 미리보기 출력"""
        if not self.config:
            raise ValueError("Config must be loaded first")
        
        exp_info = self.get_experiment_info()
        print(f"\n🧪 실험: {exp_info.get('name', 'Unknown')}")
        print(f"📝 설명: {exp_info.get('description', 'No description')}")
        
        model_config = self.get_model_config()
        print(f"\n🤖 모델 설정:")
        print(f"   Vision: {model_config['vision']['name']}")
        print(f"   Language: {model_config['language']['name']}")
        print(f"   Resampler: {model_config['resampler']['type']}")
        
        print(f"\n🎯 훈련 스테이지:")
        for i, stage_name in enumerate(self.get_stage_order(), 1):
            stage = self.get_stage_config(stage_name)
            print(f"   {i}. {stage.name}")
            print(f"      - 에포크: {stage.epochs}")
            print(f"      - 학습률: {stage.learning_rate}")
            print(f"      - 배치 크기: {stage.batch_size}")
            print(f"      - 로스: {[loss['type'] for loss in stage.losses]}")
            
            # VICReg 사용 여부
            use_vicreg = stage.model_config.get('use_vicreg_head', False)
            print(f"      - VICReg Head: {'✅' if use_vicreg else '❌'}")
            
            # LoRA 사용 여부
            lora_config = stage.model_config.get('lora', {})
            use_lora = lora_config.get('enabled', False) if isinstance(lora_config, dict) else False
            print(f"      - LoRA: {'✅' if use_lora else '❌'}")
