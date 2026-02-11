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
    image_size: Optional[tuple] = None

    # 리샘플러 세부 설정 (resampler_type에 따라 자동 설정됨)
    resampler_depth: int = 2
    resampler_hidden_dim: Optional[int] = None
    resampler_use_ln: bool = True
    resampler_enable_cross_view: bool = False
    resampler_num_views: int = 8
    resampler_dropout: float = 0.1
    resampler_heads: int = 8
    resampler_num_latents: int = 32
    
    # VICReg 관련 설정
    vicreg_loss_weight: float = 1.0
    overlap_ratio: float = 0.5
    use_vicreg_norm: bool = True  # VICReg 경로에서 LayerNorm 사용 여부 (False = 원 철학 준수)
    
    # VICReg 설정 - 간단한 x,y 입력 방식
    vicreg_similarity_weight: float = 25.0
    vicreg_variance_weight: float = 25.0  
    vicreg_covariance_weight: float = 1.0
    
    # 텍스트 처리 설정
    max_text_length: int = 512

    # 토큰 결합(스티칭) 방식
    # - 'drop_overlap' (기본): 각 뷰의 겹치는 좌측 영역을 드랍하여 이어붙임
    # - 'stride_views'       : s=ceil(1/(1-overlap)) 간격으로 뷰를 샘플링해 전체 열 사용
    # - 'concat'             : 단순 인터리브(중복 제거 안 함)
    # - 'resample'           : 파노라마 전역 좌표로 재표본화하여 목표 가로 토큰 수로 정규화
    stitching_mode: str = "drop_overlap"
    stitch_stride_offset: int = 0
    stitch_target_cols: int = 0              # 0이면 자동 (고유 폭)
    stitch_target_to_view_width: bool = False  # True면 최종 가로 토큰 수를 W(뷰 폭)로 맞춤
    stitch_interp: str = "nearest"           # 'nearest' | 'linear' (현재 recent 구현은 nearest)
    
    # LoRA 설정 (옵션)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[list] = None
    
    
    def __post_init__(self):
        """초기화 후 처리"""
        # LoRA 타겟 모듈 기본값 설정
        if self.use_lora and self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        # 리샘플러 타입에 따른 기본값 자동 적용
        self._apply_resampler_defaults()

    def _apply_resampler_defaults(self):
        """리샘플러 타입에 따른 기본값 자동 적용 (명시적으로 설정되지 않은 경우만)"""
        # Config.RESAMPLER_DEFAULTS는 이 파일 하단에 정의됨
        # __post_init__에서 호출되므로 클래스 정의가 완료된 후에만 실행됨
        pass  # Config 클래스 정의 후 아래에서 실제 로직 구현

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
            'overlap_ratio': self.overlap_ratio,
            'use_vicreg_norm': self.use_vicreg_norm,
            'max_text_length': self.max_text_length,
            # VICReg 파라미터들 - 간단한 x,y 입력 방식
            'vicreg_similarity_weight': self.vicreg_similarity_weight,
            'vicreg_variance_weight': self.vicreg_variance_weight,
            'vicreg_covariance_weight': self.vicreg_covariance_weight,
            # stitching
            'stitching_mode': self.stitching_mode,
            'stitch_stride_offset': self.stitch_stride_offset,
            'stitch_target_cols': self.stitch_target_cols,
            'stitch_target_to_view_width': self.stitch_target_to_view_width,
            'stitch_interp': self.stitch_interp,
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
            allowed_resamplers = {"mlp", "perceiver", "bimamba", "bidirectional_mamba", "bi_mamba"}
            assert self.resampler_type in allowed_resamplers, f"지원하지 않는 resampler_type: {self.resampler_type}"
            
            # 숫자 범위 확인
            assert self.latent_dimension > 0, "latent_dimension은 양수여야 합니다"
            assert self.vicreg_loss_weight >= 0, "vicreg_loss_weight는 0 이상이어야 합니다"
            assert 0 <= self.overlap_ratio <= 1, "overlap_ratio는 0-1 사이여야 합니다"
            assert self.max_text_length > 0, "max_text_length는 양수여야 합니다"
            # stitching mode
            assert self.stitch_stride_offset >= 0, "stitch_stride_offset는 0 이상이어야 합니다"
            if self.stitch_interp not in ["nearest", "linear"]:
                raise AssertionError("stitch_interp는 'nearest' 또는 'linear' 여야 합니다")
            if self.stitch_target_cols < 0:
                raise AssertionError("stitch_target_cols는 0 이상이어야 합니다")
            
            
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
            "batch_size": 16,
            "vicreg_loss_weight": 1.0,
            "max_text_length": 32
        },
        "resampler": {
            "epochs": 1,
            "lr": 2e-6,
            "batch_size": 8,
            "vicreg_loss_weight": 0.0,
            "max_text_length": 256
        },
        "finetune": {
            "epochs": 1,
            "lr": 2e-6,
            "batch_size": 8,
            "vicreg_loss_weight": 0.0,
            "max_text_length": 256
        }
    }

    # 리샘플러 타입별 기본 설정
    RESAMPLER_DEFAULTS = {
        "mlp": {
            "latent_dimension": 768,
            "resampler_depth": 3,
            "resampler_hidden_dim": 1536,
            "resampler_use_ln": True,
        },
        "perceiver": {
            "latent_dimension": 768,
            "resampler_num_latents": 32,
            "resampler_depth": 2,
            "resampler_heads": 8,
            "resampler_use_ln": True,
        },
        "bimamba": {
            "latent_dimension": 768,
            "resampler_depth": 2,
            "resampler_hidden_dim": 1536,
            "resampler_use_ln": True,
            "resampler_enable_cross_view": False,
        },
        "bidirectional_mamba": {
            "latent_dimension": 768,
            "resampler_depth": 2,
            "resampler_hidden_dim": 1536,
            "resampler_use_ln": True,
            "resampler_enable_cross_view": False,
        },
        "bi_mamba": {
            "latent_dimension": 768,
            "resampler_depth": 2,
            "resampler_hidden_dim": 1536,
            "resampler_use_ln": True,
            "resampler_enable_cross_view": False,
        },
    }


# Config 클래스 정의 후 ModelConfig._apply_resampler_defaults 구현
def _apply_resampler_defaults_impl(self):
    """리샘플러 타입에 따른 기본값 자동 적용"""
    defaults = Config.RESAMPLER_DEFAULTS.get(self.resampler_type, {})

    if not defaults:
        return

    # dataclass 필드의 기본값과 비교하여 변경되지 않은 필드만 업데이트
    for key, value in defaults.items():
        if hasattr(self, key):
            current_value = getattr(self, key)
            try:
                field_info = self.__dataclass_fields__.get(key)
                if field_info and current_value == field_info.default:
                    setattr(self, key, value)
            except (AttributeError, KeyError):
                # 필드 정보를 가져올 수 없으면 건너뛰기
                pass


# ModelConfig에 실제 구현 주입
ModelConfig._apply_resampler_defaults = _apply_resampler_defaults_impl


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
            
            # JSON config 구조를 ModelConfig 형태로 변환
            flat_config = ConfigManager._flatten_json_config(config_dict)
            config = ModelConfig.from_dict(flat_config)
            
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
    def _flatten_json_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """JSON config의 nested 구조를 ModelConfig의 flat 구조로 변환"""
        flat_config = {}
        
        # 기본 모델 설정 (신규 키 우선, 구키도 병행 지원)
        if 'models' in config_dict:
            models = config_dict['models']
            # 신규 표준화된 키
            lang_name = models.get('language_model_name', models.get('lm_model', 'Qwen/Qwen2.5-0.5B-Instruct'))
            resampler_type = models.get('resampler_type', models.get('resampler', 'mlp'))
            vision_name = models.get('vision_model_name', models.get('vision_name'))

            flat_config.update({
                'vision_name': vision_name,
                'language_model_name': lang_name,
                'resampler_type': resampler_type,
                'latent_dimension': models.get('latent_dimension', 768),
            })
        
        # 데이터 설정
        if 'data' in config_dict:
            data = config_dict['data']
            # max_text_length may be "auto"; only forward numeric to ModelConfig
            mtl_val = data.get('max_text_length', 512)
            try:
                if isinstance(mtl_val, (int, float)) and mtl_val > 0:
                    flat_config['max_text_length'] = int(mtl_val)
            except Exception:
                pass
        
        # 이미지 처리 설정에서 overlap_ratio와 image_size 추출
        if 'image_processing' in config_dict:
            img_proc = config_dict['image_processing']
            flat_config.update({
                'overlap_ratio': img_proc.get('overlap_ratio', 0.5),
                'image_size': img_proc.get('image_size'),
                'crop_strategy': img_proc.get('crop_strategy'),
                'stitching_mode': img_proc.get('stitching_mode', 'drop_overlap'),
                'stitch_stride_offset': img_proc.get('stitch_stride_offset', 0),
                'stitch_target_cols': img_proc.get('stitch_target_cols', 0),
                'stitch_target_to_view_width': img_proc.get('stitch_target_to_view_width', False),
                'stitch_interp': img_proc.get('stitch_interp', 'nearest'),
            })
        
        # 훈련 설정 (특히 VICReg Local)
        if 'training' in config_dict:
            training = config_dict['training']

            # Vision stage 설정에서 VICReg Local 파라미터들 추출
            vision_cfg = None
            stage_cfgs = training.get('stage_configs')
            if isinstance(stage_cfgs, dict) and isinstance(stage_cfgs.get('vision'), dict):
                vision_cfg = stage_cfgs.get('vision')
            elif isinstance(training.get('vision'), dict):
                vision_cfg = training.get('vision')

            if isinstance(vision_cfg, dict):
                flat_config.update({
                    'vicreg_loss_weight': vision_cfg.get('vicreg_loss_weight', 1.0),
                    'vicreg_similarity_weight': vision_cfg.get('vicreg_similarity_weight', 25.0),
                    'vicreg_variance_weight': vision_cfg.get('vicreg_variance_weight', 25.0),
                    'vicreg_covariance_weight': vision_cfg.get('vicreg_covariance_weight', 1.0),
                })
        
        # LoRA 설정
        if 'lora' in config_dict:
            lora = config_dict['lora']
            flat_config.update({
                'use_lora': lora.get('use_lora', False),
                'lora_r': lora.get('rank', 16),
                'lora_alpha': lora.get('alpha', 32),
                'lora_dropout': lora.get('dropout', 0.1),
                'lora_target_modules': lora.get('target_modules', None)
            })
        
        return flat_config
    
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
            # 1. 체크포인트 디렉토리에서 찾기
            search_dir / ConfigManager.DEFAULT_CONFIG_NAME,
            search_dir / "config.json",
            search_dir / "model_config.json", 
            search_dir / "panovlm_config.json",
            # 2. 현재 작업 디렉토리에서 찾기
            Path.cwd() / "config.json",
            Path.cwd() / ConfigManager.DEFAULT_CONFIG_NAME,
            # 3. 환경변수로 지정된 경로
        ]
        
        # 환경변수에서 config 경로 추가
        env_config = os.environ.get("PANOVLM_CONFIG")
        if env_config:
            config_candidates.append(Path(env_config))
        
        for config_path in config_candidates:
            if config_path.exists():
                try:
                    print(f"🔍 설정 파일 발견: {config_path}")
                    return ConfigManager.load_config(config_path)
                except Exception as e:
                    warnings.warn(f"설정 파일 로딩 실패 ({config_path}): {e}")
        
        # 디버깅: 찾은 후보들과 현재 디렉토리 정보 출력
        print(f"🔍 설정 파일 감지 실패")
        print(f"   - 검색 경로: {search_dir}")
        print(f"   - 현재 디렉토리: {Path.cwd()}")
        print(f"   - 체크포인트 경로: {checkpoint_path}")
        
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
            'vicreg_overlap': 'overlap_ratio',
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
