# coding: utf-8
"""
PanoLLaVA Comprehensive Model Evaluation System
─────────────────────────────────────────────────

단계별 평가 시스템:
1. 모델 및 LoRA 가중치 로드
2. 테스트 데이터셋 준비 (ChatPanoTestDataset, VLMDataModule)
3. 배치별 텍스트 생성 (generate)
4. 예측/정답 텍스트 저장 및 로깅
5. 평가 메트릭 계산 (BLEU, ROUGE, METEOR, SPICE, CIDEr, CLIP-S, RefCLIP-S)

사용법:
    # 방법 1: Config 기반 평가 (자동 체크포인트 탐색)
    python eval.py --config config.yaml --csv-input data/quic360/test.csv
    
    # 방법 2: 체크포인트 디렉토리 직접 지정 (권장) ✨
    python eval.py --checkpoint-dir runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/ \\
                   --csv-input data/quic360/test.csv
    
    # 방법 3: 체크포인트 파일 명시적 지정 (가장 직접적) ✨✨
    python eval.py --checkpoint runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/best.ckpt \\
                   --csv-input data/quic360/test.csv
    
    # 방법 4: 메타데이터 기반 자동 설정 (config 불필요)
    python eval.py --checkpoint-dir runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/
    # → checkpoint_metadata.json에서 모든 설정 자동 로드
    # → best.ckpt 또는 last.ckpt 자동 선택
    
    # 방법 5: CSV 메트릭 전용 모드 (모델 로딩 생략) 🚀
    python eval.py --csv-input results/model_predictions_20251113.csv
    # → CSV에 'prediction'/'reference' 컬럼이 있으면 자동으로 메트릭만 계산
    # → 모델 로딩과 생성 과정을 완전히 건너뜀 (빠른 메트릭 재계산)
    
주요 기능:
    - checkpoint_metadata.json 자동 로드 (모델 설정, 하이퍼파라미터)
    - best.ckpt/last.ckpt 심볼릭 링크 우선 사용
    - LoRA 가중치 자동 탐색 및 로드
"""

import argparse
import torch
import json
import logging
import time
import traceback
import os
import sys
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# Add src to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from panovlm.config.loader import load_config_dict as _load_train_config_dict
from panovlm.runtime.model_factory import ModelFactory

# 내부 모듈
# Silence HF tokenizers fork/parallelism warnings and avoid deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from panovlm.dataset import VLMDataModule
from panovlm.processors.universal_text_formatter import UniversalTextFormatter

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


_STAGE_CANONICAL_MAP = {
    "vision": ("vision", "vision_pretraining", "vision_pretrain"),
    "resampler": ("resampler", "resampler_training"),
    "finetune": ("finetune", "instruction_tuning", "instruction_tune"),
    "generate": ("generate", "inference"),
}

_STAGE_VARIANT_LOOKUP = {
    variant: canonical
    for canonical, variants in _STAGE_CANONICAL_MAP.items()
    for variant in variants
}


def _infer_prefix_from_runs(runs_root: Path, crop_strategy: str, stage_names: List[str], resampler: str) -> Optional[str]:
    if not runs_root.exists() or not runs_root.is_dir():
        return None

    search_stage_names = stage_names or []
    patterns: List[Tuple[str, Optional[str]]] = []

    for stage_name in search_stage_names:
        patterns.append((f"*_{crop_strategy}_{stage_name}_{resampler}", stage_name))

    if not patterns:
        patterns.append((f"*_{crop_strategy}_*_{resampler}", None))

    for pattern, stage_hint in patterns:
        try:
            matches = sorted(
                runs_root.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
        except Exception:
            matches = sorted(runs_root.glob(pattern))
        if not matches:
            continue
        for match in matches:
            name = match.name
            if stage_hint:
                suffix = f"_{crop_strategy}_{stage_hint}_{resampler}"
                if not name.endswith(suffix):
                    continue
                prefix_candidate = name[:-len(suffix)]
                if prefix_candidate.endswith('_'):
                    prefix_candidate = prefix_candidate[:-1]
            else:
                if not name.endswith(f"_{resampler}"):
                    continue
                base = name[: -len(f"_{resampler}")]
                if base.endswith('_'):
                    base = base[:-1]
                token = f"_{crop_strategy}_"
                idx = base.rfind(token)
                if idx == -1:
                    continue
                prefix_candidate = base[:idx]
            if prefix_candidate:
                logger.info(f"🔍 Inferred prefix '{prefix_candidate}' from runs/{name}")
                return prefix_candidate

    return None


def _stage_variants(stage: Optional[str]) -> List[str]:
    """Return ordered unique stage variants covering historical aliases."""
    if stage is None:
        return []

    stage_key = str(stage).strip()
    canonical = _STAGE_VARIANT_LOOKUP.get(stage_key)
    if canonical:
        variants = _STAGE_CANONICAL_MAP[canonical]
        # Preserve order while removing duplicates
        seen = set()
        return [s for s in variants if not (s in seen or seen.add(s))]

    return [stage_key] if stage_key else []


def load_checkpoint_metadata(ckpt_path: Path) -> Optional[Dict[str, Any]]:
    """
    체크포인트 디렉토리(또는 체크포인트 파일의 부모)에서 checkpoint_metadata.json을 로드합니다.
    
    Args:
        ckpt_path: 체크포인트 디렉토리 또는 개별 체크포인트 파일 경로
        
    Returns:
        메타데이터 딕셔너리 또는 None (파일이 없는 경우)
    """
    ckpt_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    if not ckpt_dir.exists():
        logger.warning(f"⚠️ 체크포인트 경로를 찾을 수 없습니다: {ckpt_dir}")
        return None
    
    metadata_path = ckpt_dir / "checkpoint_metadata.json"
    if not metadata_path.exists():
        logger.warning(f"⚠️ 메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"✅ 메타데이터 로드 성공: {metadata_path}")
        return metadata
    except Exception as e:
        logger.warning(f"⚠️ 메타데이터 로드 실패: {e}")
        return None


def find_checkpoint_in_dir(ckpt_path: Path) -> Optional[Path]:
    """
    디렉토리에서 체크포인트 파일을 찾거나, 입력이 이미 .ckpt 파일이면 그대로 반환합니다.
    우선순위: best.ckpt > last.ckpt > *.ckpt (최신)
    
    Args:
        ckpt_path: 체크포인트 디렉토리 또는 개별 체크포인트 파일 경로
        
    Returns:
        체크포인트 파일 경로 또는 None
    """
    # 파일이 직접 주어졌다면 그대로 사용 (.ckpt 확장자만 허용)
    if ckpt_path.is_file():
        if ckpt_path.suffix == ".ckpt":
            logger.info(f"✅ Using explicit checkpoint: {ckpt_path}")
            return ckpt_path
        logger.warning(f"⚠️ 지원하지 않는 체크포인트 파일 형식: {ckpt_path}")
        return None
    
    ckpt_dir = ckpt_path
    # 1. 심볼릭 링크 우선 (best.ckpt)
    best_ckpt = ckpt_dir / "best.ckpt"
    if best_ckpt.exists():
        # 심볼릭 링크인 경우 실제 경로로 해석
        resolved = best_ckpt.resolve() if best_ckpt.is_symlink() else best_ckpt
        logger.info(f"✅ Using best checkpoint: {resolved}")
        return resolved
    
    # 2. last.ckpt
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        resolved = last_ckpt.resolve() if last_ckpt.is_symlink() else last_ckpt
        logger.info(f"✅ Using last checkpoint: {resolved}")
        return resolved
    
    # 3. 가장 최근 .ckpt 파일 (수정 시간 기준)
    try:
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        if ckpt_files:
            latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"✅ Using latest checkpoint: {latest_ckpt}")
            return latest_ckpt
    except Exception as e:
        logger.warning(f"⚠️ 체크포인트 파일 검색 실패: {e}")
    
    return None


def resolve_model_dir(config_or_path, stage: str = None, crop_strategy: str = None) -> str:
    """
    HF-style 모델 디렉토리 자동 탐색 (PyTorch bin 기반)
    - config_or_path: dict 또는 JSON 파일 경로(str)
    - stage/crop_strategy: runs/<prefix>_<crop>_<stage>_<resampler>/hf_model 힌트 구성에 사용
    """
    try:
        # config 로딩 (dict 또는 파일 경로)
        if isinstance(config_or_path, (str, Path)):
            config = _load_train_config_dict(str(config_or_path))
        elif isinstance(config_or_path, dict):
            config = config_or_path
        else:
            raise TypeError(f"Unsupported config type: {type(config_or_path)}")

        resampler = config.get('models', {}).get('resampler_type') or config.get('models', {}).get('resampler', 'mlp')
        if stage is None:
            stage = config.get('training', {}).get('default_stage', 'finetune')

        if crop_strategy is None:
            crop_strategy = config.get('image_processing', {}).get('crop_strategy', 'e2p')

        stage_variants = _stage_variants(stage)

        prefix = config.get('training', {}).get('prefix')
        if not prefix:
            prefix = config.get('experiment', {}).get('name') if isinstance(config.get('experiment'), dict) else None
        if not prefix:
            prefix = config.get('experiment', {}).get('id') if isinstance(config.get('experiment'), dict) else None
        if not prefix:
            prefix = _infer_prefix_from_runs(
                Path(config.get('paths', {}).get('runs_dir', 'runs')),
                crop_strategy,
                stage_variants,
                resampler
            )
        if not prefix:
            raise KeyError("training.prefix is required in config.json")

        # 추가: pretrained_dir 지원 및 HF 디렉토리/체크포인트 자동 탐색
        paths_cfg = config.get('paths', {}) if isinstance(config, dict) else {}
        pretrained_dir = paths_cfg.get('pretrained_dir')
        if pretrained_dir and Path(pretrained_dir).exists():
            p = Path(pretrained_dir)
            if p.is_file() and p.suffix == '.ckpt':
                logger.info(f"✅ Using checkpoint from config: {pretrained_dir}")
            else:
                logger.info(f"✅ Using pretrained_dir from config: {pretrained_dir}")
            return str(p)

        # runs 디렉토리 내 hf_model 폴더 자동 탐색
        runs_root = Path(paths_cfg.get('runs_dir', 'runs'))

        def try_from_run_dir(run_dir: Path, stage_hint: str) -> Optional[str]:
            if not run_dir.exists() or not run_dir.is_dir():
                return None

            hf_dir = run_dir / 'hf_model'
            if hf_dir.exists() and hf_dir.is_dir():
                logger.info(f"✅ Using HF model dir (stage='{stage_hint}'): {str(hf_dir)}")
                return str(hf_dir)

            pano_dir = run_dir / 'panorama_model'
            if pano_dir.exists() and pano_dir.is_dir():
                logger.info(f"✅ Using panorama_model dir (stage='{stage_hint}'): {str(pano_dir)}")
                return str(pano_dir)

            best_ckpt = run_dir / 'best.ckpt'
            if best_ckpt.exists():
                logger.info(f"✅ Using best checkpoint (stage='{stage_hint}'): {str(best_ckpt)}")
                return str(best_ckpt)

            last_ckpt = run_dir / 'last.ckpt'
            if last_ckpt.exists():
                logger.info(f"✅ Using last checkpoint (stage='{stage_hint}'): {str(last_ckpt)}")
                return str(last_ckpt)

            try:
                any_ckpts = sorted(run_dir.glob('*.ckpt'))
                if any_ckpts:
                    logger.info(f"✅ Using checkpoint (stage='{stage_hint}'): {str(any_ckpts[0])}")
                    return str(any_ckpts[0])
            except Exception:
                pass

            return None

        checked_stages = set()
        for stage_name in stage_variants:
            checked_stages.add(stage_name)
            candidate_run_dir = runs_root / f"{prefix}_{crop_strategy}_{stage_name}_{resampler}"
            resolved = try_from_run_dir(candidate_run_dir, stage_name)
            if resolved:
                return resolved

        # wildcard fallback: scan any stage between prefix/crop/resampler
        pattern_prefix = f"{prefix}_{crop_strategy}_"
        pattern_suffix = f"_{resampler}"
        wildcard_pattern = f"{pattern_prefix}*{pattern_suffix}"
        for run_dir in sorted(runs_root.glob(wildcard_pattern)):
            stage_hint = run_dir.name[len(pattern_prefix):-len(pattern_suffix)] if len(pattern_suffix) > 0 else run_dir.name[len(pattern_prefix):]
            if stage_hint in checked_stages:
                continue
            resolved = try_from_run_dir(run_dir, stage_hint or "unknown")
            if resolved:
                return resolved

        raise FileNotFoundError("No pretrained model dir found. Set paths.pretrained_dir or pass --model-dir")

    except Exception as e:
        logger.error(f"Failed to resolve model dir: {e}")
        raise



def load_model_and_lora(
    model_dir: str,
    lora_weights_path: Optional[str],
    device: torch.device,
    config_path: Optional[str] = None,
    config_data: Optional[Dict[str, Any]] = None,
    **model_kwargs
):
    """
    1단계: 체크포인트와 LoRA 가중치를 로드하여 생성용 모델 준비 (설정 시스템 통합)
    - 새로운 PanoramaVLM 인터페이스 우선 시도
    - 실패 시 VLMModule 폴백 (이때 model_config를 반드시 전달)
    """
    logger.info("=" * 60)
    logger.info("🚀 1단계: 모델 및 LoRA 가중치 로드 (설정 시스템 통합)")
    logger.info("=" * 60)

    # 디바이스 문자열
    device_str = str(device) if device != "auto" else "auto"

    # config 객체 준비 (ModelConfig 또는 dict)
    config_obj = None
    if config_data is not None:
        if isinstance(config_data, dict):
            config_obj = config_data
        else:
            logger.warning("config_data is not a dict; ignoring runtime config override")
    elif config_path:
        try:
            from panovlm.config import ModelConfig
            try:
                config_obj = ModelConfig.load(config_path)
                logger.info(f"📋 ModelConfig 로드 완료(from {config_path})")
            except Exception as e:
                logger.warning(f"ModelConfig.load 실패, config dict로 대체: {e}")
                config_obj = _load_train_config_dict(str(config_path))
        except Exception as e:
            logger.warning(f"panovlm.config.ModelConfig 사용 불가, config dict로 대체: {e}")
            config_obj = _load_train_config_dict(str(config_path))

    # ── PanoramaVLM (HF 디렉토리 또는 .ckpt) ──────────────────────
    try:
        from panovlm.models.model import PanoramaVLM

        model_factory = None
        if config_obj is not None:
            from panovlm.config import ModelConfig as _ModelConfig

            if isinstance(config_obj, _ModelConfig):
                model_factory = ModelFactory(config_obj)
            elif isinstance(config_obj, dict):
                try:
                    model_factory = ModelFactory(_ModelConfig.from_dict(config_obj))
                except Exception:
                    model_factory = None

        extra_cfg = {}
        if config_obj is not None:
            extra_cfg["config"] = config_obj
            extra_cfg["model_config"] = config_obj

        mpath = Path(model_dir)
        if model_factory is not None:
            if mpath.is_file() and mpath.suffix == ".ckpt":
                logger.info(f"📦 Loading from checkpoint: {str(mpath)} (factory)")
                model = model_factory.load_checkpoint(
                    str(mpath),
                    device=device_str,
                    **{k: v for k, v in model_kwargs.items() if v is not None},
                )
            else:
                model = model_factory.load_pretrained_dir(
                    str(mpath),
                    device=device_str,
                    **{k: v for k, v in model_kwargs.items() if v is not None},
                )
        else:
            if mpath.is_file() and mpath.suffix == ".ckpt":
                logger.info(f"📦 Loading from checkpoint: {str(mpath)}")
                model = PanoramaVLM.from_checkpoint(
                    str(mpath),
                    device=device_str,
                    **extra_cfg,
                    **{k: v for k, v in model_kwargs.items() if v is not None}
                )
            else:
                model = PanoramaVLM.from_pretrained_dir(
                    str(mpath),
                    device=device_str,
                    **extra_cfg,
                    **{k: v for k, v in model_kwargs.items() if v is not None}
                )

        # 설정 정보 로그
        if hasattr(model, "config") and model.config:
            logger.info("📋 Model Configuration 요약:")
            for k in [
                "vision_name", "language_model_name", "latent_dimension",
                "image_size", "crop_strategy", "use_lora", "lora_r", "lora_alpha"
            ]:
                try:
                    val = getattr(model.config, k, None)
                except Exception:
                    val = None
                if val is not None:
                    logger.info(f"   - {k}: {val}")

        # 기존 코드와 호환을 위한 래퍼
        class ModelWrapper:
            def __init__(self, panorama_model):
                self.model = panorama_model
                self._stage_key = "finetune"
            def eval(self):
                self.model.eval(); return self
            def to(self, dev):
                self.model = self.model.to(dev); return self

        wrapped_model = ModelWrapper(model).eval()
        logger.info(f"✓ 모델 준비 완료 - Device: {device}")
        return wrapped_model

    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {e}")
        raise

def prepare_test_dataset(
    csv_input: str,
    batch_size: int,
    max_text_length: str | int,
    crop_strategy: str,
    lm_name: str,
    num_workers: int = 0,
    overlap_ratio: float = 0.5,
    *,
    image_size: Tuple[int, int] | List[int] | None = None,
    fov_deg: float = 90.0,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    anyres_patch_size: Optional[int] = None,  # None이면 image_size에서 자동 추론
    anyres_max_patches: int = 12,
    normalize: bool = True,
    vision_name: Optional[str] = None,
    system_msg: Optional[str] = None,
    use_vision_processor: bool = True,
    auto_max_text_length_cap: Optional[int] = None,
    auto_max_text_length_floor: Optional[int] = None,
    auto_max_text_length_scan_limit: Optional[int] = None
) -> Tuple[VLMDataModule, Any]:
    """
    2단계: ChatPanoTestDataset과 VLMDataModule을 활용한 테스트 데이터 준비
    - config.json의 image_processing/ training 내용을 인자화하여 반영
    """
    logger.info("=" * 60)
    logger.info("📊 2단계: 테스트 데이터셋 준비")
    logger.info("=" * 60)

    logger.info(f"📂 CSV 입력: {csv_input}")
    system_msg = system_msg or "You are an expert assistant specialized in analyzing panoramic images. Please provide detailed, accurate, and helpful responses about what you observe in the panoramic view shortly."

    # Normalize image_size
    _img_size = None
    if image_size is not None:
        try:
            if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
                _img_size = (int(image_size[0]), int(image_size[1]))
        except Exception:
            _img_size = None

    datamodule = VLMDataModule(
        csv_train=csv_input,
        csv_val=csv_input,  # 평가용으로 동일한 파일 사용
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer_name=lm_name,
        max_text_length=max_text_length,
        image_size=_img_size or (224, 224),
        crop_strategy=crop_strategy,
        eval_mode=True,
        system_msg=system_msg,
        overlap_ratio=overlap_ratio,
        fov_deg=fov_deg,
        image_mean=image_mean,
        image_std=image_std,
        anyres_patch_size=anyres_patch_size,
        anyres_max_patches=anyres_max_patches,
        normalize=normalize,
        vision_model_name=vision_name,
        use_vision_processor=use_vision_processor,
        auto_max_text_length_cap=int(auto_max_text_length_cap) if auto_max_text_length_cap is not None else 8192,
        auto_max_text_length_floor=int(auto_max_text_length_floor) if auto_max_text_length_floor is not None else None,
        auto_max_text_length_scan_limit=int(auto_max_text_length_scan_limit) if auto_max_text_length_scan_limit is not None else None
    )

    datamodule.setup()
    test_dataloader = datamodule.val_dataloader()

    logger.info(f"✓ 데이터셋 준비 완료")
    logger.info(f"   - 총 배치 수: {len(test_dataloader)}")
    logger.info(f"   - 배치 크기: {batch_size}")
    logger.info(f"   - 텍스트 최대 길이 (requested): {max_text_length}")
    logger.info(f"   - 크롭 전략: {crop_strategy}")
    logger.info(f"   - 겹침 비율: {overlap_ratio}")
    logger.info(f"   - 워커 수: {num_workers}")
    logger.info(f"   - Vision 모델: {vision_name}")
    logger.info(f"   - 이미지 크기: {(_img_size or (224, 224))}")
    logger.info(f"   - fov_deg: {fov_deg}")
    logger.info(f"   - normalize: {normalize} | use_vision_processor: {use_vision_processor}")
    if image_mean is not None and image_std is not None:
        logger.info(f"   - image_mean/std: {image_mean} / {image_std}")
    logger.info(f"   - use_vision_processor: {use_vision_processor}")

    return datamodule, test_dataloader

def generate_predictions(
    model: Any,
    test_dataloader,
    datamodule: VLMDataModule,
    device: torch.device,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    repetition_penalty: float = 1.1,
    length_penalty: float = 1.0,
    min_new_tokens: int = 5,
    system_msg: Optional[str] = None,
    max_samples: Optional[int] = None,
    log_samples: bool = True,
    log_interval: int = 25,
    log_max_samples: int = 50,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    3단계: 테스트 데이터에서 배치별 텍스트 생성
    - config.training.system_msg(또는 system_messages.default) 를 UniversalTextFormatter에 반영
    """
    logger.info("=" * 60)
    logger.info("🤖 3단계: 텍스트 생성 (UniversalTextFormatter 활용)")
    logger.info("=" * 60)

    predictions, references, image_paths, input_texts = [], [], [], []

    tokenizer = datamodule.tokenizer
    sys_msg = system_msg or "You are an expert assistant specialized in analyzing panoramic images. Please provide detailed, accurate, and helpful responses about what you observe in the panoramic view shortly."
    text_formatter = UniversalTextFormatter(
        tokenizer,
        system_msg=sys_msg
    )

    logger.info(f"🎯 생성 파라미터 - Max tokens: {max_new_tokens}, Min tokens: {min_new_tokens}, Temperature: {temperature}, Top P: {top_p}, Top K: {top_k}")
    logger.info(f"📝 텍스트 포맷터 - 모델: {text_formatter.model_family} ({'Instruct' if text_formatter.is_instruct else 'Base'})")
    if max_samples is not None:
        logger.info(f"🔢 최대 평가 샘플 수 제한: {max_samples}")
    if not log_samples:
        logger.info("🛑 상세 샘플 로그 비활성화 ( --log-samples 로 활성화 가능 )")

    with torch.no_grad():
        total_logged_samples = 0
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="생성 중")):
            try:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                input_ids = batch.get("input_ids")
                if input_ids is not None:
                    input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device, non_blocking=True)

                batch_size = pixel_values.shape[0]

                # 간소화된 정답·메타 추출
                batch_references = []
                if "reference" in batch:
                    refs = batch["reference"]
                    batch_references = [str(r).strip() for r in (refs if isinstance(refs, list) else [refs]*batch_size)]
                else:
                    batch_references = [f"no_reference_{i}" for i in range(batch_size)]

                batch_image_paths = batch.get("image_path", [f"batch_{batch_idx}_sample_{i}" for i in range(batch_size)])
                batch_input_texts = batch.get("original_query", batch.get("input_text", [f"no_query_{i}" for i in range(batch_size)]))
                if not isinstance(batch_input_texts, list):
                    batch_input_texts = [batch_input_texts] * batch_size

                generation_config = text_formatter.get_generation_config()
                gen_kwargs = {
                    "pixel_values": pixel_values,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    "length_penalty": length_penalty,
                    "min_new_tokens": min_new_tokens,
                    "do_sample": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if hasattr(model, 'model') and hasattr(model.model, 'generation_config'):
                    if hasattr(model.model.generation_config, 'stop_strings'):
                        gen_kwargs["stop_strings"] = generation_config["stop_strings"][:3]

                if hasattr(model, 'model') and hasattr(model.model, 'generate'):
                    output = model.model.generate(**gen_kwargs)
                elif hasattr(model, 'generate'):
                    output = model.generate(**gen_kwargs)
                else:
                    raise AttributeError("모델에 generate 메서드가 없습니다")

                batch_predictions = []
                if isinstance(output, torch.Tensor):
                    for i in range(batch_size):
                        input_length = input_ids[i].shape[0] if input_ids is not None else 0
                        generated_tokens = output[i][input_length:]
                        raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        clean_prediction = text_formatter.extract_assistant_response(raw_text)
                        batch_predictions.append(clean_prediction)
                elif isinstance(output, dict) and "text" in output:
                    for raw_text in output["text"]:
                        clean_prediction = text_formatter.extract_assistant_response(raw_text)
                        batch_predictions.append(clean_prediction)
                else:
                    logger.warning(f"Unexpected output format: {type(output)}")
                    batch_predictions = ["[생성 실패]"] * batch_size

                # 크기 정합
                if len(batch_predictions) != batch_size:
                    if len(batch_predictions) < batch_size:
                        batch_predictions.extend(["[크기 부족]"] * (batch_size - len(batch_predictions)))
                    else:
                        batch_predictions = batch_predictions[:batch_size]

                # 정리
                cleaned_predictions = []
                for pred in batch_predictions:
                    cleaned_predictions.append(pred.strip().replace('\n\n', '\n') if pred and pred.strip() else "[빈 응답]")

                # 로그 & 축적
                should_log_batch = log_samples and (
                    batch_idx == 0
                    or (log_interval > 0 and (batch_idx + 1) % log_interval == 0)
                )
                if should_log_batch and total_logged_samples < log_max_samples:
                    logger.info(f"=== 배치 {batch_idx} 결과 로그 ===")
                    for i, (pred, ref) in enumerate(zip(cleaned_predictions, batch_references)):
                        if total_logged_samples >= log_max_samples:
                            break
                        logger.info(f"  샘플 {len(predictions) + i}")
                        logger.info(f"    예측: '{pred}'")
                        logger.info(f"    정답: '{ref}'")
                        total_logged_samples += 1
                    logger.info(f"==========================")

                predictions.extend(cleaned_predictions)
                references.extend(batch_references)
                image_paths.extend(batch_image_paths)
                input_texts.extend(batch_input_texts)

                if max_samples is not None and len(predictions) >= max_samples:
                    overflow = len(predictions) - max_samples
                    if overflow > 0:
                        del predictions[-overflow:]
                        del references[-overflow:]
                        del image_paths[-overflow:]
                        del input_texts[-overflow:]
                    logger.info(f"📉 최대 샘플 수 {max_samples}에 도달하여 조기 중단합니다.")
                    break

                if batch_idx % 10 == 0:
                    logger.info(f"진행: {batch_idx + 1}/{len(test_dataloader)} 배치 완료 ({len(predictions)} 샘플)")

            except Exception as e:
                logger.error(f"배치 {batch_idx} 전체 처리 실패: {e}", exc_info=True)
                bs = pixel_values.shape[0] if 'pixel_values' in locals() else 1
                predictions.extend([f"[배치 오류_{i}]" for i in range(bs)])
                references.extend(batch_references if 'batch_references' in locals() else [f"[정답 없음_{i}]" for i in range(bs)])
                image_paths.extend(batch_image_paths if 'batch_image_paths' in locals() else [f"error_batch_{batch_idx}_sample_{i}" for i in range(bs)])
                input_texts.extend(batch_input_texts if 'batch_input_texts' in locals() else [f"error_input_{i}" for i in range(bs)])
                continue

        if max_samples is not None and len(predictions) > max_samples:
            overflow = len(predictions) - max_samples
            del predictions[-overflow:]
            del references[-overflow:]
            del image_paths[-overflow:]
            del input_texts[-overflow:]

    logger.info(f"✓ 텍스트 생성 완료! 총 샘플 수: {len(predictions)}")
    return predictions, references, image_paths, input_texts



def save_and_log_results(
    predictions: List[str],
    references: List[str],
    image_paths: List[str],
    input_texts: List[str],
    output_dir: Path,
    timestamp: str,
    prefix: str,
) -> pd.DataFrame:
    """
    4단계: 생성된 답변과 정답 텍스트를 저장하고 로깅 (개선된 분석 포함)
    """
    logger.info("=" * 60)
    logger.info("💾 4단계: 결과 저장 및 분석")
    logger.info("=" * 60)
    
    # 개선된 CSV 데이터 준비
    results_data = []
    for i, (pred, ref, img_path) in enumerate(zip(predictions, references, image_paths)):
        # 빈 값 처리 및 기본 정리
        pred_str = str(pred).strip() if pred is not None else ""
        ref_str = str(ref).strip() if ref is not None else ""
        img_path_str = str(img_path) if img_path is not None else ""
        
        # 예측값 품질 분석
        is_error = pred_str.startswith('[') and pred_str.endswith(']')
        is_empty = not pred_str or pred_str in ["", "[빈 응답]"]
        
        # input_text 추출 (인덱스 확인 후 안전하게)
        input_text_str = ""
        if i < len(input_texts):
            input_text_str = str(input_texts[i]).strip() if input_texts[i] is not None else ""
        
        results_data.append({
            'sample_id': i,
            'image_path': img_path_str,
            'original_query': input_text_str,
            'prediction': pred_str,
            'reference': ref_str,
            'pred_length': len(pred_str.split()),
            'ref_length': len(ref_str.split()),
            'is_error': is_error,
            'is_empty': is_empty
        })
    
    # DataFrame 생성 및 저장
    df = pd.DataFrame(results_data)
    safe_prefix = prefix if prefix else "model"
    csv_path = output_dir / f"{safe_prefix}_predictions_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 개선된 결과 통계 분석
    total_samples = len(df)
    error_count = df['is_error'].sum()
    empty_count = df['is_empty'].sum()
    valid_count = total_samples - error_count - empty_count
    
    # 길이 통계 (유효한 예측값만)
    valid_df = df[~df['is_error'] & ~df['is_empty']]
    if len(valid_df) > 0:
        avg_pred_length = valid_df['pred_length'].mean()
        avg_ref_length = valid_df['ref_length'].mean()
        pred_length_std = valid_df['pred_length'].std()
    else:
        avg_pred_length = avg_ref_length = pred_length_std = 0.0
    
    logger.info(f"📊 생성 품질 분석:")
    logger.info(f"   - 총 샘플: {total_samples}")
    logger.info(f"   - 성공적 생성: {valid_count}개 ({valid_count/total_samples*100:.1f}%)")
    logger.info(f"   - 생성 오류: {error_count}개 ({error_count/total_samples*100:.1f}%)")
    logger.info(f"   - 빈 응답: {empty_count}개 ({empty_count/total_samples*100:.1f}%)")
    
    if valid_count > 0:
        logger.info(f"📝 텍스트 길이 분석:")
        logger.info(f"   - 평균 예측 길이: {avg_pred_length:.1f} ± {pred_length_std:.1f} 단어")
        logger.info(f"   - 평균 정답 길이: {avg_ref_length:.1f} 단어")
        logger.info(f"   - 길이 비율 (예측/정답): {avg_pred_length/avg_ref_length:.2f}")
    
    logger.info(f"💾 결과 저장 완료: {csv_path}")
    return df
    


def basic_cleanup(text: str) -> str:
    """
    Level 1: 기본 정리 - 모델 아티팩트만 제거 (의미 보존)

    - 특수 토큰 제거 (<image>, <|im_start|> 등)
    - 역할 태그 제거 (ASSISTANT:, USER: 등)
    - <think> 태그 및 내용 완전 제거
    - 메타 텍스트 패턴 제거 ("Okay, let's...", "First, I need to..." 등)
    - 프롬프트 누수 제거
    - 과도한 공백 정리

    대소문자, 구두점은 보존하여 실제 품질을 반영합니다.
    """
    if not text or pd.isna(text):
        return ""

    text = str(text)

    # 1. <think>...</think> 태그와 내용 완전 제거 (줄바꿈 포함)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # 남은 </think> 태그도 제거
    text = re.sub(r'</think>', '', text, flags=re.IGNORECASE)

    # 2. 메타 텍스트 패턴 제거 (모델의 사고 과정)
    # "Okay, let's..." 형태의 문장 제거
    text = re.sub(r'^(Okay|Alright|Well|So),?\s+(let\'?s?|I\'?ll?|we\'?ll?)\s+.*?\.\s*', '', text, flags=re.IGNORECASE)
    # "First, I need to..." 형태의 문장 제거  
    text = re.sub(r'^(First|Then|Next|Now),?\s+(I|we)\s+(need to|should|will|can)\s+.*?\.\s*', '', text, flags=re.IGNORECASE)
    # "The user mentioned..." 형태의 문장 제거
    text = re.sub(r'^The (user|question|query|prompt)\s+(mentioned|asked|provided|wants).*?\.\s*', '', text, flags=re.IGNORECASE)
    # "Looking at..." 형태의 문장 제거
    text = re.sub(r'^(Looking at|Analyzing|Examining|Considering)\s+.*?\.\s*', '', text, flags=re.IGNORECASE)

    # 3. 특수 토큰 제거
    text = re.sub(r"<\|.*?\|>|<image>|</image>|<img>|</img>", " ", text, flags=re.I)
    text = re.sub(r"<vision_start>|<vision_end>|<image_pad>", " ", text, flags=re.I)

    # 4. 역할 태그 제거 (문장 시작 부분에서)
    text = re.sub(r"^(USER:|ASSISTANT:|Question:|Answer:)\s*", "", text, flags=re.I)

    # 5. 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_evaluation_metrics(data_input, output_dir: Path, timestamp: str, prefix: str) -> Dict[str, float]:
    """
    5단계: 평가 메트릭 계산 (BLEU-4, METEOR, ROUGE-L, SPICE, CIDEr)

    Args:
        data_input: pandas DataFrame 또는 CSV 파일 경로 (str/Path)
        output_dir: 결과 저장 디렉토리
        timestamp: 타임스탬프 문자열
        prefix: 결과 파일 접두어

    Changes:
        - sacrebleu 사용 (표준 토큰화, 재현 가능한 BLEU)
        - basic_cleanup으로 특수 토큰/역할 태그 제거
        - 대소문자/구두점 보존 (실제 품질 반영)
    """
    logger.info("=" * 60)
    logger.info("📈 5단계: 평가 메트릭 계산")
    logger.info("=" * 60)
    
    # 입력 데이터 처리: CSV 파일이면 DataFrame으로 변환
    if isinstance(data_input, (str, Path)):
        csv_path = Path(data_input)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
        
        logger.info(f"📂 CSV 파일 로드: {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8')
        logger.info(f"✓ DataFrame 변환 완료 - 총 {len(df)}개 샘플")
        
        # 필수 컬럼 확인
        required_columns = ['prediction', 'reference']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV 파일에 필수 컬럼이 없습니다: {missing_columns}. 필요한 컬럼: {required_columns}")
        
        # 옵션 컬럼 확인 및 로그
        optional_columns = ['image_path']
        available_optional = [col for col in optional_columns if col in df.columns]
        logger.info(f"📊 사용 가능한 컬럼: 필수 {required_columns} + 선택 {available_optional}")
    
    elif isinstance(data_input, pd.DataFrame):
        df = data_input
        logger.info(f"✓ DataFrame 입력 - 총 {len(df)}개 샘플")
    else:
        raise TypeError(f"지원하지 않는 데이터 타입: {type(data_input)}. pandas DataFrame 또는 CSV 파일 경로를 입력하세요.")
    
    # 유효한 샘플만 선택 (예측과 정답가 모두 비어있지 않은 경우)
    valid_df = df[(df['prediction'].str.strip() != '') & (df['reference'].str.strip() != '')]
    
    if len(valid_df) == 0:
        logger.error("❌ 유효한 샘플이 없습니다.")
        return {}
    
    logger.info(f"📊 평가 대상: {len(valid_df)}/{len(df)} 샘플")
    
    # 안전한 텍스트 추출 (NaN 값 처리)
    raw_predictions = [str(pred) if pred is not None and not pd.isna(pred) else "" for pred in valid_df['prediction'].tolist()]
    raw_references = [str(ref) if ref is not None and not pd.isna(ref) else "" for ref in valid_df['reference'].tolist()]

    # Level 1 정리: 특수 토큰, 역할 태그 제거
    logger.info("🧹 텍스트 정리 중 (특수 토큰/역할 태그 제거)...")
    predictions = [basic_cleanup(pred) for pred in raw_predictions]
    references = [basic_cleanup(ref) for ref in raw_references]

    # "Assistant:" 부분 처리 (이미 basic_cleanup에서 제거되지만 추가 체크)
    ref_texts_cleaned = []
    for ref in references:
        if "Assistant:" in ref:
            assistant_part = ref.split("Assistant:")[-1].strip()
            ref_texts_cleaned.append(assistant_part)
        else:
            ref_texts_cleaned.append(ref)
    references = ref_texts_cleaned

    # 빈 문자열 필터링
    valid_pairs = [(pred, ref) for pred, ref in zip(predictions, references) if pred.strip() and ref.strip()]

    if not valid_pairs:
        logger.error("❌ 유효한 예측-정답 쌍이 없습니다.")
        return {}

    predictions, references = zip(*valid_pairs)
    predictions = list(predictions)
    references = list(references)

    logger.info(f"📊 최종 평가 대상: {len(valid_pairs)} 샘플")
    logger.info(f"📝 예시 - 예측: '{predictions[0][:100]}...'")
    logger.info(f"📝 예시 - 정답: '{references[0][:100]}...'")

    metrics = {}
    
    # 1. BLEU-4 계산 (sacrebleu 공식 레포지토리 사용)
    try:
        import sacrebleu

        # sacrebleu는 문자열 리스트를 입력으로 받음
        if len(predictions) == 0 or len(references) == 0:
            logger.warning("⚠️ BLEU-4: 유효한 텍스트가 없습니다.")
            metrics['bleu4'] = 0.0
        else:
            logger.info("📊 BLEU-4 계산 중...")
            try:
                # sacrebleu 계산 (표준 설정)
                # 공식 레포지토리: https://github.com/mjpost/sacrebleu
                bleu = sacrebleu.corpus_bleu(
                    predictions,
                    [references],           # 참조는 리스트의 리스트
                    smooth_method="exp",    # 표준 스무딩
                    lowercase=False,        # 대소문자 보존 (실제 품질 반영)
                    tokenize="13a",         # Moses 토크나이저 (학술 표준)
                    use_effective_order=True  # 짧은 문장 안정화
                )
                metrics['bleu4'] = bleu.score / 100.0  # 0~1 스케일로 변환
                logger.info(f"✓ BLEU-4 (공식 sacrebleu): {metrics['bleu4']:.4f}")
                logger.info(f"  → 토큰화: 13a (Moses), 스무딩: exp, 대소문자: 보존")
            except Exception as bleu_e:
                logger.warning(f"⚠️ sacrebleu 계산 오류: {bleu_e}")
                # BLEU 폴백: NLTK 사용
                logger.info("BLEU-4 폴백: NLTK 사용...")
                try:
                    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

                    ref_tokens = [[ref.split()] for ref in references if ref.strip()]
                    pred_tokens = [pred.split() for pred in predictions if pred.strip()]

                    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
                        logger.warning("⚠️ BLEU-4: 유효한 토큰이 없습니다.")
                        metrics['bleu4'] = 0.0
                    else:
                        smoothing = SmoothingFunction().method1
                        metrics['bleu4'] = corpus_bleu(ref_tokens, pred_tokens, 
                                                       weights=(0.25, 0.25, 0.25, 0.25), 
                                                       smoothing_function=smoothing)
                        logger.info(f"✓ BLEU-4 (NLTK 폴백): {metrics['bleu4']:.4f}")
                except Exception as nltk_bleu_e:
                    logger.error(f"❌ NLTK BLEU-4도 실패: {nltk_bleu_e}")
                    metrics['bleu4'] = 0.0
                    
    except ImportError:
        logger.error("❌ sacrebleu를 설치하지 않았습니다.")
        logger.error("   권장 설치: pip install sacrebleu")
        logger.error("   github: https://github.com/mjpost/sacrebleu")
        logger.info("BLEU-4 폴백: NLTK 사용...")
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

            ref_tokens = [[ref.split()] for ref in references if ref.strip()]
            pred_tokens = [pred.split() for pred in predictions if pred.strip()]

            if len(ref_tokens) == 0 or len(pred_tokens) == 0:
                logger.warning("⚠️ BLEU-4: 유효한 토큰이 없습니다.")
                metrics['bleu4'] = 0.0
            else:
                smoothing = SmoothingFunction().method1
                metrics['bleu4'] = corpus_bleu(ref_tokens, pred_tokens, 
                                               weights=(0.25, 0.25, 0.25, 0.25), 
                                               smoothing_function=smoothing)
                logger.info(f"✓ BLEU-4 (NLTK 폴백): {metrics['bleu4']:.4f}")
        except Exception as e:
            logger.error(f"❌ NLTK BLEU-4도 실패: {e}")
            metrics['bleu4'] = 0.0
    except Exception as e:
        logger.error(f"❌ BLEU-4 계산 오류: {e}")
        metrics['bleu4'] = 0.0
    
    # 2. METEOR 계산 (공식 NLTK 레포지토리 사용)
    try:
        import nltk
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("NLTK 데이터 다운로드 중 (wordnet, punkt)...")
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)

        from nltk.translate.meteor_score import meteor_score

        logger.info("📊 METEOR 계산 중...")
        meteor_scores = []
        batch_size = 500
        
        # 배치별 처리 (진행상황 표시)
        for idx, (ref, pred) in enumerate(zip(references, predictions)):
            if (idx + 1) % batch_size == 0:
                logger.info(f"  처리 중: {idx + 1}/{len(references)}")
            
            if ref.strip() and pred.strip():  # 빈 문자열 체크
                ref_tokens = ref.split()
                pred_tokens = pred.split()
                if len(ref_tokens) > 0 and len(pred_tokens) > 0:
                    try:
                        score = meteor_score([ref_tokens], pred_tokens)
                        meteor_scores.append(score)
                    except Exception as item_e:
                        logger.debug(f"  샘플 {idx} METEOR 계산 오류: {item_e}")
                        meteor_scores.append(0.0)

        if meteor_scores:
            metrics['meteor'] = float(np.mean(meteor_scores))
            logger.info(f"✓ METEOR (공식 NLTK): {metrics['meteor']:.4f}")
        else:
            logger.warning("⚠️ METEOR: 유효한 점수가 없습니다.")
            metrics['meteor'] = 0.0
            
    except ImportError:
        logger.error("❌ NLTK를 설치하지 않았습니다.")
        logger.error("   설치: pip install nltk")
        metrics['meteor'] = 0.0
    except Exception as e:
        logger.error(f"❌ METEOR 계산 오류: {e}")
        metrics['meteor'] = 0.0

    # 3. ROUGE-L 계산 (공식 rouge-score 라이브러리 사용)
    try:
        from rouge_score import rouge_scorer
        
        logger.info("📊 ROUGE-L 계산 중 (메모리 효율적 처리)...")
        
        # ROUGE는 매우 큰 데이터에서 메모리 사용량이 많을 수 있으므로 배치 처리
        rouge_scores = []
        batch_size = 100
        
        # 배치별 처리
        for batch_idx in range(0, len(predictions), batch_size):
            batch_end = min(batch_idx + batch_size, len(predictions))
            batch_preds = predictions[batch_idx:batch_end]
            batch_refs = references[batch_idx:batch_end]
            
            if (batch_idx + batch_size) % 500 == 0:
                logger.info(f"  처리 중: {batch_end}/{len(predictions)}")
            
            # 각 배치마다 새로운 scorer 생성 (메모리 관리)
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            
            for ref, pred in zip(batch_refs, batch_preds):
                if ref.strip() and pred.strip():  # 빈 문자열 체크
                    try:
                        scores = scorer.score(ref, pred)
                        rouge_scores.append(scores['rougeL'].fmeasure)
                    except Exception as item_e:
                        # 개별 샘플 오류는 스킵하고 계속 진행
                        logger.debug(f"  샘플 ROUGE-L 계산 오류: {item_e}")
                        rouge_scores.append(0.0)

        if rouge_scores:
            metrics['rougeL'] = float(np.mean(rouge_scores))
            logger.info(f"✓ ROUGE-L (공식 rouge-score): {metrics['rougeL']:.4f}")
        else:
            logger.warning("⚠️ ROUGE-L: 유효한 점수가 없습니다.")
            metrics['rougeL'] = 0.0
            
    except ImportError:
        logger.error("❌ rouge-score를 설치하지 않았습니다.")
        logger.error("   설치: pip install rouge-score")
        metrics['rougeL'] = 0.0
    except Exception as e:
        logger.error(f"❌ ROUGE-L 계산 오류: {e}")
        metrics['rougeL'] = 0.0
    
    # 4. SPICE 계산 (pycocoevalcap 공식 레포지토리 사용 - Java 11+ 호환성)
    try:
        import os
        import subprocess
        from pycocoevalcap.spice.spice import Spice
        
        logger.info("📊 SPICE 계산 시작...")
        logger.info(f"   총 샘플 수: {len(predictions)}")
        
        # Java 버전 확인 및 적절한 옵션 설정
        java_version = 8  # 기본값
        try:
            java_version_output = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT, text=True)
            logger.info(f"   Java 버전: {java_version_output.split('\\n')[0].strip()}")
            
            # 버전 번호 추출 (예: "1.8.0" -> 8, "11.0.1" -> 11, "21.0.8" -> 21)
            import re
            version_match = re.search(r'version "(\d+)\.?(\d*)', java_version_output)
            if version_match:
                major_version = version_match.group(1)
                if major_version == "1":  # Java 8 형식: "1.8.0"
                    java_version = int(version_match.group(2))
                else:  # Java 9+ 형식: "11.0.1", "21.0.8"
                    java_version = int(major_version)
                logger.info(f"   감지된 Java 메이저 버전: {java_version}")
        except Exception as jv_e:
            logger.warning(f"   Java 버전 확인 실패: {jv_e}, Java 8로 가정합니다")
        
        # Java 버전별 옵션 설정
        if java_version >= 9:
            # Java 9+ Module System 호환성 설정
            # SPICE의 FST 직렬화 라이브러리가 리플렉션으로 java.base 패키지 접근 시 제한됨
            # _JAVA_OPTIONS 사용 (java -jar 명령에 전달됨, JAVA_TOOL_OPTIONS는 무시됨)
            logger.info(f"   Java {java_version} 감지 - Module System 호환성 옵션 적용")
            java_opts = (
                '-Xmx8G '
                '--add-opens=java.base/java.lang=ALL-UNNAMED '
                '--add-opens=java.base/java.util=ALL-UNNAMED '
                '--add-opens=java.base/java.io=ALL-UNNAMED '
                '--add-opens=java.base/java.lang.reflect=ALL-UNNAMED '
                '--add-opens=java.base/java.text=ALL-UNNAMED '
                '--add-opens=java.base/java.math=ALL-UNNAMED '
                '--add-opens=java.base/java.util.concurrent=ALL-UNNAMED '
                '--add-opens=java.base/java.net=ALL-UNNAMED '
                '--add-opens=java.desktop/java.awt.font=ALL-UNNAMED'
            ).strip()
            os.environ['_JAVA_OPTIONS'] = java_opts
        else:
            # Java 8 - --add-opens 옵션 불필요 (오히려 에러 발생)
            logger.info(f"   Java {java_version} 감지 - 기본 메모리 설정만 적용")
            os.environ['_JAVA_OPTIONS'] = '-Xmx8G'
        
        # JAVA_TOOL_OPTIONS 제거 (java -jar에서 충돌 방지)
        if 'JAVA_TOOL_OPTIONS' in os.environ:
            del os.environ['JAVA_TOOL_OPTIONS']
        
        logger.info(f"   _JAVA_OPTIONS 설정 완료 (Java 21 호환성)")
        
        spice_scorer = Spice()
        logger.info("   ✓ Spice scorer 초기화 완료")
        
        # 빈 문자열 필터링 (쌍을 유지하면서 필터링)
        valid_pairs_for_spice = [(pred, ref) for pred, ref in zip(predictions, references) 
                                  if pred.strip() and ref.strip()]
        
        if len(valid_pairs_for_spice) == 0:
            logger.warning("⚠️ SPICE: 유효한 텍스트 쌍이 없습니다.")
            metrics['spice'] = 0.0
        else:
            valid_preds_for_spice, valid_refs_for_spice = zip(*valid_pairs_for_spice)
            valid_preds_for_spice = list(valid_preds_for_spice)
            valid_refs_for_spice = list(valid_refs_for_spice)
            
            logger.info(f"   SPICE 계산: {len(valid_pairs_for_spice)}개 유효 샘플")
            
            # 텍스트 전처리: 토큰화 오류를 일으킬 수 있는 문자 정리
            def clean_for_spice(text, max_length=250):
                """SPICE 토큰화 오류 방지를 위한 추가 텍스트 정리
                
                Args:
                    text: 입력 텍스트 (이미 basic_cleanup 적용됨)
                    max_length: 최대 문자 길이 (SPICE 캐시 제한)
                
                Note:
                    <think> 태그는 이미 basic_cleanup에서 제거되었음
                """
                if not text or not isinstance(text, str):
                    return ""
                
                # 제어 문자 제거 (SPICE 토큰화 오류 방지)
                text = ''.join(char for char in text if char.isprintable() or char.isspace())
                
                # 연속된 공백 정리
                text = ' '.join(text.split())
                
                # 길이 제한 (단어 단위로 자르기)
                if len(text) > max_length:
                    words = text.split()
                    truncated = []
                    current_length = 0
                    for word in words:
                        if current_length + len(word) + 1 > max_length:
                            break
                        truncated.append(word)
                        current_length += len(word) + 1
                    text = ' '.join(truncated)
                    if text:  # 마침표 추가
                        text = text.rstrip('.,!?;:') + '.'
                
                # 특수 유니코드 문자를 ASCII로 근사 (SPICE는 ASCII만 처리)
                text = text.encode('ascii', 'ignore').decode('ascii')
                
                return text.strip()
            
            # 전처리 적용
            cleaned_preds = [clean_for_spice(pred) for pred in valid_preds_for_spice]
            cleaned_refs = [clean_for_spice (ref) for ref in valid_refs_for_spice]
            
            # 길이 통계
            truncated_preds = sum(1 for orig, clean in zip(valid_preds_for_spice, cleaned_preds) if len(orig) > len(clean))
            truncated_refs = sum(1 for orig, clean in zip(valid_refs_for_spice, cleaned_refs) if len(orig) > len(clean))
            if truncated_preds > 0 or truncated_refs > 0:
                logger.info(f"   📏 길이 제한으로 잘린 텍스트: Predictions={truncated_preds}, References={truncated_refs}")
            
            # 전처리 후 빈 문자열 필터링
            final_pairs = [(pred, ref, i) for i, (pred, ref) in enumerate(zip(cleaned_preds, cleaned_refs))
                          if pred and ref and len(pred.split()) > 0 and len(ref.split()) > 0]
            
            if len(final_pairs) == 0:
                logger.warning("⚠️ SPICE: 전처리 후 유효한 텍스트 쌍이 없습니다.")
                metrics['spice'] = 0.0
            else:
                filtered_preds, filtered_refs, indices = zip(*final_pairs)
                filtered_preds = list(filtered_preds)
                filtered_refs = list(filtered_refs)
                
                skipped_count = len(valid_pairs_for_spice) - len(final_pairs)
                if skipped_count > 0:
                    logger.warning(f"   ⚠️ 토큰화 불가능한 {skipped_count}개 샘플 제외됨")
                
                logger.info(f"   최종 SPICE 계산: {len(final_pairs)}개 샘플")
                
                gts = {str(i): [ref] for i, ref in enumerate(filtered_refs)}
                res = {str(i): [pred] for i, pred in enumerate(filtered_preds)}
            
            logger.info("   compute_score 호출 중... (시간이 걸릴 수 있습니다)")
            
            try:
                # 직접 compute_score 호출 (pycocoevalcap 공식 인터페이스)
                spice_score, spice_scores = spice_scorer.compute_score(gts, res)
                metrics['spice'] = float(spice_score)
                logger.info(f"✓ SPICE (공식 pycocoevalcap): {metrics['spice']:.4f}")
                if skipped_count > 0:
                    logger.info(f"  (참고: {skipped_count}개 샘플 제외하고 계산됨)")
            except subprocess.CalledProcessError as spice_e:
                # Java 프로세스 실행 실패
                logger.error(f"❌ SPICE Java 프로세스 실패 (exit code: {spice_e.returncode})")
                logger.error(f"   명령어: {' '.join(spice_e.cmd[:5])}...")
                
                # SPICE 임시 디렉토리에서 로그 파일 확인 시도
                import glob
                spice_pkg_path = os.path.dirname(os.path.abspath(Spice.__init__.__globals__['__file__']))
                tmp_dir = os.path.join(spice_pkg_path, 'tmp')
                
                if os.path.exists(tmp_dir):
                    # 최근 생성된 파일들 확인
                    recent_files = sorted(glob.glob(os.path.join(tmp_dir, '*')), 
                                        key=os.path.getmtime, reverse=True)[:5]
                    if recent_files:
                        logger.info(f"   SPICE 임시 파일 디렉토리: {tmp_dir}")
                        logger.info(f"   최근 파일: {[os.path.basename(f) for f in recent_files]}")
                        
                        # JSON 입력 파일 내용 확인 (첫 몇 줄)
                        for f in recent_files:
                            if os.path.isfile(f) and os.path.getsize(f) < 10000:  # 10KB 이하만
                                try:
                                    with open(f, 'r', encoding='utf-8', errors='ignore') as tmp_f:
                                        content = tmp_f.read(500)
                                        if content:
                                            logger.debug(f"   파일 {os.path.basename(f)} 내용 (일부):")
                                            logger.debug(f"   {content[:200]}...")
                                except:
                                    pass
                
                # 토큰화 오류로 추정되는 경우 개별 재시도
                error_msg = str(spice_e)
                logger.warning("⚠️ Java 실행 오류 발생 - 개별 샘플 단위로 재시도합니다")
                logger.info("   대안: 개별 샘플 SPICE 계산 (느리지만 안정적)")
                
                # 개별 샘플 단위로 재시도
                individual_scores = []
                failed_samples = []
                
                # 작은 배치로 나누어 시도 (10개씩)
                batch_size = 10
                for batch_start in range(0, len(filtered_preds), batch_size):
                    batch_end = min(batch_start + batch_size, len(filtered_preds))
                    batch_preds = filtered_preds[batch_start:batch_end]
                    batch_refs = filtered_refs[batch_start:batch_end]
                    
                    try:
                        # 작은 배치로 시도
                        batch_gts = {str(i): [ref] for i, ref in enumerate(batch_refs)}
                        batch_res = {str(i): [pred] for i, pred in enumerate(batch_preds)}
                        batch_score, batch_scores = spice_scorer.compute_score(batch_gts, batch_res)
                        individual_scores.extend(batch_scores)
                        logger.debug(f"   배치 {batch_start}-{batch_end} 성공: 평균 {batch_score:.4f}")
                    except Exception as batch_e:
                        # 배치도 실패하면 개별로
                        logger.debug(f"   배치 {batch_start}-{batch_end} 실패, 개별 시도...")
                        for i, (pred, ref) in enumerate(zip(batch_preds, batch_refs)):
                            abs_idx = batch_start + i
                            try:
                                mini_gts = {'0': [ref]}
                                mini_res = {'0': [pred]}
                                mini_score, _ = spice_scorer.compute_score(mini_gts, mini_res)
                                individual_scores.append(mini_score)
                            except Exception as sample_e:
                                failed_samples.append((abs_idx, pred[:50], ref[:50], str(sample_e)[:100]))
                
                if individual_scores:
                    metrics['spice'] = float(np.mean(individual_scores))
                    logger.info(f"✓ SPICE (개별/배치 계산): {metrics['spice']:.4f}")
                    logger.info(f"  성공: {len(individual_scores)}/{len(filtered_preds)} 샘플")
                    if failed_samples:
                        logger.warning(f"  실패한 샘플 {len(failed_samples)}개:")
                        for idx, pred_preview, ref_preview, error in failed_samples[:3]:
                            logger.warning(f"    [{idx}] Pred: {pred_preview}...")
                            logger.warning(f"         Ref: {ref_preview}...")
                            logger.warning(f"         Error: {error}")
                else:
                    logger.error("❌ 모든 샘플에서 SPICE 계산 실패")
                    metrics['spice'] = 0.0
                    
            except Exception as spice_e:
                # 기타 SPICE 계산 실패 시
                logger.error(f"❌ SPICE 계산 실패: {spice_e}")
                logger.error(f"   상세 오류: {traceback.format_exc()}")
                
                # 에러 로그에서 문제가 되는 텍스트 정보 추출 시도
                error_msg = str(spice_e)
                if "tokenize" in error_msg.lower() or "parse" in error_msg.lower():
                    logger.warning("⚠️ 토큰화 오류 발생 - 일부 샘플에 문제가 있을 수 있습니다")
                    logger.info("   대안: 개별 샘플 단위로 SPICE 계산 시도 중...")
                    
                    # 개별 샘플 단위로 재시도
                    individual_scores = []
                    failed_samples = []
                    
                    for i, (pred, ref) in enumerate(zip(filtered_preds, filtered_refs)):
                        try:
                            mini_gts = {'0': [ref]}
                            mini_res = {'0': [pred]}
                            mini_score, _ = spice_scorer.compute_score(mini_gts, mini_res)
                            individual_scores.append(mini_score)
                        except Exception as sample_e:
                            failed_samples.append((i, pred[:50], ref[:50]))
                            logger.debug(f"   샘플 {i} 실패: {str(sample_e)[:100]}")
                    
                    if individual_scores:
                        metrics['spice'] = float(np.mean(individual_scores))
                        logger.info(f"✓ SPICE (개별 계산): {metrics['spice']:.4f}")
                        logger.info(f"  성공: {len(individual_scores)}/{len(filtered_preds)} 샘플")
                        if failed_samples:
                            logger.warning(f"  실패한 샘플 {len(failed_samples)}개:")
                            for idx, pred_preview, ref_preview in failed_samples[:5]:
                                logger.warning(f"    [{idx}] Pred: {pred_preview}... / Ref: {ref_preview}...")
                    else:
                        logger.error("❌ 모든 샘플에서 SPICE 계산 실패")
                        metrics['spice'] = 0.0
                else:
                    logger.warning("⚠️ Java Module System 호환성 문제일 수 있습니다")
                    logger.info("해결 방법:")
                    logger.info("  Option 1: Java 8 설치 (권장)")
                    logger.info("    sudo apt-get install openjdk-8-jre")
                    logger.info("    export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64")
                    metrics['spice'] = 0.0
            
    except ImportError as ie:
        logger.error(f"❌ pycocoevalcap 임포트 실패: {ie}")
        logger.error("   설치: pip install git+https://github.com/salaniz/pycocoevalcap.git")
        metrics['spice'] = 0.0
    except Exception as e:
        logger.error(f"❌ SPICE 계산 오류: {e}")
        logger.error(f"   상세: {traceback.format_exc()}")
        metrics['spice'] = 0.0
    
    # 5. CIDEr 계산 (pycocoevalcap 공식 레포지토리 사용)
    try:
        from pycocoevalcap.cider.cider import Cider
        
        logger.info("📊 CIDEr 계산 중...")
        cider_scorer = Cider()
        
        # 빈 문자열 필터링 (쌍을 유지하면서 필터링)
        valid_pairs_for_cider = [(pred, ref) for pred, ref in zip(predictions, references) 
                                  if pred.strip() and ref.strip()]
        
        if len(valid_pairs_for_cider) == 0:
            logger.warning("⚠️ CIDEr: 유효한 텍스트 쌍이 없습니다.")
            metrics['cider'] = 0.0
        else:
            valid_preds_for_cider, valid_refs_for_cider = zip(*valid_pairs_for_cider)
            valid_preds_for_cider = list(valid_preds_for_cider)
            valid_refs_for_cider = list(valid_refs_for_cider)
            
            logger.info("  CIDEr 계산: {}개 유효 샘플".format(len(valid_pairs_for_cider)))
            
            gts = {str(i): [ref] for i, ref in enumerate(valid_refs_for_cider)}
            res = {str(i): [pred] for i, pred in enumerate(valid_preds_for_cider)}
            
            try:
                # 직접 compute_score 호출 (pycocoevalcap 공식 인터페이스)
                cider_score, cider_scores = cider_scorer.compute_score(gts, res)
                metrics['cider'] = float(cider_score)
                logger.info(f"✓ CIDEr (공식 pycocoevalcap): {metrics['cider']:.4f}")
            except Exception as cider_e:
                logger.error(f"❌ CIDEr compute_score 오류: {cider_e}")
                metrics['cider'] = 0.0
                
    except ImportError:
        logger.error("❌ pycocoevalcap을 설치하지 않았습니다.")
        logger.error("   설치: pip install git+https://github.com/salaniz/pycocoevalcap.git")
        metrics['cider'] = 0.0
    except Exception as e:
        logger.error(f"❌ CIDEr 계산 오류: {e}")
        metrics['cider'] = 0.0
    
    # 메트릭 저장
    safe_prefix = prefix if prefix else "model"
    metrics_path = output_dir / f"{safe_prefix}_metrics_{timestamp}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 메트릭 저장: {metrics_path}")
    return metrics


def print_final_results(metrics: Dict[str, float]):
    """
    최종 결과 출력 (모든 메트릭 포함)
    """
    print("\n" + "=" * 90)
    print("🎉 PanoLLaVA 모델 평가 완료 - 모든 메트릭 계산 성공")
    print("=" * 90)
    
    print("\n📊 평가 메트릭 결과 (공식 레포지토리 기반):")
    print("-" * 90)
    
    metric_info = {
        'bleu4': ('BLEU-4', 'sacrebleu (https://github.com/mjpost/sacrebleu)'),
        'meteor': ('METEOR', 'NLTK (https://www.nltk.org/)'),
        'rougeL': ('ROUGE-L', 'rouge-score (https://github.com/google-research/rouge)'),
        'spice': ('SPICE', 'pycocoevalcap (https://github.com/salaniz/pycocoevalcap)'),
        'cider': ('CIDEr', 'pycocoevalcap (https://github.com/salaniz/pycocoevalcap)'),
    }
    
    for key, (display_name, source) in metric_info.items():
        if key in metrics:
            value = metrics[key]
            status = "✓" if value > 0 else "✗"
            print(f"{status} {display_name:12s} (↑): {value:8.4f}  | 출처: {source}")
    
    print("-" * 90)
    print("💡 (↑) 표시는 높을수록 좋은 메트릭입니다.")
    print("\n📌 메트릭 설명:")
    print("  • BLEU-4   : 기계 번역 품질 평가 (n-gram 정확도)")
    print("  • METEOR   : 의미론적 유사도 고려 (동의어, 어근 일치)")
    print("  • ROUGE-L  : 재현율 중심 평가 (최대 공통 부분수열)")
    print("  • SPICE    : 의미적 명제 기반 평가 (그래프 구조)")
    print("  • CIDEr    : 이미지 캡션 평가 (용어 신뢰도 기반)")
    print("=" * 90)




def load_global_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load evaluation configuration using the shared training loader."""

    try:
        return _load_train_config_dict(config_path)
    except Exception as exc:
        logger.error(f"Failed to load configuration: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description="PanoLLaVA 모델 평가 시스템")
    # 입력 인자: --config, --csv-input, --checkpoint-dir, --checkpoint
    parser.add_argument('--config', help='Global config YAML 경로 (미지정 시 PANOVLM_CONFIG or ./config.yaml 사용)')
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', default=None,
                        help='체크포인트 디렉토리 경로 (예: runs/ADDDATA_SQ3_1/finetune/anyres-e2p_mlp/). '
                             'checkpoint_metadata.json이 있으면 자동으로 설정을 로드합니다. '
                             'best.ckpt 또는 last.ckpt 심볼릭 링크를 우선 사용합니다.')
    parser.add_argument('--checkpoint', '--ckpt', dest='checkpoint_file', default=None,
                        help='체크포인트 파일 경로를 명시적으로 지정 (예: runs/.../best.ckpt). '
                             '--checkpoint-dir보다 우선 적용됩니다.')
    parser.add_argument('--csv-input', dest='csv_input', default=None,
                        help='평가에 사용할 CSV 경로 (예: data/quic360/test.csv). '
                             'prediction과 reference 컬럼이 있으면 바로 메트릭 계산, 없으면 모델로 생성합니다.')
    parser.add_argument('--metrics-only', action='store_true',
                        help='CSV에 prediction/reference가 있을 때 메트릭만 계산 (모델 로딩 생략)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='평가에 사용할 최대 샘플 수 (None이면 전체 데이터 사용)')
    parser.add_argument('--log-samples', action='store_true',
                        help='배치별 상세 예측/정답 로그를 활성화합니다.')
    parser.add_argument('--log-interval', type=int, default=25,
                        help='--log-samples 사용 시 배치 로그 간격 (기본 25)')
    parser.add_argument('--log-max-samples', type=int, default=50,
                        help='--log-samples 사용 시 최대 로그 샘플 수 (기본 50)')

    args = parser.parse_args()

    # ========== CSV 파일 사전 검사: prediction/reference 존재 여부 확인 ==========
    # 메트릭 전용 모드 판별을 위해 가장 먼저 실행
    preliminary_csv_input = args.csv_input
    metrics_only_mode = False
    
    if preliminary_csv_input:
        csv_path = Path(preliminary_csv_input)
        if csv_path.exists() and csv_path.suffix.lower() == '.csv':
            try:
                # CSV 컬럼 확인
                df_check = pd.read_csv(csv_path, nrows=5)  # 상위 5개 행만 읽어서 확인
                has_prediction = 'prediction' in df_check.columns
                has_reference = 'reference' in df_check.columns
                
                if has_prediction and has_reference:
                    metrics_only_mode = True
                    logger.info("=" * 60)
                    logger.info("🔍 CSV 파일에 prediction/reference 컬럼 발견!")
                    logger.info("📊 메트릭 전용 모드 활성화 (모델 로딩 및 Config 로딩 생략)")
                    logger.info("=" * 60)
                elif args.metrics_only:
                    logger.warning("⚠️ --metrics-only 옵션이 지정되었으나 CSV에 필수 컬럼이 없습니다.")
                    logger.warning(f"   현재 컬럼: {df_check.columns.tolist()}")
                    logger.warning("   필수 컬럼: ['prediction', 'reference']")
                    raise ValueError("메트릭 전용 모드를 사용하려면 CSV에 prediction과 reference 컬럼이 필요합니다.")
            except pd.errors.EmptyDataError:
                logger.warning(f"⚠️ CSV 파일이 비어있습니다: {csv_path}")
            except Exception as e:
                logger.warning(f"⚠️ CSV 사전 검사 실패 (일반 모드로 진행): {e}")
    
    if args.metrics_only and not metrics_only_mode:
        raise ValueError("--metrics-only 옵션은 CSV에 prediction과 reference 컬럼이 있을 때만 사용 가능합니다.")

    # ========== 메트릭 전용 모드: 바로 메트릭 계산으로 이동 ==========
    if metrics_only_mode:
        logger.info("📊 메트릭 전용 모드 - Config 및 모델 로딩 완전 생략")
        
        # 필요한 최소 변수만 설정
        max_samples_cli = args.max_samples if args.max_samples and args.max_samples > 0 else None
        safe_prefix = csv_path.stem  # CSV 파일명을 prefix로 사용
        output_dir = Path("results/eval_results") / safe_prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        try:
            logger.info("=" * 60)
            logger.info("📊 메트릭 계산 모드")
            logger.info("=" * 60)
            logger.info(f"📂 CSV 입력: {csv_path}")
            
            # CSV 전체 로드
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"✓ 데이터 로드 완료: {len(df)}개 샘플")
            
            # max_samples 적용
            if max_samples_cli is not None and len(df) > max_samples_cli:
                logger.info(f"📉 샘플 수 제한: {len(df)} → {max_samples_cli}")
                df = df.head(max_samples_cli)
            
            # 5단계: 평가 메트릭 계산 (CSV DataFrame 직접 전달)
            metrics = calculate_evaluation_metrics(df, output_dir, timestamp, safe_prefix)
            
            # 최종 결과 출력
            print_final_results(metrics)
            
            return  # 메트릭 계산 후 종료
            
        except Exception as e:
            logger.error(f"❌ 메트릭 계산 중 오류 발생: {e}")
            logger.error(f"상세 오류: {traceback.format_exc()}")
            raise

    # ========== 일반 모드: Config 로딩 필요 ==========
    # 체크포인트 파일/디렉토리 우선 처리
    checkpoint_metadata = None
    explicit_checkpoint_path = None
    
    # --checkpoint 옵션이 주어진 경우 우선 사용
    if args.checkpoint_file:
        ckpt_file = Path(args.checkpoint_file)
        
        if not ckpt_file.exists():
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {ckpt_file}")
        
        if not ckpt_file.is_file():
            raise ValueError(f"체크포인트 경로가 파일이 아닙니다: {ckpt_file}")
        
        logger.info("=" * 60)
        logger.info(f"📄 명시적 체크포인트 파일: {ckpt_file}")
        logger.info("=" * 60)
        
        explicit_checkpoint_path = ckpt_file
        
        # 체크포인트 파일의 부모 디렉토리에서 메타데이터 로드 시도
        ckpt_dir = ckpt_file.parent
        checkpoint_metadata = load_checkpoint_metadata(ckpt_dir)
        
        if checkpoint_metadata:
            logger.info("=" * 60)
            logger.info("📋 메타데이터에서 로드된 정보:")
            logger.info(f"  - Experiment: {checkpoint_metadata.get('experiment_name', 'N/A')}")
            logger.info(f"  - Stage: {checkpoint_metadata.get('stage', 'N/A')}")
            logger.info(f"  - Vision: {checkpoint_metadata.get('model_config', {}).get('vision_name', 'N/A')}")
            logger.info(f"  - Language: {checkpoint_metadata.get('model_config', {}).get('language_model_name', 'N/A')}")
            logger.info(f"  - Resampler: {checkpoint_metadata.get('model_config', {}).get('resampler_type', 'N/A')}")
            logger.info(f"  - Crop Strategy: {checkpoint_metadata.get('training_config', {}).get('crop_strategy', 'N/A')}")
            logger.info("=" * 60)
    
    elif args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"체크포인트 디렉토리를 찾을 수 없습니다: {ckpt_dir}")
        
        logger.info("=" * 60)
        node_desc = "디렉토리" if ckpt_dir.is_dir() else "파일"
        logger.info(f"📂 체크포인트 {node_desc}: {ckpt_dir}")
        logger.info("=" * 60)
        
        # 메타데이터 로드 시도
        checkpoint_metadata = load_checkpoint_metadata(ckpt_dir)
        
        # 체크포인트 파일 찾기
        explicit_checkpoint_path = find_checkpoint_in_dir(ckpt_dir)
        
        if not explicit_checkpoint_path:
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {ckpt_dir}")
        
        logger.info(f"✅ 체크포인트 파일: {explicit_checkpoint_path}")
        
        if checkpoint_metadata:
            logger.info("=" * 60)
            logger.info("📋 메타데이터에서 로드된 정보:")
            logger.info(f"  - Experiment: {checkpoint_metadata.get('experiment_name', 'N/A')}")
            logger.info(f"  - Stage: {checkpoint_metadata.get('stage', 'N/A')}")
            logger.info(f"  - Vision: {checkpoint_metadata.get('model_config', {}).get('vision_name', 'N/A')}")
            logger.info(f"  - Language: {checkpoint_metadata.get('model_config', {}).get('language_model_name', 'N/A')}")
            logger.info(f"  - Resampler: {checkpoint_metadata.get('model_config', {}).get('resampler_type', 'N/A')}")
            logger.info(f"  - Crop Strategy: {checkpoint_metadata.get('training_config', {}).get('crop_strategy', 'N/A')}")
            logger.info("=" * 60)

    # ========== CSV 파일 사전 검사: prediction/reference 존재 여부 확인 ==========
    # CLI에서 지정한 CSV 또는 기본값
    preliminary_csv_input = args.csv_input or "data/quic360/test.csv"
    csv_path = Path(preliminary_csv_input)
    metrics_only_mode = False
    
    if csv_path.exists() and csv_path.suffix.lower() == '.csv':
        try:
            # CSV 컬럼 확인
            df_check = pd.read_csv(csv_path, nrows=5)  # 상위 5개 행만 읽어서 확인
            has_prediction = 'prediction' in df_check.columns
            has_reference = 'reference' in df_check.columns
            
            if has_prediction and has_reference:
                metrics_only_mode = True
                logger.info("=" * 60)
                logger.info("🔍 CSV 파일에 prediction/reference 컬럼 발견!")
                logger.info("📊 메트릭 전용 모드 활성화 (모델 로딩 생략)")
                logger.info("=" * 60)
            elif args.metrics_only:
                logger.warning("⚠️ --metrics-only 옵션이 지정되었으나 CSV에 필수 컬럼이 없습니다.")
                logger.warning(f"   현재 컬럼: {df_check.columns.tolist()}")
                logger.warning("   필수 컬럼: ['prediction', 'reference']")
                raise ValueError("메트릭 전용 모드를 사용하려면 CSV에 prediction과 reference 컬럼이 필요합니다.")
        except pd.errors.EmptyDataError:
            logger.warning(f"⚠️ CSV 파일이 비어있습니다: {csv_path}")
        except Exception as e:
            logger.warning(f"⚠️ CSV 사전 검사 실패 (일반 모드로 진행): {e}")
    
    if args.metrics_only and not metrics_only_mode:
        raise ValueError("--metrics-only 옵션은 CSV에 prediction과 reference 컬럼이 있을 때만 사용 가능합니다.")

    # ========== 메트릭 전용 모드: Config 로딩 생략 ==========
    if metrics_only_mode:
        logger.info("📊 메트릭 전용 모드 - Config 로딩 생략")
        # 필요한 최소 변수만 설정
        max_samples_cli = args.max_samples if args.max_samples and args.max_samples > 0 else None
        safe_prefix = csv_path.stem  # CSV 파일명을 prefix로 사용
        output_dir = Path("results/eval_results") / safe_prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        try:
            # CSV 전체 로드
            logger.info("=" * 60)
            logger.info("📊 메트릭 계산 모드")
            logger.info("=" * 60)
            logger.info(f"📂 CSV 입력: {csv_path}")
            
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"✓ 데이터 로드 완료: {len(df)}개 샘플")
            
            # max_samples 적용
            if max_samples_cli is not None and len(df) > max_samples_cli:
                logger.info(f"📉 샘플 수 제한: {len(df)} → {max_samples_cli}")
                df = df.head(max_samples_cli)
            
            # 평가 메트릭 계산 (CSV DataFrame 직접 전달)
            metrics = calculate_evaluation_metrics(df, output_dir, timestamp, safe_prefix)
            
            # 최종 결과 출력
            print_final_results(metrics)
            return  # 여기서 종료
            
        except Exception as e:
            logger.error(f"❌ 메트릭 계산 중 오류 발생: {e}")
            logger.error(f"상세 오류: {traceback.format_exc()}")
            raise

    # ========== 일반 모드: Config 로딩 필요 ==========
    global_config = load_global_config(args.config)
    max_samples_cli = args.max_samples if args.max_samples and args.max_samples > 0 else None
    log_samples_flag = bool(args.log_samples)
    log_interval_cli = args.log_interval if args.log_interval and args.log_interval > 0 else 0
    log_max_samples_cli = args.log_max_samples if args.log_max_samples and args.log_max_samples > 0 else 50

    # ========== 메타데이터 우선 설정 병합 ==========
    # checkpoint_metadata가 있으면 우선 사용, 없으면 config 사용
    if checkpoint_metadata:
        model_config_meta = checkpoint_metadata.get('model_config', {})
        training_config_meta = checkpoint_metadata.get('training_config', {})
        dataset_meta = checkpoint_metadata.get('dataset', {})
        
        # 모델 설정 병합 (메타데이터 우선)
        global_config.setdefault('models', {})
        global_config['models']['vision_name'] = model_config_meta.get('vision_name', global_config.get('models', {}).get('vision_name'))
        global_config['models']['language_model_name'] = model_config_meta.get('language_model_name', global_config.get('models', {}).get('language_model_name'))
        global_config['models']['resampler_type'] = model_config_meta.get('resampler_type', global_config.get('models', {}).get('resampler_type'))
        global_config['models']['latent_dimension'] = model_config_meta.get('latent_dimension', global_config.get('models', {}).get('latent_dimension'))
        
        # 이미지 처리 설정 병합
        global_config.setdefault('image_processing', {})
        global_config['image_processing']['crop_strategy'] = training_config_meta.get('crop_strategy', global_config.get('image_processing', {}).get('crop_strategy'))
        global_config['image_processing']['image_size'] = model_config_meta.get('image_size', global_config.get('image_processing', {}).get('image_size'))
        global_config['image_processing']['fov_deg'] = training_config_meta.get('fov_deg', global_config.get('image_processing', {}).get('fov_deg'))
        global_config['image_processing']['overlap_ratio'] = training_config_meta.get('overlap_ratio', global_config.get('image_processing', {}).get('overlap_ratio'))
        global_config['image_processing']['use_vision_processor'] = training_config_meta.get('use_vision_processor', global_config.get('image_processing', {}).get('use_vision_processor'))
        global_config['image_processing']['normalize'] = training_config_meta.get('normalize', global_config.get('image_processing', {}).get('normalize'))
        
        # 훈련 설정 병합
        global_config.setdefault('training', {})
        global_config['training']['max_text_length'] = model_config_meta.get('max_text_length', global_config.get('training', {}).get('max_text_length'))
        
        logger.info("✅ 메타데이터를 config에 병합 완료")

    env_config = global_config.get("environment", {})
    model_config = global_config.get("models", {})
    data_config = global_config.get("data", {})
    training_config = global_config.get("training", {})
    image_cfg = global_config.get("image_processing", {})
    system_msgs = global_config.get("system_messages", {})

    # 디바이스 설정: 환경변수 대신 config 기반으로 GPU index를 선택
    cuda_vis = env_config.get("cuda_visible_devices")
    if torch.cuda.is_available():
        try:
            first_idx = int(str(cuda_vis).split(",")[0].strip())
            torch.cuda.set_device(first_idx)
            logger.info(f"Device: using GPU index {first_idx} (from config)")
        except Exception as e:
            logger.warning(f"Invalid cuda_visible_devices in config: {cuda_vis} ({e})")

    # CSV 입력 경로: CLI 우선 -> config 우선순위 -> 기본값
    eff_csv_input = (
        args.csv_input
        or data_config.get("csv_test")
        or data_config.get("csv_val")
        or global_config.get("paths", {}).get("csv_val")
        or "data/quic360/test.csv"
    )


    # Model core
    eff_vision_name = model_config.get("vision_name")
    eff_lm_name = model_config.get("language_model_name") or model_config.get("lm_model")
    eff_resampler = model_config.get("resampler_type") or model_config.get("resampler")
    # Image processing
    eff_crop_strategy = image_cfg.get("crop_strategy", "e2p")
    eff_overlap_ratio = image_cfg.get("overlap_ratio", 0.5)
    eff_use_vp = image_cfg.get("use_vision_processor", True)
    eff_image_size = image_cfg.get("image_size", [224, 224])
    eff_fov_deg = image_cfg.get("fov_deg", 90.0)
    eff_image_mean = image_cfg.get("image_mean")
    eff_image_std = image_cfg.get("image_std")
    eff_anyres_patch_size = image_cfg.get("anyres_patch_size")  # None이면 image_size에서 자동 추론
    eff_anyres_max_patches = image_cfg.get("anyres_max_patches", 12)
    eff_normalize = image_cfg.get("normalize", True)
    # Tokenization
    eff_max_text_length = str(training_config.get("max_text_length", data_config.get("max_text_length", "auto")))
    eff_num_workers = training_config.get("num_workers", 16)
    eff_batch_size = (
        training_config.get("eval_batch_size")
        or training_config.get("batch_size")
        or training_config.get("finetune", {}).get("batch_size")
        or 16
    )
    eff_system_msg = training_config.get("system_msg", system_msgs.get("default", "You are a helpful assistant."))
    eff_output_dir = global_config.get("paths", {}).get("eval_dir", "results/eval_results")
    eff_prefix = training_config.get("prefix") or "model"
    safe_prefix = str(eff_prefix).strip() or "model"
    for ch in ["/", "\\", " "]:
        safe_prefix = safe_prefix.replace(ch, "_")
    # Generation
    gen_cfg = global_config.get("generation", {}) if isinstance(global_config, dict) else {}
    def _g(key, default):
        return gen_cfg.get(key, default) if isinstance(gen_cfg, dict) else default
    eff_gen_max_new_tokens = _g('max_new_tokens', 128)
    eff_gen_temperature = _g('temperature', 0.6)
    eff_gen_min_new_tokens = _g('min_new_tokens', 5)
    eff_gen_top_p = _g('top_p', 0.95)
    eff_gen_top_k = _g('top_k', 20)
    eff_gen_repetition_penalty = _g('repetition_penalty', 1.1)
    eff_gen_length_penalty = _g('length_penalty', 1.0)

    # ========== 모델 디렉토리/체크포인트 해결 ==========
    # --checkpoint-dir이 지정되면 우선 사용, 아니면 자동 탐색
    if explicit_checkpoint_path:
        model_dir = str(explicit_checkpoint_path)
        logger.info(f"✅ 명시적 체크포인트 사용: {model_dir}")
    else:
        # stage strictly from config
        stage_from_cfg = training_config.get('default_stage', 'finetune')
        # 모델 디렉토리 자동 해결
        cfg_source = args.config if args.config else global_config
        model_dir = resolve_model_dir(cfg_source, stage_from_cfg, crop_strategy=eff_crop_strategy)

    # LoRA 가중치 자동 설정 (config-only; no CLI override)
    lora_weights_path = None
    if model_dir:
        checkpoint_dir = Path(model_dir)
        checkpoint_dir = checkpoint_dir if checkpoint_dir.is_dir() else checkpoint_dir.parent
        potential_lora_path = checkpoint_dir / "lora_weights"
        if potential_lora_path.exists():
            lora_weights_path = str(potential_lora_path)
            logger.info(f"✅ Auto-found LoRA weights: {lora_weights_path}")

    # 출력 디렉토리
    output_dir = Path(eff_output_dir) / safe_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  사용 디바이스: {device}")

    try:
        # ========== 일반 모드: 모델 로딩 + 생성 + 메트릭 계산 ==========
        # 1단계: 모델 및 LoRA 가중치 로드
        # Convert max_text_length for model only if numeric; otherwise omit (DataModule handles "auto")
        _mtl_val = None
        try:
            _mtl_val = int(eff_max_text_length)
        except Exception:
            _mtl_val = None
        # Use canonical config keys to avoid mismatches; include only when provided
        model_kwargs = {}
        if eff_vision_name:
            model_kwargs["vision_name"] = eff_vision_name
        if eff_lm_name:
            model_kwargs["language_model_name"] = eff_lm_name
        if eff_resampler:
            model_kwargs["resampler_type"] = eff_resampler
        if _mtl_val is not None:
            model_kwargs["max_text_length"] = _mtl_val
        model = load_model_and_lora(
            model_dir,
            lora_weights_path,
            device,
            config_path=args.config,  # ModelConfig를 별도로 쓰는 경우
            config_data=global_config if isinstance(global_config, dict) else None,
            **model_kwargs
        )

        # 2단계: 테스트 데이터셋 준비 (config 반영 인자 추가)
        datamodule, test_dataloader = prepare_test_dataset(
            csv_input=eff_csv_input,
            batch_size=eff_batch_size,
            max_text_length=eff_max_text_length,
            crop_strategy=(eff_crop_strategy or "e2p"),
            lm_name=(eff_lm_name or "Qwen/Qwen2.5-0.5B-Instruct"),
            num_workers=eff_num_workers,
            overlap_ratio=(eff_overlap_ratio if eff_overlap_ratio is not None else 0.5),
            image_size=eff_image_size,
            fov_deg=eff_fov_deg,
            image_mean=eff_image_mean,
            image_std=eff_image_std,
            anyres_patch_size=eff_anyres_patch_size,
            anyres_max_patches=eff_anyres_max_patches,
            normalize=eff_normalize,
            vision_name=eff_vision_name,
            system_msg=eff_system_msg,
            use_vision_processor=(bool(eff_use_vp) if eff_use_vp is not None else False),
            auto_max_text_length_cap=int(global_config.get("data", {}).get("auto_max_text_length_cap", 8192)) if isinstance(global_config, dict) else 8192,
            auto_max_text_length_floor=int(global_config.get("data", {}).get("auto_max_text_length_floor", 512)) if isinstance(global_config, dict) else None,
            auto_max_text_length_scan_limit=int(global_config.get("data", {}).get("auto_max_text_length_scan_limit", 1000)) if isinstance(global_config, dict) else None
        )

        # 3단계: 텍스트 생성 (system_msg 전달)
        predictions, references, image_paths, input_texts = generate_predictions(
            model, test_dataloader, datamodule, device,
            max_new_tokens=int(eff_gen_max_new_tokens),
            temperature=float(eff_gen_temperature),
            top_p=float(eff_gen_top_p),
            top_k=int(eff_gen_top_k),
            repetition_penalty=float(eff_gen_repetition_penalty),
            length_penalty=float(eff_gen_length_penalty),
            min_new_tokens=int(eff_gen_min_new_tokens),
            system_msg=eff_system_msg,
            max_samples=max_samples_cli,
            log_samples=log_samples_flag,
            log_interval=log_interval_cli,
            log_max_samples=log_max_samples_cli
        )

        # 4단계: 결과 저장 및 로깅
        df = save_and_log_results(
            predictions,
            references,
            image_paths,
            input_texts,
            output_dir,
            timestamp,
            safe_prefix,
        )

        # 5단계: 평가 메트릭 계산
        if max_samples_cli is not None:
            logger.info(f"⚠️ 제한된 {len(df)}개 샘플에 대해서만 메트릭을 계산합니다.")
        metrics = calculate_evaluation_metrics(df, output_dir, timestamp, safe_prefix)

        # 최종 결과 출력
        print_final_results(metrics)

    except Exception as e:
        logger.error(f"❌ 평가 중 오류 발생: {e}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        raise

if __name__ == '__main__':
    main()
