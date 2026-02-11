#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract Metadata from Checkpoint
================================

체크포인트 파일(.ckpt)에서 `hyper_parameters`를 추출하여 
같은 디렉토리에 `checkpoint_metadata.json`으로 저장합니다.

사용법:
    python scripts/extract_metadata.py --checkpoint runs/.../last.ckpt
"""

import argparse
import torch
import json
import os
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """JSON 직렬화를 위해 타입을 변환합니다."""
    if isinstance(obj, (torch.Tensor,)):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, (set, tuple)):
        return list(obj)
    elif isinstance(obj, Path):
        return str(obj)
    return obj

def extract_metadata(ckpt_path: Path):
    if not ckpt_path.exists():
        logger.error(f"❌ 체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
        return

    logger.info(f"📂 Loading checkpoint: {ckpt_path}")
    try:
        # CPU로 로드하여 메모리 절약
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        if 'hyper_parameters' not in ckpt:
            logger.warning(f"⚠️ 'hyper_parameters' key not found in {ckpt_path.name}")
            return

        hparams = ckpt['hyper_parameters']
        logger.info(f"✅ Extracted {len(hparams)} hyperparameters")

        # 직렬화 가능한 형태로 변환
        serializable_hparams = {k: convert_to_serializable(v) for k, v in hparams.items()}

        # 구조화된 메타데이터 생성
        # eval.py가 기대하는 구조에 맞춰서 매핑
        metadata = {
            "source_checkpoint": str(ckpt_path.name),
            "experiment_name": serializable_hparams.get("experiment_name", "unknown"),
            "stage": serializable_hparams.get("stage", "unknown"),
            "model_config": {
                "vision_name": serializable_hparams.get("vision_name"),
                "language_model_name": serializable_hparams.get("language_model_name"),
                "resampler_type": serializable_hparams.get("resampler_type"),
                "latent_dimension": serializable_hparams.get("latent_dimension"),
                "image_size": serializable_hparams.get("image_size"),
            },
            "training_config": {
                "crop_strategy": serializable_hparams.get("crop_strategy"),
                "fov_deg": serializable_hparams.get("fov_deg"),
                "overlap_ratio": serializable_hparams.get("model_overlap_ratio"), # 이름 차이 주의
                "use_vision_processor": serializable_hparams.get("use_vision_processor", True),
                "normalize": serializable_hparams.get("normalize", True),
            },
            # 원본 데이터도 포함 (디버깅용)
            "raw_hyper_parameters": serializable_hparams
        }
        
        # 일부 필수 필드 기본값 채우기 (None인 경우)
        if metadata["training_config"]["overlap_ratio"] is None:
             metadata["training_config"]["overlap_ratio"] = serializable_hparams.get("overlap_ratio", 0.5)
        
        # FOV 및 Crop Strategy 기본값 안전장치
        logger.info(f"   [DEBUG] Before defaults - fov_deg: {metadata['training_config']['fov_deg']}, crop_strategy: {metadata['training_config']['crop_strategy']}")
        
        if metadata["training_config"]["fov_deg"] is None:
             metadata["training_config"]["fov_deg"] = 90.0
             logger.info("   [DEBUG] Applied default fov_deg=90.0")
             
        if metadata["training_config"]["crop_strategy"] is None:
             metadata["training_config"]["crop_strategy"] = "e2p"
             logger.info("   [DEBUG] Applied default crop_strategy='e2p'")

        # 저장 경로: 체크포인트와 같은 디렉토리
        output_path = ckpt_path.parent / "checkpoint_metadata.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"💾 Saved metadata to: {output_path}")
        logger.info(f"   - Resampler: {metadata['model_config']['resampler_type']}")
        logger.info(f"   - Latent Dim: {metadata['model_config']['latent_dimension']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to extract metadata: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract hyperparameters from checkpoint")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .ckpt file')
    args = parser.parse_args()

    extract_metadata(Path(args.checkpoint))

if __name__ == "__main__":
    main()
