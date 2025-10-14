#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM 모델 평가 스크립트 (학습 없이)

다양한 HuggingFace VLM 모델들을 동일한 데이터셋으로 평가합니다.
평가 지표: BLEU-4, METEOR, ROUGE-L, Exact Match 등
이미지 크기: 224x224로 고정
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

# Monkey-patch for LLaVA-OneVision flash attention compatibility
# LLaVA-OneVision models expect flash_attn_varlen_func in transformers.modeling_flash_attention_utils
# but newer transformers versions removed it. We provide a compatibility layer.
import sys
import importlib

# Step 1: Patch transformers.modeling_flash_attention_utils before any model imports
try:
    import transformers.modeling_flash_attention_utils as flash_utils

    if not hasattr(flash_utils, 'flash_attn_varlen_func'):
        # Try importing from flash_attn package
        try:
            from flash_attn import flash_attn_varlen_func
            flash_utils.flash_attn_varlen_func = flash_attn_varlen_func
            print("✓ Patched flash_attn_varlen_func from flash_attn package")
        except ImportError:
            # Create a dummy function that will work with eager attention
            def flash_attn_varlen_func(*args, **kwargs):
                raise NotImplementedError(
                    "FlashAttention is not available. Using eager attention instead."
                )
            flash_utils.flash_attn_varlen_func = flash_attn_varlen_func
            print("⚠️ Created dummy flash_attn_varlen_func - models will use eager attention")

    # Also patch _flash_attention_forward if needed
    if not hasattr(flash_utils, '_flash_attention_forward'):
        def _flash_attention_forward(*args, **kwargs):
            raise NotImplementedError(
                "FlashAttention is not available. Using eager attention instead."
            )
        flash_utils._flash_attention_forward = _flash_attention_forward

except Exception as e:
    print(f"⚠️ Warning: Could not patch flash attention utilities: {e}")
    print("   Models will attempt to use eager attention")

# Allow large images
Image.MAX_IMAGE_PIXELS = None

# Add project root to Python path to import eval.py utilities
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import shared evaluation metrics from eval.py
try:
    from scripts.eval import calculate_evaluation_metrics as eval_calculate_metrics
    USE_EVAL_METRICS = True
    logging.info("✓ Using shared evaluation metrics from scripts/eval.py")
except ImportError:
    USE_EVAL_METRICS = False
    logging.warning("⚠️ Could not import eval.py metrics, falling back to local implementation")

# ─────────────────────────────────────────────────────────────
# InternVL3.5 이미지 전처리 헬퍼 함수
# ─────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform_internvl(input_size):
    """InternVL용 이미지 변환"""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """InternVL용 최적 aspect ratio 찾기"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess_internvl(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """InternVL용 dynamic preprocessing"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_internvl(image, input_size=448, max_num=12):
    """InternVL용 이미지 로드 및 전처리"""
    transform = build_transform_internvl(input_size=input_size)
    images = dynamic_preprocess_internvl(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ─────────────────────────────────────────────────────────────
# VLM 모델 정의
# ─────────────────────────────────────────────────────────────
# Padding side는 _get_padding_side() 메서드에서 자동 설정:
#   - Decoder-only 모델: left padding (LLaVA, Qwen2.5-VL, BLIP2-OPT, InstructBLIP-Vicuna, PaliGemma, Gemma-3, InternVL)
#   - Encoder-decoder 모델: right padding (Florence-2, BLIP2-T5, InstructBLIP-T5)

VLM_MODELS = {
    "llava-1.5-7b": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "processor_id": "llava-hf/llava-1.5-7b-hf",
        "model_class": "LlavaForConditionalGeneration",
        "processor_class": "LlavaProcessor",
        "prompt_template": "USER: <image>\n{instruction}\nASSISTANT:",
    },
    "llava-1.6-mistral-7b": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "processor_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "model_class": "LlavaNextForConditionalGeneration",
        "processor_class": "LlavaNextProcessor",
        "prompt_template": "[INST] <image>\n{instruction} [/INST]",
    },
    "llava-onevision-0.5b": {
        "model_id": "lmms-lab/llava-onevision-qwen2-0.5b-ov",
        "processor_id": "lmms-lab/llava-onevision-qwen2-0.5b-ov",
        "model_class": "LlavaOnevisionForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "use_chat_template": True,
        "requires_vision_utils": False,  # LLaVA-OneVision uses its own processing
    },
    "llava-onevision-4b": {
        "model_id": "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        "processor_id": "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        "model_class": "LlavaOnevisionForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "use_chat_template": True,
        "requires_vision_utils": False,  # LLaVA-OneVision uses its own processing
    },
    "llava-onevision-7b": {
        "model_id": "lmms-lab/llava-onevision-qwen2-7b-ov",
        "processor_id": "lmms-lab/llava-onevision-qwen2-7b-ov",
        "model_class": "LlavaOnevisionForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "use_chat_template": True,
        "requires_vision_utils": False,  # LLaVA-OneVision uses its own processing
    },
    "blip2-opt-2.7b": {
        "model_id": "Salesforce/blip2-opt-2.7b",
        "processor_id": "Salesforce/blip2-opt-2.7b",
        "model_class": "Blip2ForConditionalGeneration",
        "processor_class": "Blip2Processor",
        "prompt_template": "Question: {instruction} Answer:",
    },
    "instructblip-vicuna-7b": {
        "model_id": "Salesforce/instructblip-vicuna-7b",
        "processor_id": "Salesforce/instructblip-vicuna-7b",
        "model_class": "InstructBlipForConditionalGeneration",
        "processor_class": "InstructBlipProcessor",
        "prompt_template": "{instruction}",
    },
    "qwen2.5-vl-3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "processor_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "use_chat_template": True,  # Use chat template with process_vision_info
        "requires_vision_utils": True,  # Requires qwen_vl_utils
    },
    "paligemma-3b": {
        "model_id": "google/paligemma-3b-mix-448",
        "processor_id": "google/paligemma-3b-mix-448",
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoProcessor",
        "prompt_template": "describe en",  # PaliGemma uses simple task prompts
    },
    "florence-2-large": {
        "model_id": "microsoft/Florence-2-large",
        "processor_id": "microsoft/Florence-2-large",
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoProcessor",
        "prompt_template": "<MORE_DETAILED_CAPTION>",  # Florence-2 uses task tokens
        "is_florence": True,  # Special handling for Florence-2
    },
    "internvl3.5-1b": {
        "model_id": "OpenGVLab/InternVL3_5-1B",
        "processor_id": "OpenGVLab/InternVL3_5-1B",
        "model_class": "AutoModel",
        "processor_class": "AutoTokenizer",
        "prompt_template": "<image>\n{instruction}",
        "is_internvl": True,  # Special handling for InternVL
        "requires_custom_image_processing": True,
    },
    "internvl3.5-2b": {
        "model_id": "OpenGVLab/InternVL3_5-2B",
        "processor_id": "OpenGVLab/InternVL3_5-2B",
        "model_class": "AutoModel",
        "processor_class": "AutoTokenizer",
        "prompt_template": "<image>\n{instruction}",
        "is_internvl": True,  # Special handling for InternVL
        "requires_custom_image_processing": True,
    },
    "internvl3.5-4b": {
        "model_id": "OpenGVLab/InternVL3_5-4B",
        "processor_id": "OpenGVLab/InternVL3_5-4B",
        "model_class": "AutoModel",
        "processor_class": "AutoTokenizer",
        "prompt_template": "<image>\n{instruction}",
        "is_internvl": True,  # Special handling for InternVL
        "requires_custom_image_processing": True,
    },
    "internvl3.5-8b": {
        "model_id": "OpenGVLab/InternVL3_5-8B",
        "processor_id": "OpenGVLab/InternVL3_5-8B",
        "model_class": "AutoModel",
        "processor_class": "AutoTokenizer",
        "prompt_template": "<image>\n{instruction}",
        "is_internvl": True,  # Special handling for InternVL
        "requires_custom_image_processing": True,
    },
    "gemma-3-4b": {
        "model_id": "google/gemma-3-4b-it",
        "processor_id": "google/gemma-3-4b-it",
        "model_class": "Gemma3ForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "use_chat_template": True,  # Uses chat template like Qwen2.5-VL
        "requires_vision_utils": False,  # Doesn't need qwen_vl_utils
    },
}


# ─────────────────────────────────────────────────────────────
# 평가 메트릭 계산
# ─────────────────────────────────────────────────────────────

def compute_text_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """BLEU-4, METEOR, ROUGE-L, SPICE, CIDEr 계산 (eval.py와 동일한 구현 사용)

    이 함수는 scripts/eval.py의 calculate_evaluation_metrics를 재사용하여
    모든 평가 스크립트에서 동일한 메트릭 계산을 보장합니다.
    """
    if USE_EVAL_METRICS:
        # Use shared implementation from eval.py for consistency
        try:
            # Create a temporary DataFrame matching eval.py's expected format
            temp_df = pd.DataFrame({
                'prediction': predictions,
                'reference': references,
            })

            # Use eval.py's calculate_evaluation_metrics
            # Note: we pass a dummy output_dir and timestamp since we only need metrics
            import tempfile
            import time
            with tempfile.TemporaryDirectory() as tmpdir:
                metrics = eval_calculate_metrics(
                    temp_df,
                    output_dir=Path(tmpdir),
                    timestamp=time.strftime('%Y%m%d_%H%M%S'),
                    prefix='temp'
                )

            logging.info("✓ 메트릭 계산 완료 (eval.py 구현 사용)")
            return metrics

        except Exception as exc:
            logging.error(f"❌ eval.py 메트릭 계산 실패, 로컬 구현으로 폴백: {exc}")
            # Fall through to local implementation

    # Local fallback implementation (kept for backwards compatibility)
    metrics: Dict[str, float] = {}

    paired = [
        (pred.strip(), ref.strip())
        for pred, ref in zip(predictions, references)
        if ref is not None and str(ref).strip() != ""
    ]

    if not paired:
        logging.warning("평가 가능한 예측-정답 쌍이 없습니다.")
        return metrics

    preds = [p for p, _ in paired]
    refs = [r for _, r in paired]

    # BLEU-4
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        smoothing = SmoothingFunction().method1
        ref_tokens = [[r.split()] for r in refs]
        pred_tokens = [p.split() for p in preds]
        if ref_tokens and pred_tokens:
            metrics["bleu4"] = float(
                corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            )
    except Exception as exc:
        logging.warning(f"BLEU-4 계산 실패: {exc}")

    # METEOR
    try:
        import nltk

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)

        from nltk.translate.meteor_score import meteor_score

        meteor_scores = []
        for ref, pred in zip(refs, preds):
            if ref and pred:
                meteor_scores.append(meteor_score([ref.split()], pred.split()))
        if meteor_scores:
            metrics["meteor"] = float(np.mean(meteor_scores))
    except Exception as exc:
        logging.warning(f"METEOR 계산 실패: {exc}")

    # ROUGE-L
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_values = []
        for ref, pred in zip(refs, preds):
            if ref and pred:
                rouge = scorer.score(ref, pred)
                rouge_values.append(rouge["rougeL"].fmeasure)
        if rouge_values:
            metrics["rougeL"] = float(np.mean(rouge_values))
    except Exception as exc:
        logging.warning(f"ROUGE-L 계산 실패: {exc}")

    # SPICE
    try:
        from pycocoevalcap.spice.spice import Spice
        spice_scorer = Spice()

        gts = {str(i): [ref] for i, ref in enumerate(refs)}
        res = {str(i): [pred] for i, pred in enumerate(preds)}

        spice_score, _ = spice_scorer.compute_score(gts, res)
        metrics["spice"] = float(spice_score)
        logging.info(f"✓ SPICE: {metrics['spice']:.4f}")
    except Exception as exc:
        logging.warning(f"SPICE 계산 실패: {exc}")
        metrics["spice"] = 0.0

    # CIDEr
    try:
        from pycocoevalcap.cider.cider import Cider
        cider_scorer = Cider()

        gts = {str(i): [ref] for i, ref in enumerate(refs)}
        res = {str(i): [pred] for i, pred in enumerate(preds)}

        cider_score, _ = cider_scorer.compute_score(gts, res)
        metrics["cider"] = float(cider_score)
        logging.info(f"✓ CIDEr: {metrics['cider']:.4f}")
    except Exception as exc:
        logging.warning(f"CIDEr 계산 실패: {exc}")
        metrics["cider"] = 0.0

    return metrics


# ─────────────────────────────────────────────────────────────
# VLM 모델 평가기
# ─────────────────────────────────────────────────────────────

class VLMEvaluator:
    def __init__(
        self,
        model_name: str,
        data_csv: str,
        output_dir: str = "eval_results",
        batch_size: int = 1,
        max_samples: Optional[int] = None,
        device: str = "cuda",
        image_size: int = 224,
        max_new_tokens: int = 128,
        image_column: str = "url",
        instruction_column: str = "query",
        response_column: str = "annotation",
    ):
        self.model_name = model_name
        self.data_csv = Path(data_csv)
        # 모델별 하위 디렉토리 생성: ablation/{모델명}
        self.output_dir = Path(output_dir) / "ablation" / model_name
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.device = device
        self.image_size = image_size
        self.max_new_tokens = max_new_tokens
        self.image_column = image_column
        self.instruction_column = instruction_column
        self.response_column = response_column

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name not in VLM_MODELS:
            raise ValueError(f"지원하지 않는 모델: {model_name}. 사용 가능: {list(VLM_MODELS.keys())}")
        
        self.model_config = VLM_MODELS[model_name]
        
        logging.info(f"모델 로드 중: {self.model_config['model_id']}")
        self._load_model_and_processor()

    def _load_model_and_processor(self):
        """모델과 프로세서 로드"""
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
            Blip2ForConditionalGeneration,
            Blip2Processor,
            Gemma3ForConditionalGeneration,
            InstructBlipForConditionalGeneration,
            InstructBlipProcessor,
            LlavaForConditionalGeneration,
            LlavaNextForConditionalGeneration,
            LlavaNextProcessor,
            LlavaOnevisionForConditionalGeneration,
            LlavaProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        model_class_name = self.model_config["model_class"]
        processor_class_name = self.model_config["processor_class"]

        # 모델 클래스 매핑
        model_classes = {
            "LlavaForConditionalGeneration": LlavaForConditionalGeneration,
            "LlavaNextForConditionalGeneration": LlavaNextForConditionalGeneration,
            "LlavaOnevisionForConditionalGeneration": LlavaOnevisionForConditionalGeneration,
            "Blip2ForConditionalGeneration": Blip2ForConditionalGeneration,
            "InstructBlipForConditionalGeneration": InstructBlipForConditionalGeneration,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoModel": AutoModel,
            "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
            "Gemma3ForConditionalGeneration": Gemma3ForConditionalGeneration,
        }

        # 프로세서 클래스 매핑
        processor_classes = {
            "LlavaProcessor": LlavaProcessor,
            "LlavaNextProcessor": LlavaNextProcessor,
            "Blip2Processor": Blip2Processor,
            "InstructBlipProcessor": InstructBlipProcessor,
            "AutoProcessor": AutoProcessor,
            "AutoTokenizer": AutoTokenizer,
        }

        model_class = model_classes.get(model_class_name)
        processor_class = processor_classes.get(processor_class_name)

        if model_class is None:
            raise ValueError(f"지원하지 않는 모델 클래스: {model_class_name}")
        if processor_class is None:
            raise ValueError(f"지원하지 않는 프로세서 클래스: {processor_class_name}")

        # 모델 로드
        # InternVL 모델은 bfloat16, 나머지는 float16
        dtype = torch.bfloat16 if "internvl" in self.model_name else torch.float16

        model_kwargs = {
            "dtype": dtype,
            "device_map": self.device,
            "trust_remote_code": True,
        }

        # FlashAttention2가 없거나 호환성 문제가 있는 경우 eager attention 사용
        # LLaVA-OneVision 모델은 transformers의 flash_attn_varlen_func를 요구하는데
        # 최신 transformers에서는 이 함수가 제거됨
        if "llava-onevision" in self.model_name or "internvl" in self.model_name:
            model_kwargs["attn_implementation"] = "eager"

        # 네트워크 이슈로 인한 재시도 로직
        max_retries = 3
        retry_delay = 10  # seconds

        for attempt in range(max_retries):
            try:
                logging.info(f"모델 다운로드 시도 {attempt + 1}/{max_retries}...")
                self.model = model_class.from_pretrained(
                    self.model_config["model_id"],
                    **model_kwargs,
                )
                break  # 성공하면 루프 종료
            except (OSError, ConnectionError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    logging.warning(f"모델 다운로드 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                    logging.info(f"{retry_delay}초 후 재시도합니다...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    logging.error(f"모델 다운로드 최종 실패: {e}")
                    raise

        self.model.eval()

        # 프로세서 로드 (재시도 로직 포함)
        for attempt in range(max_retries):
            try:
                logging.info(f"프로세서 다운로드 시도 {attempt + 1}/{max_retries}...")
                if processor_class_name == "AutoTokenizer":
                    self.processor = processor_class.from_pretrained(
                        self.model_config["processor_id"],
                        trust_remote_code=True,
                    )
                else:
                    self.processor = processor_class.from_pretrained(
                        self.model_config["processor_id"],
                        trust_remote_code=True,
                    )
                break  # 성공하면 루프 종료
            except (OSError, ConnectionError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    logging.warning(f"프로세서 다운로드 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                    logging.info(f"{retry_delay}초 후 재시도합니다...")
                    import time
                    time.sleep(retry_delay)
                else:
                    logging.error(f"프로세서 다운로드 최종 실패: {e}")
                    raise

        # Tokenizer 접근
        if hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

        # Padding token 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델별 패딩 방향 설정 (공식 구현 기준)
        # Decoder-only 모델: left padding (generation 최적화)
        # Encoder-decoder 모델: right padding (기본값)
        padding_side = self._get_padding_side()
        self.tokenizer.padding_side = padding_side
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = padding_side

        logging.info(f"모델 로드 완료: {self.model_config['model_id']}")
        logging.info(f"Padding side: {padding_side}")

    def _get_padding_side(self) -> str:
        """모델별 올바른 패딩 방향 반환 (공식 구현 기준)

        Returns:
            "left" for decoder-only models (better for generation)
            "right" for encoder-decoder models (default)
        """
        # Encoder-decoder 모델: right padding
        encoder_decoder_models = [
            "florence",  # Florence-2: encoder-decoder
        ]

        # BLIP2의 경우 language model에 따라 다름
        if "blip2" in self.model_name.lower():
            # Flan-T5 기반: encoder-decoder (right padding)
            if "flan" in self.model_config["model_id"].lower() or "t5" in self.model_config["model_id"].lower():
                return "right"
            # OPT 기반: decoder-only (left padding)
            else:
                return "left"

        # InstructBLIP의 경우도 language model에 따라 다름
        if "instructblip" in self.model_name.lower():
            if "flan" in self.model_config["model_id"].lower() or "t5" in self.model_config["model_id"].lower():
                return "right"
            else:
                return "left"

        # Encoder-decoder 모델 체크
        for model_type in encoder_decoder_models:
            if model_type in self.model_name.lower():
                return "right"

        # 나머지는 모두 decoder-only: left padding
        # LLaVA 시리즈, Qwen2.5-VL, PaliGemma, Gemma-3, InternVL 등
        return "left"

    def _load_image(self, image_path: str) -> Image.Image:
        """이미지 로드 (전처리는 processor에 맡김)"""
        img_path = Path(image_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

        # RGB로 변환만 수행, 리사이즈는 각 모델의 processor가 처리
        img = Image.open(img_path).convert("RGB")
        return img

    def _format_prompt(self, instruction: str, image: Optional[Image.Image] = None) -> str:
        """프롬프트 템플릿 적용"""
        # Chat template을 사용하는 모델의 경우
        if self.model_config.get("use_chat_template", False):
            return None  # Will be handled differently in evaluate()

        # 일반 템플릿 사용
        template = self.model_config.get("prompt_template", "{instruction}")
        return template.format(instruction=instruction)

    def _prepare_chat_messages(self, instruction: str, image_path: Optional[str] = None) -> List[Dict]:
        """Chat template 형식의 메시지 준비 (Gemma3, Qwen2.5-VL용)"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Will be replaced with actual image
                    {"type": "text", "text": instruction}
                ]
            }
        ]
        return messages

    def evaluate(self) -> Dict[str, Any]:
        """평가 실행"""
        # 데이터 로드
        df = pd.read_csv(self.data_csv)
        if self.max_samples is not None:
            df = df.head(self.max_samples)
        
        logging.info(f"데이터 로드 완료: {len(df)} 샘플")

        predictions = []
        references = []
        instructions = []
        image_paths = []

        # 배치 처리
        num_batches = math.ceil(len(df) / self.batch_size)
        
        with torch.inference_mode():
            for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {self.model_name}"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]

                batch_images = []
                batch_prompts = []
                batch_instructions = []
                batch_refs = []
                batch_paths = []

                for _, row in batch_df.iterrows():
                    try:
                        # 이미지 로드
                        img = self._load_image(row[self.image_column])
                        batch_images.append(img)

                        # 프롬프트/메시지 생성
                        instruction = str(row.get(self.instruction_column, "Describe the image."))

                        if self.model_config.get("use_chat_template", False):
                            # Chat template 사용 모델
                            batch_prompts.append(instruction)  # Store raw instruction
                        else:
                            # 일반 프롬프트 템플릿
                            prompt = self._format_prompt(instruction)
                            batch_prompts.append(prompt)

                        batch_instructions.append(instruction)
                        batch_refs.append(str(row.get(self.response_column, "")))
                        batch_paths.append(str(row.get(self.image_column, "")))
                    except Exception as e:
                        logging.warning(f"샘플 처리 실패 (idx={start_idx}): {e}")
                        continue

                if not batch_images:
                    continue

                try:
                    # 프로세서로 입력 준비
                    if self.model_config.get("is_florence", False):
                        # Florence-2: 특별 처리
                        # Florence-2는 배치 처리가 어려우므로 개별 처리
                        for inst, img, ref, path in zip(batch_prompts, batch_images, batch_refs, batch_paths):
                            # Florence-2는 task token을 사용
                            task_prompt = self.model_config.get("prompt_template", "<MORE_DETAILED_CAPTION>")

                            inputs = self.processor(text=task_prompt, images=img, return_tensors="pt")
                            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                            generated_ids = self.model.generate(
                                input_ids=inputs["input_ids"],
                                pixel_values=inputs["pixel_values"],
                                max_new_tokens=self.max_new_tokens,
                                num_beams=3,
                                do_sample=False
                            )
                            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

                            # Post-process Florence-2 output
                            parsed_answer = self.processor.post_process_generation(
                                generated_text,
                                task=task_prompt,
                                image_size=(img.width, img.height)
                            )

                            # Extract text from parsed answer
                            pred_text = str(parsed_answer.get(task_prompt, ""))

                            predictions.append(pred_text)
                            references.append(ref)
                            instructions.append(inst)
                            image_paths.append(path)

                        continue  # Skip the rest of the batch processing

                    elif self.model_config.get("is_internvl", False):
                        # InternVL3.5: 특별 처리
                        # InternVL3.5는 chat 메서드를 사용
                        for inst, img, ref, path in zip(batch_prompts, batch_images, batch_refs, batch_paths):
                            # 이미지 전처리
                            pixel_values = load_image_internvl(img, input_size=448, max_num=12)
                            pixel_values = pixel_values.to(torch.bfloat16).to(self.device)

                            # 프롬프트 생성
                            question = f"<image>\n{inst}"

                            generation_config = {
                                "max_new_tokens": self.max_new_tokens,
                                "do_sample": False,
                            }

                            # InternVL의 chat 메서드 사용
                            response = self.model.chat(
                                self.tokenizer,
                                pixel_values,
                                question,
                                generation_config
                            )

                            predictions.append(response)
                            references.append(ref)
                            instructions.append(inst)
                            image_paths.append(path)

                        continue  # Skip the rest of the batch processing

                    elif self.model_config.get("use_chat_template", False):
                        # Chat template 사용 (Gemma3, Qwen2.5-VL)
                        if self.model_config.get("requires_vision_utils", False):
                            # Qwen2.5-VL: process_vision_info 필요
                            from qwen_vl_utils import process_vision_info

                            # Prepare messages for each sample in batch
                            all_messages = []
                            for inst, img in zip(batch_prompts, batch_images):
                                messages = [{
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": img},
                                        {"type": "text", "text": inst}
                                    ]
                                }]
                                all_messages.append(messages)

                            # Process each sample individually for Qwen2.5-VL
                            texts = []
                            all_image_inputs = []
                            all_video_inputs = []
                            for messages in all_messages:
                                text = self.processor.apply_chat_template(
                                    messages, tokenize=False, add_generation_prompt=True
                                )
                                texts.append(text)
                                image_inputs, video_inputs = process_vision_info(messages)
                                all_image_inputs.extend(image_inputs if image_inputs else [])
                                all_video_inputs.extend(video_inputs if video_inputs else [])

                            inputs = self.processor(
                                text=texts,
                                images=all_image_inputs if all_image_inputs else None,
                                videos=all_video_inputs if all_video_inputs else None,
                                padding=True,
                                return_tensors="pt",
                            )
                        else:
                            # Gemma3, LLaVA-OneVision: 일반 chat template
                            # pixel_values가 가변 길이이므로 개별 처리
                            for inst, img, ref, path in zip(batch_prompts, batch_images, batch_refs, batch_paths):
                                try:
                                    messages = [{
                                        "role": "user",
                                        "content": [
                                            {"type": "image", "image": img},
                                            {"type": "text", "text": inst}
                                        ]
                                    }]

                                    # Use processor's apply_chat_template (single sample)
                                    inputs = self.processor.apply_chat_template(
                                        [messages],
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_dict=True,
                                        return_tensors="pt"
                                    )

                                    # GPU로 이동
                                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                             for k, v in inputs.items()}

                                    # Generation
                                    gen_kwargs = {
                                        "max_new_tokens": self.max_new_tokens,
                                        "do_sample": False,
                                        "num_beams": 1,
                                    }
                                    if self.tokenizer.pad_token_id is not None:
                                        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
                                    if self.tokenizer.eos_token_id is not None:
                                        gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

                                    outputs = self.model.generate(**inputs, **gen_kwargs)

                                    # Decode
                                    if isinstance(outputs, tuple):
                                        outputs = outputs[0]

                                    # Prompt 길이 계산
                                    input_ids = inputs.get("input_ids")
                                    if input_ids is not None:
                                        prompt_length = (input_ids[0] != self.tokenizer.pad_token_id).sum().item()
                                    else:
                                        prompt_length = 0

                                    # Prompt 부분 제거
                                    generated_tokens = outputs[0][prompt_length:].tolist()
                                    pred_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                                    predictions.append(pred_text)
                                    references.append(ref)
                                    instructions.append(inst)
                                    image_paths.append(path)

                                except Exception as e:
                                    logging.warning(f"샘플 처리 실패 (batch={batch_idx}): {e}")
                                    continue

                            continue  # Skip the rest of the batch processing
                    else:
                        # 일반 모델: 기존 방식
                        inputs = self.processor(
                            text=batch_prompts,
                            images=batch_images,
                            return_tensors="pt",
                            padding=True,
                        )

                    # GPU로 이동
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                    # Generation
                    gen_kwargs = {
                        "max_new_tokens": self.max_new_tokens,
                        "do_sample": False,
                        "num_beams": 1,
                    }

                    if self.tokenizer.pad_token_id is not None:
                        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
                    if self.tokenizer.eos_token_id is not None:
                        gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

                    outputs = self.model.generate(**inputs, **gen_kwargs)

                    # Decode
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    # Prompt 길이 계산
                    input_ids = inputs.get("input_ids")
                    if input_ids is not None:
                        prompt_lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1).cpu()
                    else:
                        prompt_lengths = torch.zeros(len(batch_prompts), dtype=torch.long)

                    for i, output in enumerate(outputs):
                        # Prompt 부분 제거
                        cut = int(prompt_lengths[i].item()) if i < len(prompt_lengths) else 0
                        generated_tokens = output[cut:].tolist()
                        pred_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                        predictions.append(pred_text)
                        references.append(batch_refs[i])
                        instructions.append(batch_instructions[i])
                        image_paths.append(batch_paths[i])

                except Exception as e:
                    logging.error(f"배치 생성 실패 (batch={batch_idx}): {e}")
                    continue

        # 메트릭 계산
        metrics = compute_text_metrics(predictions, references)
        
        # 결과 저장
        results = {
            "model_name": self.model_name,
            "model_id": self.model_config["model_id"],
            "num_samples": len(predictions),
            "image_size": f"{self.image_size}x{self.image_size}",
            "metrics": metrics,
        }

        # 메트릭 저장 (JSON만)
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logging.info(f"메트릭 저장: {metrics_file}")

        # 예측 결과 저장 (CSV)
        predictions_df = pd.DataFrame({
            "image_path": image_paths,
            "instruction": instructions,
            "reference": references,
            "prediction": predictions,
        })
        predictions_file = self.output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_file, index=False, encoding="utf-8")
        logging.info(f"예측 결과 저장: {predictions_file}")

        return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VLM 모델 평가 (학습 없이)")
    parser.add_argument("--data_csv", type=str, required=True, help="평가 데이터 CSV 파일")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["llava-1.5-7b", "blip2-opt-2.7b"],
        help=f"평가할 모델 이름들. 사용 가능: {list(VLM_MODELS.keys())}",
    )
    parser.add_argument("--output_dir", type=str, default="eval_results", help="결과 저장 디렉토리")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기")
    parser.add_argument("--max_samples", type=int, default=None, help="최대 평가 샘플 수 (디버깅용)")
    parser.add_argument("--device", type=str, default="cuda", help="디바이스 (cuda/cpu)")
    parser.add_argument("--image_size", type=int, default=224, help="이미지 크기 (정사각형)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="최대 생성 토큰 수")
    parser.add_argument("--image_column", type=str, default="url", help="이미지 경로 컬럼명")
    parser.add_argument("--instruction_column", type=str, default="query", help="질문 컬럼명 (기본: query)")
    parser.add_argument("--response_column", type=str, default="annotation", help="정답 컬럼명 (기본: annotation)")
    parser.add_argument("--log_level", type=str, default="INFO", help="로그 레벨")

    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # 각 모델 평가
    all_results = []
    for model_name in args.models:
        logging.info(f"\n{'='*60}")
        logging.info(f"모델 평가 시작: {model_name}")
        logging.info(f"{'='*60}\n")

        try:
            evaluator = VLMEvaluator(
                model_name=model_name,
                data_csv=args.data_csv,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                device=args.device,
                image_size=args.image_size,
                max_new_tokens=args.max_new_tokens,
                image_column=args.image_column,
                instruction_column=args.instruction_column,
                response_column=args.response_column,
            )
            
            results = evaluator.evaluate()
            all_results.append(results)

            # 메트릭 출력
            logging.info(f"\n{model_name} 평가 결과:")
            for key, value in results["metrics"].items():
                logging.info(f"  {key}: {value:.4f}")

        except Exception as e:
            logging.error(f"모델 {model_name} 평가 실패: {e}")
            import traceback
            traceback.print_exc()
            continue

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 전체 결과 요약
    if all_results:
        summary_file = Path(args.output_dir) / "all_models_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logging.info(f"\n전체 결과 저장: {summary_file}")

        # 비교표 생성
        logging.info("\n" + "="*60)
        logging.info("모델 성능 비교")
        logging.info("="*60)
        for result in all_results:
            logging.info(f"\n{result['model_name']}:")
            for key, value in result["metrics"].items():
                if isinstance(value, (int, float)):
                    logging.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
