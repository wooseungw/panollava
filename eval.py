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
    python eval.py --model-dir runs/<run_name>/hf_model --lora-weights-path runs/<run_name>/lora_weights --csv-input data/quic360/test.csv
"""

import argparse
import torch
import json
import logging
import time
import traceback
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# 내부 모듈
# Silence HF tokenizers fork/parallelism warnings and avoid deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from panovlm.dataset import VLMDataModule
from panovlm.processors.universal_text_formatter import UniversalTextFormatter

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def resolve_model_dir(config_or_path, stage: str = None, crop_strategy: str = None) -> str:
    """
    HF-style 모델 디렉토리 자동 탐색 (PyTorch bin 기반)
    - config_or_path: dict 또는 JSON 파일 경로(str)
    - stage/crop_strategy: runs/<prefix>_<crop>_<stage>_<resampler>/hf_model 힌트 구성에 사용
    """
    try:
        # config 로딩 (dict 또는 파일 경로)
        if isinstance(config_or_path, (str, Path)):
            with open(config_or_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        elif isinstance(config_or_path, dict):
            config = config_or_path
        else:
            raise TypeError(f"Unsupported config type: {type(config_or_path)}")

        prefix = config.get('training', {}).get('prefix')
        if not prefix:
            raise KeyError("training.prefix is required in config.json")

        resampler = config.get('models', {}).get('resampler_type') or config.get('models', {}).get('resampler', 'mlp')
        if stage is None:
            stage = config.get('training', {}).get('default_stage', 'finetune')

        if crop_strategy is None:
            crop_strategy = config.get('image_processing', {}).get('crop_strategy', 'e2p')

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
        candidate_run_dir = Path(paths_cfg.get('runs_dir', 'runs')) / f"{prefix}_{crop_strategy}_{stage}_{resampler}"
        # 1) HF 폴더 우선
        hf_dir = candidate_run_dir / 'hf_model'
        if hf_dir.exists() and hf_dir.is_dir():
            logger.info(f"✅ Using HF model dir: {str(hf_dir)}")
            return str(hf_dir)

        # 2) panorama_model 폴더도 허용 (동일한 규약으로 저장된 경우)
        pano_dir = candidate_run_dir / 'panorama_model'
        if pano_dir.exists() and pano_dir.is_dir():
            logger.info(f"✅ Using panorama_model dir: {str(pano_dir)}")
            return str(pano_dir)

        # 3) 체크포인트(.ckpt) 자동 선택 (best > last > any)
        best_ckpt = candidate_run_dir / 'best.ckpt'
        last_ckpt = candidate_run_dir / 'last.ckpt'
        if best_ckpt.exists():
            logger.info(f"✅ Using best checkpoint: {str(best_ckpt)}")
            return str(best_ckpt)
        if last_ckpt.exists():
            logger.info(f"✅ Using last checkpoint: {str(last_ckpt)}")
            return str(last_ckpt)
        # any *.ckpt as fallback
        try:
            any_ckpts = sorted(candidate_run_dir.glob('*.ckpt'))
            if any_ckpts:
                logger.info(f"✅ Using checkpoint: {str(any_ckpts[0])}")
                return str(any_ckpts[0])
        except Exception:
            pass

        raise FileNotFoundError("No pretrained model dir found. Set paths.pretrained_dir or pass --model-dir")

    except Exception as e:
        logger.error(f"Failed to resolve model dir: {e}")
        raise



def load_model_and_lora(
    model_dir: str,
    lora_weights_path: Optional[str],
    device: torch.device,
    config_path: Optional[str] = None,
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
    if config_path:
        try:
            from panovlm.config import ModelConfig
            try:
                config_obj = ModelConfig.load(config_path)
                logger.info(f"📋 ModelConfig 로드 완료(from {config_path})")
            except Exception as e:
                logger.warning(f"ModelConfig.load 실패, JSON dict로 대체: {e}")
                with open(config_path, "r", encoding="utf-8") as f:
                    config_obj = json.load(f)
        except Exception as e:
            logger.warning(f"panovlm.config.ModelConfig 사용 불가, JSON dict로 대체: {e}")
            with open(config_path, "r", encoding="utf-8") as f:
                config_obj = json.load(f)

    # ── PanoramaVLM (HF 디렉토리 또는 .ckpt) ──────────────────────
    try:
        from panovlm.model import PanoramaVLM

        # from_checkpoint/from_pretrained_dir에 config/model_config 어느 쪽 이름을 쓰는지 모듈별로 다를 수 있어
        # 모두 안전하게 전달(받는 쪽에서 무시해도 무해)
        extra_cfg = {}
        if config_obj is not None:
            extra_cfg["config"] = config_obj
            extra_cfg["model_config"] = config_obj

        mpath = Path(model_dir)
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

    datamodule = VLMDataModule(
        csv_train=csv_input,
        csv_val=csv_input,  # 평가용으로 동일한 파일 사용
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer_name=lm_name,
        max_text_length=max_text_length,
        crop_strategy=crop_strategy,
        eval_mode=True,
        system_msg=system_msg,
        overlap_ratio=overlap_ratio,
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
    logger.info(f"   - use_vision_processor: {use_vision_processor}")

    return datamodule, test_dataloader

def generate_predictions(
    model: Any,
    test_dataloader,
    datamodule: VLMDataModule,
    device: torch.device,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    length_penalty: float = 1.0,
    min_new_tokens: int = 5,
    system_msg: Optional[str] = None
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
    tokenizer_name = getattr(tokenizer, 'name_or_path', 'Qwen/Qwen2.5-0.5B-Instruct')
    sys_msg = system_msg or "You are an expert assistant specialized in analyzing panoramic images. Please provide detailed, accurate, and helpful responses about what you observe in the panoramic view shortly."
    text_formatter = UniversalTextFormatter(
        tokenizer_name_or_path=tokenizer_name,
        system_msg=sys_msg
    )

    logger.info(f"🎯 생성 파라미터 - Max tokens: {max_new_tokens}, Min tokens: {min_new_tokens}, Temperature: {temperature}")
    logger.info(f"📝 텍스트 포맷터 - 모델: {text_formatter.model_family} ({'Instruct' if text_formatter.is_instruct else 'Base'})")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="생성 중")):
            try:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch.get("input_ids")
                if input_ids is not None:
                    input_ids = input_ids.to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

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
                logger.info(f"=== 배치 {batch_idx} 결과 로그 ===")
                for i, (pred, ref) in enumerate(zip(cleaned_predictions, batch_references)):
                    logger.info(f"  샘플 {len(predictions) + i}")
                    logger.info(f"    예측: '{pred}'")
                    logger.info(f"    정답: '{ref}'")
                logger.info(f"==========================")

                predictions.extend(cleaned_predictions)
                references.extend(batch_references)
                image_paths.extend(batch_image_paths)
                input_texts.extend(batch_input_texts)

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

    logger.info(f"✓ 텍스트 생성 완료! 총 샘플 수: {len(predictions)}")
    return predictions, references, image_paths, input_texts



def save_and_log_results(predictions: List[str], references: List[str], image_paths: List[str], input_texts: List[str], output_dir: Path, timestamp: str) -> pd.DataFrame:
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
    csv_path = output_dir / f"predictions_{timestamp}.csv"
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
    


def calculate_evaluation_metrics(data_input, output_dir: Path, timestamp: str) -> Dict[str, float]:
    """
    5단계: 평가 메트릭 계산 (BLEU-4, METEOR, ROUGE-L, SPICE, CIDEr, CLIP-S, RefCLIP-S)
    
    Args:
        data_input: pandas DataFrame 또는 CSV 파일 경로 (str/Path)
        output_dir: 결과 저장 디렉토리
        timestamp: 타임스탬프 문자열
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
    predictions = [str(pred) if pred is not None and not pd.isna(pred) else "" for pred in valid_df['prediction'].tolist()]
    references = [str(ref) if ref is not None and not pd.isna(ref) else "" for ref in valid_df['reference'].tolist()]
    
    # 빈 문자열 필터링
    valid_pairs = [(pred, ref) for pred, ref in zip(predictions, references) if pred.strip() and ref.strip()]
    
    if not valid_pairs:
        logger.error("❌ 유효한 예측-정답 쌍이 없습니다.")
        return {}
    
    predictions, references = zip(*valid_pairs)
    predictions = list(predictions)
    references = list(references)
    
    logger.info(f"📊 최종 평가 대상: {len(valid_pairs)} 샘플")
    
    metrics = {}
    
    # Assistant 응답 부분만 추출 (정답용) - NaN 처리 추가
    ref_texts_for_bleu = []
    for ref in references:
        if "Assistant:" in ref:
            assistant_part = ref.split("Assistant:")[-1].strip()
            ref_texts_for_bleu.append(assistant_part)
        else:
            ref_texts_for_bleu.append(ref)
    
    # 1. BLEU-4 계산
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        
        ref_tokens = [[ref.split()] for ref in ref_texts_for_bleu if ref.strip()]
        pred_tokens = [pred.split() for pred in predictions if pred.strip()]
        
        
        if len(ref_tokens) == 0 or len(pred_tokens) == 0:
            logger.warning("⚠️ BLEU-4: 유효한 토큰이 없습니다.")
            metrics['bleu4'] = 0.0
        else:
            smoothing = SmoothingFunction().method1
            metrics['bleu4'] = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            logger.info(f"✓ BLEU-4: {metrics['bleu4']:.4f}")
    except Exception as e:
        logger.error(f"❌ BLEU-4 계산 오류: {e}")
        metrics['bleu4'] = 0.0
    
    # 2. METEOR 계산
    try:
        import nltk
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("NLTK 데이터 다운로드 중...")
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
        
        from nltk.translate.meteor_score import meteor_score
        
        meteor_scores = []
        for ref, pred in zip(ref_texts_for_bleu, predictions):
            if ref.strip() and pred.strip():  # 빈 문자열 체크
                ref_tokens = ref.split()
                pred_tokens = pred.split()
                if len(ref_tokens) > 0 and len(pred_tokens) > 0:
                    score = meteor_score([ref_tokens], pred_tokens)
                    meteor_scores.append(score)
        
        if meteor_scores:
            metrics['meteor'] = float(np.mean(meteor_scores))
            logger.info(f"✓ METEOR: {metrics['meteor']:.4f}")
        else:
            logger.warning("⚠️ METEOR: 유효한 점수가 없습니다.")
            metrics['meteor'] = 0.0
    except Exception as e:
        logger.error(f"❌ METEOR 계산 오류: {e}")
        metrics['meteor'] = 0.0
    
    # 3. ROUGE-L 계산
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        rouge_scores = []
        for ref, pred in zip(ref_texts_for_bleu, predictions):
            if ref.strip() and pred.strip():  # 빈 문자열 체크
                scores = scorer.score(ref, pred)
                rouge_scores.append(scores['rougeL'].fmeasure)
        
        if rouge_scores:
            metrics['rougeL'] = float(np.mean(rouge_scores))
            logger.info(f"✓ ROUGE-L: {metrics['rougeL']:.4f}")
        else:
            logger.warning("⚠️ ROUGE-L: 유효한 점수가 없습니다.")
            metrics['rougeL'] = 0.0
        logger.info(f"✓ ROUGE-L: {metrics['rougeL']:.4f}")
    except Exception as e:
        logger.error(f"❌ ROUGE-L 계산 오류: {e}")
        metrics['rougeL'] = 0.0
    
    # 4. SPICE 계산 (더 안전한 타임아웃 처리)
    try:
        from pycocoevalcap.spice.spice import Spice
        
        # SPICE 계산을 더 안전하게 처리
        spice_scorer = Spice()
        
        # 빈 문자열 필터링
        valid_refs_for_spice = [ref for ref in ref_texts_for_bleu if ref.strip()]
        valid_preds_for_spice = [pred for pred in predictions if pred.strip()]
        
        if len(valid_refs_for_spice) == 0 or len(valid_preds_for_spice) == 0:
            logger.warning("⚠️ SPICE: 유효한 텍스트가 없습니다.")
            metrics['spice'] = 0.0
        else:
            gts = {str(i): [ref] for i, ref in enumerate(valid_refs_for_spice)}
            res = {str(i): [pred] for i, pred in enumerate(valid_preds_for_spice)}
            
            # 멀티프로세싱을 이용한 타임아웃 처리 (더 안전함)
            import multiprocessing
            import queue
            
            def spice_calculate(gts, res, result_queue):
                try:
                    spice_score, _ = spice_scorer.compute_score(gts, res)
                    result_queue.put(('success', spice_score))
                except Exception as e:
                    result_queue.put(('error', str(e)))
            
            # 프로세스를 사용한 타임아웃
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=spice_calculate, args=(gts, res, result_queue))
            process.start()
            process.join(timeout=60)  # 60초 타임아웃
            
            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutError("SPICE calculation timeout (60s)")
            
            # 결과 확인
            try:
                result_type, result_value = result_queue.get_nowait()
                if result_type == 'success':
                    metrics['spice'] = float(result_value)
                    logger.info(f"✓ SPICE: {metrics['spice']:.4f}")
                else:
                    raise Exception(f"SPICE calculation failed: {result_value}")
            except queue.Empty:
                raise Exception("SPICE calculation returned no result")
            
    except (Exception, TimeoutError) as e:
        logger.warning(f"⚠️ SPICE 계산 오류: {e}")
        # SPICE 대안: 간단한 의미적 유사도 계산
        try:
            logger.info("SPICE 대안으로 의미적 유사도 계산 시도...")
            from sentence_transformers import SentenceTransformer
            model_st = SentenceTransformer('all-MiniLM-L6-v2')
            
            pred_embeddings = model_st.encode(predictions)
            ref_embeddings = model_st.encode(ref_texts_for_bleu)
            
            # 코사인 유사도 계산
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                similarities.append(sim)
            
            metrics['spice'] = float(np.mean(similarities))
            logger.info(f"✓ SPICE (대안-의미유사도): {metrics['spice']:.4f}")
        except Exception as fallback_e:
            logger.warning(f"⚠️ SPICE 대안 계산도 실패: {fallback_e}")
            metrics['spice'] = 0.0
    
    # 5. CIDEr 계산
    try:
        from pycocoevalcap.cider.cider import Cider
        cider_scorer = Cider()
        
        # 빈 문자열 필터링
        valid_refs_for_cider = [ref for ref in ref_texts_for_bleu if ref.strip()]
        valid_preds_for_cider = [pred for pred in predictions if pred.strip()]
        
        if len(valid_refs_for_cider) == 0 or len(valid_preds_for_cider) == 0:
            logger.warning("⚠️ CIDEr: 유효한 텍스트가 없습니다.")
            metrics['cider'] = 0.0
        else:
            gts = {str(i): [ref] for i, ref in enumerate(valid_refs_for_cider)}
            res = {str(i): [pred] for i, pred in enumerate(valid_preds_for_cider)}
            
            cider_score, _ = cider_scorer.compute_score(gts, res)
            metrics['cider'] = float(cider_score)
            logger.info(f"✓ CIDEr: {metrics['cider']:.4f}")
    except Exception as e:
        logger.warning(f"⚠️ CIDEr 계산 오류: {e}")
        metrics['cider'] = 0.0
    
    # 6. CLIP Score 측정 제거됨 (사용자 요청)
    logger.info("ℹ️ CLIP Score 및 RefCLIP-S 측정이 제거되었습니다.")
    
    # 메트릭 저장
    metrics_path = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 메트릭 저장: {metrics_path}")
    return metrics


def print_final_results(metrics: Dict[str, float]):
    """
    최종 결과 출력
    """
    print("\n" + "=" * 80)
    print("🎉 PanoLLaVA 모델 평가 완료")
    print("=" * 80)
    
    print("\n📊 평가 메트릭 결과:")
    print("-" * 40)
    
    if 'bleu4' in metrics:
        print(f"BLEU-4     (↑): {metrics['bleu4']:.4f}")
    if 'meteor' in metrics:
        print(f"METEOR     (↑): {metrics['meteor']:.4f}")
    if 'rougeL' in metrics:
        print(f"ROUGE-L    (↑): {metrics['rougeL']:.4f}")
    if 'spice' in metrics:
        print(f"SPICE      (↑): {metrics['spice']:.4f}")
    if 'cider' in metrics:
        print(f"CIDEr      (↑): {metrics['cider']:.4f}")
    # CLIP Score 출력 제거됨
    
    print("-" * 40)
    print("💡 (↑) 표시는 높을수록 좋은 메트릭입니다.")
    print("=" * 80)


def load_global_config():
    """Load global configuration from config.json"""
    config_path = Path("config.json")
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
    return {}

def main():
    # Load global configuration
    global_config = load_global_config()

    env_config = global_config.get("environment", {})
    model_config = global_config.get("models", {})
    data_config = global_config.get("data", {})
    training_config = global_config.get("training", {})
    image_cfg = global_config.get("image_processing", {})
    system_msgs = global_config.get("system_messages", {})

    # 디바이스 설정: 환경변수 대신 config 기반으로 GPU index를 선택
    cuda_vis = env_config.get("cuda_visible_devices")
    if cuda_vis and torch.cuda.is_available():
        try:
            # 첫 번째 인덱스만 사용 (예: "1" 또는 "0,1" → 1 또는 0)
            first_idx = int(str(cuda_vis).split(",")[0].strip())
            torch.cuda.set_device(first_idx)
            logger.info(f"Device: using GPU index {first_idx} (from config)")
        except Exception as e:
            logger.warning(f"Invalid cuda_visible_devices in config: {cuda_vis} ({e})")

    parser = argparse.ArgumentParser(description="PanoLLaVA 모델 평가 시스템")
    parser.add_argument('--model-dir', default=None, help='모델 디렉토리(hf_model 또는 panorama_model). 지정 없으면 config.paths.pretrained_dir 자동 사용')
    parser.add_argument('--stage', default=None, help='학습 단계 (e.g., finetune, resampler, vision)')
    parser.add_argument('--lora-weights-path', default=None, help='LoRA 가중치 경로 (자동으로 체크포인트 경로에서 찾음)')
    parser.add_argument('--csv-input', default=data_config.get("csv_test", "data/quic360/test.csv"), help='테스트 CSV 파일 경로')
    parser.add_argument('--output-dir', default='eval_results', help='결과 저장 디렉토리')

    # ✔ config 키 정정: vision_name / lm_model / resampler
    parser.add_argument('--vision-name', default=model_config.get("vision_name", "google/siglip-base-patch16-224"))
    parser.add_argument('--lm-name', default=model_config.get("lm_model", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser.add_argument('--resampler', default=model_config.get("resampler", "mlp"))

    # ✔ image_processing에서 crop_strategy/overlap_ratio/use_vision_processor 가져오기
    parser.add_argument('--crop-strategy', default=image_cfg.get("crop_strategy", "e2p"),
                        choices=['sliding_window', 'e2p', 'cubemap', 'resize', 'anyres', 'anyres_max'])
    parser.add_argument('--overlap-ratio', type=float, default=image_cfg.get("overlap_ratio", 0.5))
    parser.add_argument('--use-vision-processor', action='store_true' if image_cfg.get("use_vision_processor", True) else 'store_false')
    # Allow "auto" or an integer for max text length
    parser.add_argument(
        '--max-text-length',
        type=str,
        default=str(training_config.get("max_text_length", data_config.get("max_text_length", 256))),
        help='Max text length for tokenization. Use an integer, "auto" (model-based), or "auto:dataset" (measure from CSV).'
    )

    # 생성 관련
    # generation 섹션에서 max_new_tokens 가져오기(키 유연 처리)
    gen_cfg = global_config.get("generation", {}) if isinstance(global_config, dict) else {}
    def _get_gen_default(*keys, fallback=None):
        for k in keys:
            if isinstance(gen_cfg, dict) and k in gen_cfg:
                return gen_cfg.get(k)
        return fallback

    parser.add_argument('--max-new-tokens', type=int, default=_get_gen_default('max_new_tokens', 'max_new_token', 'max_tokens', fallback=128))
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--min-new-tokens', type=int, default=5)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    parser.add_argument('--length-penalty', type=float, default=1.0)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=training_config.get("num_workers", 16), help='데이터로더 워커 수')

    # 시스템 메시지 (training.system_msg 우선, 없으면 system_messages.default)
    default_sys_msg = training_config.get("system_msg", system_msgs.get("default", "You are a helpful assistant."))
    parser.add_argument('--system-msg', type=str, default=default_sys_msg)

    # 설정 시스템 파라미터들
    parser.add_argument('--config', help='ModelConfig JSON 파일 경로 (미지정 시 config.json 로드값 사용)')

    args = parser.parse_args()

    # 모델 디렉토리 자동 해결 (args.config가 없으면 로드된 global_config 사용)
    model_dir = args.model_dir
    if model_dir is None:
        cfg_source = args.config if args.config else global_config
        model_dir = resolve_model_dir(cfg_source, args.stage, crop_strategy=args.crop_strategy)

    # LoRA 가중치 자동 설정
    lora_weights_path = args.lora_weights_path
    if lora_weights_path is None and model_dir:
        checkpoint_dir = Path(model_dir)
        checkpoint_dir = checkpoint_dir if checkpoint_dir.is_dir() else checkpoint_dir.parent
        potential_lora_path = checkpoint_dir / "lora_weights"
        if potential_lora_path.exists():
            lora_weights_path = str(potential_lora_path)
            logger.info(f"✅ Auto-found LoRA weights: {lora_weights_path}")

    # 출력 디렉토리
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  사용 디바이스: {device}")

    try:
        # 1단계: 모델 및 LoRA 가중치 로드
        # Convert max_text_length for model only if numeric; otherwise omit (DataModule handles "auto")
        _mtl_val = None
        try:
            _mtl_val = int(args.max_text_length)
        except Exception:
            _mtl_val = None
        model_kwargs = {
            "vision_name": args.vision_name,
            "lm_name": args.lm_name,
            "resampler": args.resampler,
            "lr": 1e-5,
            **({"max_text_length": _mtl_val} if _mtl_val is not None else {})
        }
        model = load_model_and_lora(
            model_dir,
            lora_weights_path,
            device,
            config_path=args.config,  # ModelConfig를 별도로 쓰는 경우
            **model_kwargs
        )

        # 2단계: 테스트 데이터셋 준비 (config 반영 인자 추가)
        datamodule, test_dataloader = prepare_test_dataset(
            csv_input=args.csv_input,
            batch_size=args.batch_size,
            max_text_length=args.max_text_length,
            crop_strategy=args.crop_strategy,
            lm_name=args.lm_name,
            num_workers=args.num_workers,
            overlap_ratio=args.overlap_ratio,
            vision_name=args.vision_name,
            system_msg=args.system_msg,
            use_vision_processor=args.use_vision_processor,
            auto_max_text_length_cap=int(global_config.get("data", {}).get("auto_max_text_length_cap", 8192)) if isinstance(global_config, dict) else 8192,
            auto_max_text_length_floor=int(global_config.get("data", {}).get("auto_max_text_length_floor", 512)) if isinstance(global_config, dict) else None,
            auto_max_text_length_scan_limit=int(global_config.get("data", {}).get("auto_max_text_length_scan_limit", 1000)) if isinstance(global_config, dict) else None
        )

        # 3단계: 텍스트 생성 (system_msg 전달)
        predictions, references, image_paths, input_texts = generate_predictions(
            model, test_dataloader, datamodule, device,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty, length_penalty=args.length_penalty,
            min_new_tokens=args.min_new_tokens,
            system_msg=args.system_msg
        )

        # 4단계: 결과 저장 및 로깅
        df = save_and_log_results(predictions, references, image_paths, input_texts, output_dir, timestamp)

        # 5단계: 평가 메트릭 계산
        metrics = calculate_evaluation_metrics(df, output_dir, timestamp)

        # 최종 결과 출력
        print_final_results(metrics)

    except Exception as e:
        logger.error(f"❌ 평가 중 오류 발생: {e}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        raise

if __name__ == '__main__':
    main()
