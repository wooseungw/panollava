# coding: utf-8
"""
PanoLLaVA Comprehensive Model Evaluation
──────────────────────────────────────────
최종 모델 종합 평가 시스템:
• vlm_finetune 및 vlm_resampler 모델 지원
• 모든 평가 메트릭 포함 (BLEU, ROUGE, METEOR, CIDEr, CLIP Score)
• generate 기반의 정확한 텍스트 생성 평가
• GPU 메모리 효율적 처리 및 OOM 방지
• 상세한 결과 분석 및 비교

사용 예시:
    # Finetune 모델 평가
    python eval.py --ckpt runs/e2p_finetune_mlp/best-v1.ckpt --lora-weights-path runs/e2p_finetune_mlp/lora_weights --csv-input data/quic360/test.csv
"""

import argparse
import torch
import json
import logging
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from collections import defaultdict

# 내부 모듈
from train import VLMModule, VLMDataModule, get_gpu_memory_info, safe_load_checkpoint

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(checkpoint_path: str, stage: str, device: torch.device, 
               lora_weights_path: Optional[str] = None, **kwargs) -> VLMModule:
    """모델 로드 (LoRA 지원)"""
    logger.info(f"Loading {stage} model from: {checkpoint_path}")
    
    # 체크포인트 검증
    checkpoint = safe_load_checkpoint(checkpoint_path)
    if not checkpoint:
        raise ValueError(f"Failed to load checkpoint: {checkpoint_path}")
    
    # LoRA 자동 감지
    auto_detected_lora_path = None
    if lora_weights_path is None:
        # 체크포인트와 같은 디렉토리에서 lora_weights 폴더 찾기
        checkpoint_dir = Path(checkpoint_path).parent
        potential_lora_path = checkpoint_dir / "lora_weights"
        if potential_lora_path.exists():
            auto_detected_lora_path = str(potential_lora_path)
            logger.info(f"🔍 Auto-detected LoRA weights: {auto_detected_lora_path}")
    
    final_lora_path = lora_weights_path or auto_detected_lora_path
    
    # 모델 로드 - evaluation을 위해서는 원래 stage로 로드하되 나중에 generate 모드로 설정
    original_stage = stage
    model = VLMModule.load_from_checkpoint(
        checkpoint_path,
        stage=original_stage,
        map_location=device,
        **kwargs
    )
    
    # LoRA 가중치 로드 (finetune 단계에서만)
    if final_lora_path and stage == "finetune":
        try:
            logger.info(f"🔧 Loading LoRA weights from: {final_lora_path}")
            
            # LoRA 파일 존재 확인
            lora_path = Path(final_lora_path)
            if not lora_path.exists():
                logger.warning(f"⚠️ LoRA weights path does not exist: {final_lora_path}")
                logger.info("Continuing with base model...")
            else:
                # LoRA 파일 구조 확인
                adapter_config = lora_path / "adapter_config.json"
                adapter_model = lora_path / "adapter_model.safetensors"
                
                if not adapter_config.exists():
                    logger.warning(f"⚠️ adapter_config.json not found in {final_lora_path}")
                if not adapter_model.exists():
                    logger.warning(f"⚠️ adapter_model.safetensors not found in {final_lora_path}")
                
                success = model.model.load_lora_weights(final_lora_path)
                if success:
                    logger.info("✅ LoRA weights loaded successfully!")
                    
                    # LoRA 정보 출력
                lora_info = model.model.get_lora_info()
                if lora_info.get("is_lora_enabled", False):
                    logger.info(f"📊 LoRA Configuration:")
                    logger.info(f"   - Rank: {lora_info.get('lora_r', 'N/A')}")
                    logger.info(f"   - Alpha: {lora_info.get('lora_alpha', 'N/A')}")
                    logger.info(f"   - Dropout: {lora_info.get('lora_dropout', 'N/A')}")
                    logger.info(f"   - Target modules: {lora_info.get('target_modules', 'N/A')}")
                else:
                    logger.warning("⚠️ LoRA weights loading failed, using base model")
        except Exception as e:
            logger.warning(f"⚠️ Error loading LoRA weights: {e}")
            logger.info("Continuing with base model...")
    elif final_lora_path and stage != "finetune":
        logger.warning(f"⚠️ LoRA weights found but stage is '{stage}'. LoRA only supported for finetune stage.")
    
    # 평가 모드로 설정
    model.eval()
    model = model.to(device)
    
    # 모든 파라미터를 unfroze하여 generate할 수 있도록 설정
    model.model.requires_grad_(False)  # gradient는 필요없지만 forward는 가능하도록
    
    logger.info(f"✓ {stage.capitalize()} model loaded successfully on {device}")
    logger.info(f"Model stage: {model._stage_key}")
    
    return model

def evaluate_model(model: VLMModule, dataloader, stage: str, device: torch.device, 
                  generation_params: Dict[str, Any], datamodule) -> Tuple[List[str], List[str], List[Dict], int]:
    """모델 평가 수행"""
    logger.info(f"Starting {stage} model evaluation...")
    
    predictions = []
    references = []
    metadata = []
    error_count = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {stage}")):
            try:
                # 입력 데이터 준비
                pixel_values = batch["pixel_values"].to(device)
                batch_size = pixel_values.shape[0]
                
                # 첫 번째 배치에서 디버깅 정보 출력
                if batch_idx == 0:
                    logger.info(f"First batch debug info:")
                    logger.info(f"  Pixel values shape: {pixel_values.shape}")
                    logger.info(f"  Batch keys: {list(batch.keys())}")
                    logger.info(f"  Device: {pixel_values.device}")
                    logger.info(f"  Model device: {next(model.parameters()).device}")
                
                # 5번째 배치마다 진행 상황 로그
                if batch_idx % 5 == 0:
                    progress_pct = (batch_idx / len(dataloader)) * 100
                    elapsed = time.time() - start_time
                    estimated_total = elapsed / (batch_idx + 1) * len(dataloader)
                    remaining = estimated_total - elapsed
                    logger.info(f"Processing batch {batch_idx}/{len(dataloader)} ({progress_pct:.1f}%) - "
                              f"Elapsed: {elapsed/60:.1f}min, Remaining: {remaining/60:.1f}min")
                
                # Ground truth 추출
                gt_texts = []
                if "text" in batch:
                    gt_texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
                elif "reference" in batch:
                    # labels가 텐서인 경우 디코딩
                    labels = batch["reference"]
                    if torch.is_tensor(labels):
                        # datamodule의 tokenizer 사용해서 디코딩
                        try:
                            # -100 토큰 제거 (loss 마스킹된 부분)
                            labels_for_decode = labels.clone()
                            labels_for_decode[labels_for_decode == -100] = datamodule.tokenizer.pad_token_id
                            gt_texts = datamodule.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
                            
                            # 전체 텍스트 사용 (자르지 않음)
                            processed_gt_texts = []
                            for gt_text in gt_texts:
                                # Assistant 부분이 있으면 추출하되, 없으면 전체 텍스트 사용
                                if "Assistant:" in gt_text:
                                    # Assistant: 이후 부분만 추출 (기존 동작 유지하되 로그 추가)
                                    assistant_part = gt_text.split("Assistant:")[-1].strip()
                                    full_text = gt_text.strip()
                                    
                                    # 디버깅을 위해 잘린 텍스트와 전체 텍스트 길이 비교
                                    if len(processed_gt_texts) < 2:  # 처음 2개만 로그
                                        logger.debug(f"Reference processing - Full: {len(full_text)} chars, Assistant only: {len(assistant_part)} chars")
                                    
                                    # 전체 텍스트 사용 (Assistant 부분만 자르지 않음)
                                    processed_gt_texts.append(full_text)
                                else:
                                    processed_gt_texts.append(gt_text.strip())
                            gt_texts = processed_gt_texts
                            
                            logger.debug(f"Decoded GT texts (샘플 길이): {[len(gt) for gt in gt_texts[:2]]}")  # 길이 정보 포함
                            
                        except Exception as e:
                            logger.error(f"Failed to decode labels: {e}")
                            gt_texts = [f"decode_error_{i}" for i in range(batch_size)]
                    else:
                        gt_texts = labels if isinstance(labels, list) else [labels]
                elif "input_text" in batch:
                    # input_text에서 전체 텍스트 사용 (Assistant 부분만 자르지 않음)
                    input_texts = batch["input_text"] if isinstance(batch["input_text"], list) else [batch["input_text"]]
                    gt_texts = []
                    for text in input_texts:
                        if "Assistant:" in text:
                            # 전체 텍스트 사용하되 Assistant 부분 정보는 로그에 기록
                            assistant_part = text.split("Assistant:")[-1].strip()
                            full_text = text.strip()
                            
                            if len(gt_texts) < 2:  # 처음 2개만 로그
                                logger.debug(f"Input text processing - Full: {len(full_text)} chars, Assistant only: {len(assistant_part)} chars")
                            
                            # 전체 텍스트 사용
                            gt_texts.append(full_text)
                        else:
                            gt_texts.append(text.strip() if text else "")
                else:
                    gt_texts = [f"no_gt_{i}" for i in range(batch_size)]
                
                # 이미지 경로 추출
                image_paths = batch.get("image_path", [f"batch_{batch_idx}_sample_{i}" for i in range(batch_size)])
                
                # 모델 추론 (generate 모드)
                # PanoramaVLM의 generate 메서드 직접 호출
                try:
                    # 첫 번째 배치에서 모델 상태 확인
                    if batch_idx == 0:
                        logger.info(f"Model generate method available: {hasattr(model.model, 'generate')}")
                        logger.info(f"Model training mode: {model.training}")
                        logger.info(f"Generation params: {generation_params}")
                        
                        # 토크나이저 상태 확인
                        if hasattr(datamodule, 'tokenizer') and datamodule.tokenizer:
                            logger.info(f"Tokenizer EOS token ID: {datamodule.tokenizer.eos_token_id}")
                            logger.info(f"Tokenizer PAD token ID: {datamodule.tokenizer.pad_token_id}")
                        else:
                            logger.warning("Tokenizer not available for generation parameters")
                    
                    # input_ids는 None으로 설정 (이미지만으로 생성)
                    # 유효한 generation 파라미터만 사용
                    gen_kwargs = {
                        "pixel_values": pixel_values,
                        "input_ids": None,
                        "max_new_tokens": generation_params.get("max_new_tokens", 128),
                        "temperature": generation_params.get("temperature", 0.7),
                        "do_sample": True,  # temperature 사용 시 필요
                    }
                    
                    # 토크나이저 사용 가능한 경우에만 패딩 토큰 설정
                    if hasattr(datamodule, 'tokenizer') and datamodule.tokenizer:
                        if datamodule.tokenizer.eos_token_id is not None:
                            gen_kwargs["pad_token_id"] = datamodule.tokenizer.eos_token_id
                        elif datamodule.tokenizer.pad_token_id is not None:
                            gen_kwargs["pad_token_id"] = datamodule.tokenizer.pad_token_id
                    
                    # early_stopping 등 불필요한 파라미터 제거
                    # (transformers 경고 방지)
                    
                    output = model.model.generate(**gen_kwargs)
                    
                    # 출력 검증 및 처리
                    if isinstance(output, dict) and "text" in output:
                        batch_predictions = output["text"]
                        if batch_idx == 0:
                            logger.info(f"Successfully generated text output: {len(batch_predictions)} predictions")
                    elif isinstance(output, torch.Tensor):
                        logger.warning(f"Received tensor output instead of text dict: {output.shape}")
                        batch_predictions = []
                    else:
                        logger.error(f"Unexpected output format: {type(output)}")
                        if hasattr(output, '__dict__'):
                            logger.error(f"Output attributes: {list(output.__dict__.keys())}")
                        batch_predictions = []
                        
                    
                    if batch_predictions and batch_idx % 10 == 0:  # 매 10번째 배치마다 샘플 출력
                        for i, pred in enumerate(batch_predictions[:2]):  # 처음 2개만
                            logger.info(f"  Sample {i}: '{pred}' (length: {len(pred)})")
                    
                    logger.debug(f"Generated {len(batch_predictions)} predictions for batch {batch_idx}")
                    
                except Exception as e:
                    logger.error(f"Generation failed for batch {batch_idx}: {e}")
                    logger.error(f"Error details: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # 빈 예측으로 폴백
                    batch_predictions = [""] * batch_size
                
                # 배치 크기 검증
                if not isinstance(batch_predictions, list):
                    batch_predictions = [batch_predictions]
                
                # 배치 크기가 일치하지 않으면 로그 출력
                if len(batch_predictions) != batch_size:
                    logger.warning(f"Batch size mismatch: expected {batch_size}, got {len(batch_predictions)}")
                
                # 결과 저장
                for i, pred in enumerate(batch_predictions):
                    predictions.append(pred)
                    
                    if i < len(gt_texts):
                        references.append(gt_texts[i])
                    else:
                        references.append("")
                    
                    metadata.append({
                        'stage': stage,
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'image_path': image_paths[i] if i < len(image_paths) else "",
                        'prediction_length': len(pred.split()),
                        'reference_length': len(gt_texts[i].split()) if i < len(gt_texts) else 0
                    })
                
                # 메모리 정리
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    error_count += 1
                    logger.error(f"CUDA OOM in batch {batch_idx} for {stage} model. Batch size: {batch_size}")
                    torch.cuda.empty_cache()
                    
                    # 배치 크기가 1보다 크면 개별 처리 시도
                    if batch_size > 1:
                        logger.info(f"Attempting individual processing for batch {batch_idx}")
                        try:
                            for sample_idx in range(batch_size):
                                sample_pixel_values = pixel_values[sample_idx:sample_idx+1]
                                sample_params = {
                                    "pixel_values": sample_pixel_values,
                                    "input_ids": None,
                                    "max_new_tokens": generation_params.get("max_new_tokens", 128),
                                    "temperature": generation_params.get("temperature", 0.7),
                                    "do_sample": True,
                                }
                                
                                # 토크나이저 사용 가능한 경우에만 패딩 토큰 설정
                                if hasattr(datamodule, 'tokenizer') and datamodule.tokenizer:
                                    if datamodule.tokenizer.eos_token_id is not None:
                                        sample_params["pad_token_id"] = datamodule.tokenizer.eos_token_id
                                    elif datamodule.tokenizer.pad_token_id is not None:
                                        sample_params["pad_token_id"] = datamodule.tokenizer.pad_token_id
                                
                                sample_output = model.model.generate(**sample_params)
                                sample_pred = sample_output["text"][0] if isinstance(sample_output, dict) else sample_output[0]
                                
                                predictions.append(sample_pred)
                                
                                if sample_idx < len(gt_texts):
                                    references.append(gt_texts[sample_idx])
                                else:
                                    references.append("")
                                
                                metadata.append({
                                    'stage': stage,
                                    'batch_idx': batch_idx,
                                    'sample_idx': sample_idx,
                                    'image_path': image_paths[sample_idx] if sample_idx < len(image_paths) else "",
                                    'prediction_length': len(sample_pred.split()),
                                    'reference_length': len(gt_texts[sample_idx].split()) if sample_idx < len(gt_texts) else 0,
                                    'individual_processing': True
                                })
                            
                            logger.info(f"✓ Individual processing successful for batch {batch_idx}")
                        except Exception as individual_e:
                            logger.error(f"Individual processing also failed for batch {batch_idx}: {individual_e}")
                    continue
                else:
                    raise
            except Exception as e:
                error_count += 1
                logger.error(f"Error in batch {batch_idx} for {stage} model: {e}")
                continue
    
    elapsed_time = time.time() - start_time
    logger.info(f"{stage.capitalize()} evaluation completed in {elapsed_time/60:.1f} minutes")
    logger.info(f"Total samples: {len(predictions)}, Errors: {error_count}")
    
    return predictions, references, metadata, error_count

def main():
    parser = argparse.ArgumentParser(description="PanoLLaVA Simple Evaluation")
    parser.add_argument('--ckpt', required=True, help='Model checkpoint')
    parser.add_argument('--lora-weights-path', help='LoRA weights directory (optional)')
    parser.add_argument('--csv-input', required=True, help='입력 CSV 파일 (예: test.csv)')
    parser.add_argument('--vision-name', default='google/siglip-base-patch16-224')
    parser.add_argument('--lm-name', default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--resampler', default='mlp')
    parser.add_argument('--crop-strategy', default='e2p', choices=['sliding_window', 'e2p', 'cubemap', 'resize', 'anyres', 'anyres_max'])
    parser.add_argument('--max-text-length', type=int, default=1024, 
                        help='Maximum text sequence length for tokenization (default: 256)')
    parser.add_argument('--max-new-tokens', type=int, default=128, 
                        help='Maximum new tokens to generate (default: 128)')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    # 1. 경로/파일명 세팅
    eval_dir = Path('eval_results')
    eval_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%y%m%d_%H%M')
    csv_out = eval_dir / f"eval_{timestamp}.csv"
    json_out = eval_dir / f"result_{timestamp}.json"

    # 2. 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {
        "vision_name": args.vision_name,
        "lm_name": args.lm_name,
        "resampler": args.resampler,
        "lr": 1e-5,
        "max_text_length": args.max_text_length
    }
    model = load_model(args.ckpt, 'finetune', device, lora_weights_path=args.lora_weights_path, **model_kwargs)
    model.eval()
    model = model.to(device)

    # 3. 데이터셋 준비 (입력 CSV 사용)
    datamodule = VLMDataModule(
        csv_train=args.csv_input,
        csv_val=args.csv_input,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer_name=args.lm_name,
        max_text_length=args.max_text_length,
        crop_strategy=args.crop_strategy,
        eval_mode=True
    )
    datamodule.setup()
    dataloader = datamodule.val_dataloader()

    # 4. 예측 수행
    predictions, references, metadata, error_count = evaluate_model(
        model, dataloader, 'finetune', device,
        {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature},
        datamodule
    )

    # 5. CSV 저장
    sample_data = []
    for i, (pred, ref, meta) in enumerate(zip(predictions, references, metadata)):
        sample_data.append({
            'sample_id': i,
            'stage': 'finetune',
            'prediction': pred,
            'reference': ref,
            'prediction_length': len(pred.split()),
            'reference_length': len(ref.split()),
            'is_empty_prediction': not pred.strip(),
            'image_path': meta.get('image_path', ''),
            'batch_idx': meta.get('batch_idx', ''),
            'individual_processing': meta.get('individual_processing', False)
        })
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_out, index=False, encoding="utf-8")
    print(f"✓ 예측 결과 CSV 저장: {csv_out}")

    # 6. 지표 계산 및 JSON 저장
    valid = df[(df["prediction"].str.strip() != "") & (df["reference"].str.strip() != "")]
    if len(valid) == 0:
        print("⚠️ prediction과 reference가 모두 있는 샘플이 없습니다. 지표 계산을 건너뜁니다.")
        return
    refs = valid["reference"].tolist()
    preds = valid["prediction"].tolist()
    
    # 메트릭 계산
    metrics = {}
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        ref_tokens = [[ref.split()] for ref in refs]
        pred_tokens = [pred.split() for pred in preds]
        smoothing = SmoothingFunction().method1
        metrics['bleu1'] = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        metrics['bleu2'] = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        metrics['bleu3'] = corpus_bleu(ref_tokens, pred_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing)
        metrics['bleu4'] = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    except Exception as e:
        print(f"BLEU 계산 오류: {e}")
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        for ref, pred in zip(refs, preds):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        metrics['rouge1'] = float(np.mean(rouge1_scores))
        metrics['rouge2'] = float(np.mean(rouge2_scores))
        metrics['rougeL'] = float(np.mean(rougeL_scores))
    except Exception as e:
        print(f"ROUGE 계산 오류: {e}")
    # METEOR 점수 계산 (수정됨)
    try:
        import nltk
        # NLTK 데이터 다운로드 (필요시)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("다운로딩 NLTK WordNet...")
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        
        from nltk.translate.meteor_score import meteor_score
        meteor_scores = []
        for ref, pred in zip(refs, preds):
            try:
                # METEOR는 단어 단위로 계산됨
                ref_tokens = ref.split()
                pred_tokens = pred.split()
                if len(ref_tokens) > 0 and len(pred_tokens) > 0:
                    score = meteor_score([ref_tokens], pred_tokens)
                    meteor_scores.append(score)
                else:
                    meteor_scores.append(0.0)
            except Exception as e:
                print(f"METEOR 개별 계산 오류: {e}")
                meteor_scores.append(0.0)
        
        if meteor_scores:
            metrics['meteor'] = float(np.mean(meteor_scores))
            print(f"METEOR 점수 계산 완료: {metrics['meteor']:.4f} (총 {len(meteor_scores)}개 샘플)")
        else:
            metrics['meteor'] = 0.0
            print("METEOR 점수 계산 실패: 유효한 점수가 없음")
            
    except Exception as e:
        print(f"METEOR 계산 오류: {e}")
        metrics['meteor'] = 0.0
    
    # 기본 통계 (텍스트 처리 분석 포함)
    ref_lengths = [len(ref.split()) for ref in refs]
    pred_lengths = [len(pred.split()) for pred in preds]
    
    # 긴 텍스트 분석
    long_refs = [ref for ref in refs if len(ref.split()) > 50]
    long_preds = [pred for pred in preds if len(pred.split()) > 50]
    
    metrics['avg_ref_length'] = float(np.mean(ref_lengths))
    metrics['avg_pred_length'] = float(np.mean(pred_lengths))
    metrics['max_ref_length'] = float(max(ref_lengths)) if ref_lengths else 0
    metrics['max_pred_length'] = float(max(pred_lengths)) if pred_lengths else 0
    metrics['long_refs_count'] = len(long_refs)
    metrics['long_preds_count'] = len(long_preds)
    metrics['long_refs_ratio'] = len(long_refs) / len(refs) if refs else 0
    metrics['long_preds_ratio'] = len(long_preds) / len(preds) if preds else 0
    metrics['length_ratio'] = float(np.mean(pred_lengths) / np.mean(ref_lengths)) if np.mean(ref_lengths) > 0 else 0
    metrics['empty_predictions'] = float(sum(1 for pred in preds if not pred.strip()) / len(preds))
    
    # SPICE 점수 계산 (pycocoevalcap 사용)
    try:
        from pycocoevalcap.spice.spice import Spice
        spice_scorer = Spice()
        
        # SPICE는 딕셔너리 형태의 입력을 요구함
        gts = {str(i): [ref] for i, ref in enumerate(refs)}
        res = {str(i): [pred] for i, pred in enumerate(preds)}
        
        spice_score, _ = spice_scorer.compute_score(gts, res)
        metrics['spice'] = float(spice_score)
        print(f"SPICE 점수 계산 완료: {metrics['spice']:.4f}")
        
    except ImportError:
        print("SPICE 계산을 위해 pycocoevalcap을 설치하세요: pip install pycocoevalcap")
        metrics['spice'] = 0.0
    except Exception as e:
        print(f"SPICE 계산 오류: {e}")
        metrics['spice'] = 0.0
    
    # CLIP-based 메트릭들 계산 (CLIP Score, CLIP-S, RefCLIP-S)
    try:
        import clip
        from PIL import Image
        device_clip = "cuda" if torch.cuda.is_available() else "cpu"
        model_clip, preprocess = clip.load("ViT-B/32", device=device_clip)
        
        clip_scores = []
        clip_s_scores = []  # CLIP-S (Image-Text Cosine Similarity)
        ref_clip_s_scores = []  # RefCLIP-S (Reference-Prediction Similarity)
        
        # 텍스트 임베딩 배치 처리를 위한 준비
        pred_texts = []
        ref_texts = []
        valid_indices = []
        
        for idx, row in valid.iterrows():
            img_path = row['image_path']
            if img_path and os.path.exists(img_path):
                pred_texts.append(row['prediction'])
                ref_texts.append(row['reference'])
                valid_indices.append(idx)
        
        if pred_texts:
            print(f"CLIP 기반 메트릭 계산 중... ({len(pred_texts)}개 샘플)")
            
            # 배치 처리를 위해 텍스트들을 청크로 나누기
            chunk_size = 32  # GPU 메모리에 따라 조정
            
            for chunk_start in range(0, len(pred_texts), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(pred_texts))
                chunk_pred_texts = pred_texts[chunk_start:chunk_end]
                chunk_ref_texts = ref_texts[chunk_start:chunk_end]
                chunk_indices = valid_indices[chunk_start:chunk_end]
                
                # 텍스트 임베딩 계산 (긴 텍스트 처리 개선)
                try:
                    # 긴 텍스트를 위한 전처리 함수
                    def preprocess_text_for_clip(text, max_length=75):
                        """CLIP 토큰 제한을 고려한 텍스트 전처리"""
                        # 너무 긴 텍스트는 중요한 부분만 유지
                        words = text.split()
                        if len(words) > max_length:
                            # 앞쪽 절반과 뒤쪽 절반을 유지 (중간 생략)
                            half_length = max_length // 2
                            truncated = words[:half_length] + ["..."] + words[-half_length:]
                            return " ".join(truncated)
                        return text
                    
                    # 텍스트 전처리
                    processed_pred_texts = [preprocess_text_for_clip(text) for text in chunk_pred_texts]
                    processed_ref_texts = [preprocess_text_for_clip(text) for text in chunk_ref_texts]
                    
                    # 토큰화 시 긴 텍스트 처리 개선
                    pred_tokens = clip.tokenize(processed_pred_texts, truncate=True).to(device_clip)
                    ref_tokens = clip.tokenize(processed_ref_texts, truncate=True).to(device_clip)
                    
                    # 원본 텍스트 길이 로깅 (첫 번째 청크에서만)
                    if chunk_start == 0:
                        for i in range(min(2, len(chunk_pred_texts))):
                            orig_pred_len = len(chunk_pred_texts[i])
                            orig_ref_len = len(chunk_ref_texts[i])
                            proc_pred_len = len(processed_pred_texts[i])
                            proc_ref_len = len(processed_ref_texts[i])
                            logger.debug(f"CLIP text processing {i}: Pred {orig_pred_len}→{proc_pred_len} chars, Ref {orig_ref_len}→{proc_ref_len} chars")
                    
                    with torch.no_grad():
                        pred_text_features = model_clip.encode_text(pred_tokens)
                        ref_text_features = model_clip.encode_text(ref_tokens)
                        
                        # 정규화
                        pred_text_features = pred_text_features / pred_text_features.norm(dim=-1, keepdim=True)
                        ref_text_features = ref_text_features / ref_text_features.norm(dim=-1, keepdim=True)
                        
                        # RefCLIP-S: Reference와 Prediction 간의 유사도
                        ref_clip_s_batch = torch.diagonal(pred_text_features @ ref_text_features.T).cpu().numpy()
                        ref_clip_s_scores.extend(ref_clip_s_batch.tolist())
                        
                        # 각 이미지에 대해 CLIP Score와 CLIP-S 계산
                        for i, (pred_feat, idx) in enumerate(zip(pred_text_features, chunk_indices)):
                            try:
                                row = valid.iloc[valid_indices.index(idx)]
                                img_path = row['image_path']
                                
                                if os.path.exists(img_path):
                                    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device_clip)
                                    image_features = model_clip.encode_image(image)
                                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                                    
                                    # CLIP Score (Image-Prediction similarity)
                                    clip_score = (image_features @ pred_feat.unsqueeze(0).T).item()
                                    clip_scores.append(clip_score)
                                    
                                    # CLIP-S (동일함 - Image-Text Cosine Similarity)
                                    clip_s_scores.append(clip_score)
                                    
                            except Exception as e:
                                print(f"이미지 처리 오류 (인덱스 {idx}): {e}")
                                continue
                                
                except Exception as e:
                    print(f"텍스트 임베딩 처리 오류: {e}")
                    continue
            
            # 메트릭 저장
            if clip_scores:
                metrics['clip_score'] = float(np.mean(clip_scores))
                metrics['clip_score_std'] = float(np.std(clip_scores))
                print(f"CLIP Score 계산 완료: {metrics['clip_score']:.4f} ± {metrics['clip_score_std']:.4f}")
            
            if clip_s_scores:
                metrics['clip_s'] = float(np.mean(clip_s_scores))
                metrics['clip_s_std'] = float(np.std(clip_s_scores))
                print(f"CLIP-S 계산 완료: {metrics['clip_s']:.4f} ± {metrics['clip_s_std']:.4f}")
            
            if ref_clip_s_scores:
                metrics['ref_clip_s'] = float(np.mean(ref_clip_s_scores))
                metrics['ref_clip_s_std'] = float(np.std(ref_clip_s_scores))
                print(f"RefCLIP-S 계산 완료: {metrics['ref_clip_s']:.4f} ± {metrics['ref_clip_s_std']:.4f}")
        
        else:
            print("유효한 이미지 경로가 없어 CLIP 기반 메트릭을 계산할 수 없습니다.")
            
    except ImportError:
        print("CLIP 메트릭 계산을 위해 CLIP을 설치하세요: pip install git+https://github.com/openai/CLIP.git")
    except Exception as e:
        print(f"CLIP 기반 메트릭 계산 오류: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ 평가 메트릭 JSON 저장: {json_out}")
    
    # 메트릭 결과 요약 출력
    print("\n" + "="*60)
    print("📊 평가 메트릭 요약")
    print("="*60)
    
    # 텍스트 유사도 메트릭
    print("🔤 텍스트 유사도 메트릭:")
    if 'bleu4' in metrics:
        print(f"  BLEU-4:     {metrics['bleu4']:.4f}")
    if 'rouge1' in metrics:
        print(f"  ROUGE-1:    {metrics['rouge1']:.4f}")
    if 'rougeL' in metrics:
        print(f"  ROUGE-L:    {metrics['rougeL']:.4f}")
    if 'meteor' in metrics and metrics['meteor'] > 0:
        print(f"  METEOR:     {metrics['meteor']:.4f}")
    if 'spice' in metrics and metrics['spice'] > 0:
        print(f"  SPICE:      {metrics['spice']:.4f}")
    
    # 멀티모달 메트릭
    print("\n🖼️  멀티모달 메트릭:")
    if 'clip_score' in metrics:
        print(f"  CLIP Score: {metrics['clip_score']:.4f} ± {metrics.get('clip_score_std', 0):.4f}")
    if 'clip_s' in metrics:
        print(f"  CLIP-S:     {metrics['clip_s']:.4f} ± {metrics.get('clip_s_std', 0):.4f}")
    if 'ref_clip_s' in metrics:
        print(f"  RefCLIP-S:  {metrics['ref_clip_s']:.4f} ± {metrics.get('ref_clip_s_std', 0):.4f}")
    
    # 기본 통계
    print("\n📈 기본 통계:")
    print(f"  평균 예측 길이:     {metrics['avg_pred_length']:.1f} 단어")
    print(f"  평균 참조 길이:     {metrics['avg_ref_length']:.1f} 단어")
    print(f"  최대 예측 길이:     {metrics['max_pred_length']:.0f} 단어")
    print(f"  최대 참조 길이:     {metrics['max_ref_length']:.0f} 단어")
    print(f"  긴 텍스트 비율:     예측 {metrics['long_preds_ratio']:.1%}, 참조 {metrics['long_refs_ratio']:.1%}")
    print(f"  길이 비율:         {metrics['length_ratio']:.2f}")
    print(f"  빈 예측 비율:      {metrics['empty_predictions']:.2%}")
    print(f"  총 평가 샘플:      {len(valid)} / {len(df)}")
    
    # 텍스트 처리 품질 안내
    if metrics['long_refs_ratio'] > 0.1 or metrics['long_preds_ratio'] > 0.1:
        print(f"\n💡 긴 텍스트 감지됨 (50단어 이상): 전체 텍스트가 평가에 사용되었습니다.")
    
    print("="*60)


if __name__ == '__main__':
    main()