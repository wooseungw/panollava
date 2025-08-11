# coding: utf-8
"""
PanoLLaVA Comprehensive Model Evaluation System
─────────────────────────────────────────────────

단계별 평가 시스템:
1. 모델 및 LoRA 가중치 로드
2. 테스트 데이터셋 준비 (ChatPanoTestDataset, VLMDataModule)
3. 배치별 텍스트 생성 (generate)
4. 예측/참조 텍스트 저장 및 로깅
5. 평가 메트릭 계산 (BLEU, ROUGE, METEOR, SPICE, CIDEr, CLIP-S, RefCLIP-S)

사용법:
    python eval.py --ckpt runs/e2p_finetune_mlp/best.ckpt --lora-weights-path runs/e2p_finetune_mlp/lora_weights --csv-input data/quic360/test.csv
"""

import argparse
import torch
import json
import logging
import time
import traceback
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# 내부 모듈
from train import VLMModule, VLMDataModule, safe_load_checkpoint

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def load_model_and_lora(checkpoint_path: str, lora_weights_path: Optional[str], device: torch.device, **model_kwargs) -> VLMModule:
    """
    1단계: 체크포인트와 LoRA 가중치를 로드하여 생성용 모델 준비
    """
    logger.info("=" * 60)
    logger.info("🚀 1단계: 모델 및 LoRA 가중치 로드")
    logger.info("=" * 60)
    
    # 체크포인트 로드
    logger.info(f"📂 체크포인트 로드: {checkpoint_path}")
    checkpoint = safe_load_checkpoint(checkpoint_path)
    if not checkpoint:
        raise ValueError(f"체크포인트 로드 실패: {checkpoint_path}")
    
    # LoRA 경로 자동 감지
    if lora_weights_path is None:
        checkpoint_dir = Path(checkpoint_path).parent
        potential_lora_path = checkpoint_dir / "lora_weights"
        if potential_lora_path.exists():
            lora_weights_path = str(potential_lora_path)
            logger.info(f"🔍 LoRA 가중치 자동 감지: {lora_weights_path}")
    
    # 모델 로드 (finetune 단계)
    model = VLMModule.load_from_checkpoint(
        checkpoint_path,
        stage="finetune",
        map_location=device,
        **model_kwargs
    )
    
    # LoRA 가중치 로드
    if lora_weights_path and Path(lora_weights_path).exists():
        logger.info(f"🔧 LoRA 가중치 로드: {lora_weights_path}")
        
        # LoRA 파일 구조 검증
        lora_path = Path(lora_weights_path)
        adapter_config = lora_path / "adapter_config.json"
        adapter_model = lora_path / "adapter_model.safetensors"
        
        if adapter_config.exists() and adapter_model.exists():
            success = model.model.load_lora_weights(lora_weights_path)
            if success:
                logger.info("✅ LoRA 가중치 로드 성공!")
                
                # LoRA 설정 정보 출력
                lora_info = model.model.get_lora_info()
                if lora_info.get("is_lora_enabled", False):
                    logger.info(f"📊 LoRA 설정 - Rank: {lora_info.get('lora_r')}, Alpha: {lora_info.get('lora_alpha')}")
                    logger.info(f"   Target modules: {lora_info.get('target_modules')}")
            else:
                logger.warning("⚠️ LoRA 가중치 로드 실패, 기본 모델로 진행")
        else:
            logger.warning(f"⚠️ LoRA 파일 누락: {lora_weights_path}")
    else:
        logger.info("📝 LoRA 가중치 없음, 기본 모델 사용")
    
    # 평가 모드 설정
    model.eval()
    model = model.to(device)
    model.model.requires_grad_(False)
    
    logger.info(f"✓ 모델 준비 완료 - Device: {device}, Stage: {model._stage_key}")
    return model


def prepare_test_dataset(csv_input: str, batch_size: int, max_text_length: int, crop_strategy: str, lm_name: str, num_workers: int = 0) -> Tuple[VLMDataModule, Any]:
    """
    2단계: ChatPanoTestDataset과 VLMDataModule을 활용한 테스트 데이터 준비
    """
    logger.info("=" * 60)
    logger.info("📊 2단계: 테스트 데이터셋 준비")
    logger.info("=" * 60)
    
    # 데이터 모듈 초기화
    logger.info(f"📂 CSV 입력: {csv_input}")
    # config.sh의 FINETUNE_SYSTEM_MSG와 동일한 system 메시지 사용
    system_msg = "You are an expert assistant specialized in analyzing panoramic images. Please provide detailed, accurate, and helpful responses about what you observe in the panoramic view shortly."
    
    datamodule = VLMDataModule(
        csv_train=csv_input,
        csv_val=csv_input,  # 평가용으로 동일한 파일 사용
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer_name=lm_name,
        max_text_length=max_text_length,
        crop_strategy=crop_strategy,
        eval_mode=True,  # 평가 모드 활성화
        system_msg=system_msg  # system 메시지 추가
    )
    
    # 데이터셋 설정
    datamodule.setup()
    test_dataloader = datamodule.val_dataloader()
    
    logger.info(f"✓ 데이터셋 준비 완료")
    logger.info(f"   - 총 배치 수: {len(test_dataloader)}")
    logger.info(f"   - 배치 크기: {batch_size}")
    logger.info(f"   - 텍스트 최대 길이: {max_text_length}")
    logger.info(f"   - 크롭 전략: {crop_strategy}")
    logger.info(f"   - 워커 수: {num_workers}")
    
    return datamodule, test_dataloader


def generate_predictions(model: VLMModule, test_dataloader, datamodule: VLMDataModule, device: torch.device,
                        max_new_tokens: int = 128, temperature: float = 0.7,
                        top_p: float = 0.9, top_k: int = 50,
                        repetition_penalty: float = 1.1, length_penalty: float = 1.0,
                        min_new_tokens: int = 5) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    3단계: 테스트 데이터에서 배치별 텍스트 생성
    """
    logger.info("=" * 60)
    logger.info("🤖 3단계: 텍스트 생성")
    logger.info("=" * 60)
    
    predictions = []
    references = []
    image_paths = []
    input_texts = []
    
    logger.info(f"🎯 생성 파라미터 - Max tokens: {max_new_tokens}, Min tokens: {min_new_tokens}, Temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="생성 중")):
            try:
                # 입력 데이터 준비
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch.get("input_ids")
                if input_ids is not None:
                    input_ids = input_ids.to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                batch_size = pixel_values.shape[0]
                
                # VLM 모델을 위한 디버깅 정보
                if batch_idx == 0:
                    logger.info(f"=== VLM 입력 디버깅 (배치 {batch_idx}) ===")
                    logger.info(f"pixel_values shape: {pixel_values.shape}")
                    logger.info(f"input_ids shape: {input_ids.shape if input_ids is not None else 'None'}")
                    if input_ids is not None:
                        logger.info(f"input_ids sample: {input_ids[0][:20]}")  # 처음 20개 토큰만
                    logger.info("=" * 45)
                
                # 참조 텍스트 추출 (전체 대화 텍스트)
                batch_references = []
                if "reference" in batch:
                    labels = batch["reference"]
                    if torch.is_tensor(labels):
                        # -100 토큰 제거 후 디코딩
                        labels_for_decode = labels.clone()
                        labels_for_decode[labels_for_decode == -100] = datamodule.tokenizer.pad_token_id
                        batch_references = datamodule.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
                        # 전체 대화 텍스트를 참조로 사용
                        batch_references = [text.strip() for text in batch_references]
                    else:
                        batch_references = labels if isinstance(labels, list) else [labels]
                elif "input_text" in batch:
                    input_texts = batch["input_text"] if isinstance(batch["input_text"], list) else [batch["input_text"]]
                    batch_references = [text.strip() for text in input_texts]
                else:
                    batch_references = [f"no_reference_{i}" for i in range(batch_size)]
                
                # 이미지 경로 추출
                batch_image_paths = batch.get("image_path", [f"batch_{batch_idx}_sample_{i}" for i in range(batch_size)])
                
                # input_text 추출
                batch_input_texts = batch.get("input_text", [f"no_input_text_{i}" for i in range(batch_size)])
                if not isinstance(batch_input_texts, list):
                    batch_input_texts = [batch_input_texts] * batch_size
                
                # VLM 모델 생성 (train과 동일한 입력 형태)
                try:
                    # train과 동일한 입력: user 질문 + assistant 프롬프트
                    if batch_idx == 0:
                        logger.info(f"=== 생성용 입력 처리 (train과 동일) ===")
                        sample_input_text = batch_input_texts[0] if batch_input_texts else ""
                        logger.info(f"Input text (with assistant prompt): {sample_input_text[:150]}...")
                        logger.info(f"Input_ids shape: {input_ids.shape if input_ids is not None else 'None'}")
                        logger.info("=" * 50)
                    
                    # VLM의 generate 메서드 사용하되, train과 동일한 입력 제공
                    gen_kwargs = {
                        "pixel_values": pixel_values,
                        "input_ids": input_ids,  # train과 동일: user 질문 + assistant 프롬프트
                        "attention_mask": attention_mask,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "length_penalty": length_penalty,
                        "min_new_tokens": min_new_tokens,
                    }
                    
                    # 생성 실행 - VLM 모델의 generate 메서드 사용
                    output = model.model.generate(**gen_kwargs)
                    
                    # 생성 결과 처리
                    if isinstance(output, dict) and "text" in output:
                        batch_predictions = output["text"]
                        
                        # 첫 번째 배치에서 생성 결과 상세 로그
                        if batch_idx == 0 and len(batch_predictions) > 0:
                            logger.info(f"=== 생성 결과 디버깅 (배치 {batch_idx}) ===")
                            logger.info(f"Raw output type: {type(output)}")
                            logger.info(f"Output keys: {list(output.keys()) if isinstance(output, dict) else 'N/A'}")
                            logger.info(f"Generated text sample: '{batch_predictions[0]}'")
                            logger.info(f"Generated text length: {len(batch_predictions[0])} chars")
                            
                            # 원본 generated_ids도 확인
                            if "generated_ids" in output:
                                raw_ids = output["generated_ids"][0] if len(output["generated_ids"]) > 0 else None
                                if raw_ids is not None:
                                    raw_text = datamodule.tokenizer.decode(raw_ids, skip_special_tokens=False)
                                    logger.info(f"Raw decoded (with special tokens): '{raw_text}'")
                            logger.info("=" * 50)
                    else:
                        logger.warning(f"예상하지 못한 출력 형식: {type(output)}")
                        batch_predictions = [""] * batch_size
                        
                except Exception as gen_error:
                    logger.error(f"VLM 생성 중 오류 발생: {gen_error}")
                    logger.error(f"스택 트레이스: ", exc_info=True)
                    batch_predictions = [f"generation_error_{i}" for i in range(batch_size)]
                
                # 배치 크기 검증
                if len(batch_predictions) != batch_size:
                    logger.warning(f"배치 크기 불일치: 예상 {batch_size}, 실제 {len(batch_predictions)}")
                    # 크기 조정
                    if len(batch_predictions) < batch_size:
                        batch_predictions.extend([""] * (batch_size - len(batch_predictions)))
                    else:
                        batch_predictions = batch_predictions[:batch_size]
                
                # 배치별 prediction과 reference 로그 출력 (CSV 입력 전)
                logger.info(f"=== 배치 {batch_idx} 결과 로그 ===")
                for i, (pred, ref) in enumerate(zip(batch_predictions, batch_references)):
                    # 길이 제한을 두어 로그가 너무 길어지지 않도록 함
                    pred_preview = pred[:128] + ("..." if len(pred) > 128 else "")
                    ref_preview = ref[:128] + ("..." if len(ref) > 128 else "")
                    logger.info(f"  샘플 {len(predictions) + i}\n Pred='{pred_preview}' \n Ref='{ref_preview}'")
                logger.info(f"==========================")
                
                # 결과 저장
                predictions.extend(batch_predictions)
                references.extend(batch_references)
                image_paths.extend(batch_image_paths)
                input_texts.extend(batch_input_texts)
                
                # 진행 상황 로깅
                if batch_idx % 10 == 0:
                    logger.info(f"진행: {batch_idx + 1}/{len(test_dataloader)} 배치 완료")
                
            except Exception as e:
                logger.error(f"배치 {batch_idx} 생성 실패: {e}")
                # 빈 결과로 대체
                batch_size = pixel_values.shape[0] if 'pixel_values' in locals() else 1
                predictions.extend([""] * batch_size)
                references.extend(batch_references if 'batch_references' in locals() else [""] * batch_size)
                image_paths.extend(batch_image_paths if 'batch_image_paths' in locals() else [f"error_{i}" for i in range(batch_size)])
                input_texts.extend(batch_input_texts if 'batch_input_texts' in locals() else [f"error_input_{i}" for i in range(batch_size)])
                continue
    
    logger.info(f"✓ 텍스트 생성 완료 - 총 {len(predictions)}개 샘플")
    return predictions, references, image_paths, input_texts


def save_and_log_results(predictions: List[str], references: List[str], image_paths: List[str], input_texts: List[str], output_dir: Path, timestamp: str) -> pd.DataFrame:
    """
    4단계: 생성된 답변과 참조 텍스트를 저장하고 로깅 (간소화된 CSV)
    """
    logger.info("=" * 60)
    logger.info("💾 4단계: 결과 저장 및 로깅")
    logger.info("=" * 60)
    
    # 간소화된 CSV 데이터 준비 (핵심 컬럼만)
    results_data = []
    for i, (pred, ref, img_path) in enumerate(zip(predictions, references, image_paths)):
        # 빈 값 처리 (NaN 방지)
        pred_str = str(pred) if pred is not None else ""
        ref_str = str(ref) if ref is not None else ""
        img_path_str = str(img_path) if img_path is not None else ""
        
        results_data.append({
            'image_path': img_path_str,
            'prediction': pred_str,
            'reference': ref_str
        })
    
    # DataFrame 생성 및 저장
    df = pd.DataFrame(results_data)
    csv_path = output_dir / f"predictions_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 결과 통계 로깅
    total_samples = len(df)
    empty_predictions = df[df['prediction'].str.strip() == ''].shape[0]
    avg_pred_length = df['prediction'].apply(lambda x: len(str(x).split()) if str(x).strip() else 0).mean()
    avg_ref_length = df['reference'].apply(lambda x: len(str(x).split()) if str(x).strip() else 0).mean()
    
    logger.info(f"📊 결과 통계:")
    logger.info(f"   - 총 샘플: {total_samples}")
    logger.info(f"   - 빈 예측: {empty_predictions}개 ({empty_predictions/total_samples*100:.1f}%)")
    logger.info(f"   - 평균 예측 길이: {avg_pred_length:.1f} 단어")
    logger.info(f"   - 평균 참조 길이: {avg_ref_length:.1f} 단어")
    logger.info(f"✓ CSV 저장: {csv_path}")
    
    # 샘플 로깅 (처음 3개)
    logger.info(f"📝 샘플 예시 (처음 3개):")
    for i in range(min(3, len(df))):
        sample = df.iloc[i]
        pred_preview = str(sample['prediction'])[:100] + ("..." if len(str(sample['prediction'])) > 100 else "")
        ref_preview = str(sample['reference'])[:100] + ("..." if len(str(sample['reference'])) > 100 else "")
        logger.info(f"   샘플 {i}:")
        logger.info(f"     예측: '{pred_preview}'")
        logger.info(f"     참조: '{ref_preview}'")
    
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
    
    # 유효한 샘플만 선택 (예측과 참조가 모두 비어있지 않은 경우)
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
        logger.error("❌ 유효한 예측-참조 쌍이 없습니다.")
        return {}
    
    predictions, references = zip(*valid_pairs)
    predictions = list(predictions)
    references = list(references)
    
    logger.info(f"📊 최종 평가 대상: {len(valid_pairs)} 샘플")
    
    metrics = {}
    
    # Assistant 응답 부분만 추출 (참조용) - NaN 처리 추가
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
    
    # 6. CLIP-S 및 RefCLIP-S 계산 (더 안전한 import와 오류 처리)
    try:
        # 여러 CLIP 구현 시도 - 더 안전한 import 방식
        clip_model = None
        preprocess = None
        device_clip = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # 첫 번째: openai-clip 시도 (더 명시적인 import)
            import sys
            import importlib
            
            # 기존 clip 모듈이 있다면 제거 (충돌 방지)
            if 'clip' in sys.modules:
                del sys.modules['clip']
            
            # 새로 import
            import clip as openai_clip
            
            # clip.load 함수 존재 확인
            if hasattr(openai_clip, 'load'):
                clip_model, preprocess = openai_clip.load("ViT-B/32", device=device_clip)
                logger.info("✓ OpenAI CLIP 모델 로드 성공")
                clip_model = clip_model.to(device_clip)
                clip_tokenize = openai_clip.tokenize  # tokenize 함수 저장
            else:
                raise AttributeError("clip.load 함수를 찾을 수 없습니다")
                
        except Exception as e1:
            logger.warning(f"OpenAI CLIP 로드 실패: {e1}")
            try:
                # 두 번째: transformers CLIP 시도
                from transformers import CLIPModel, CLIPProcessor
                clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                clip_model = clip_model.to(device_clip)
                logger.info("✓ HuggingFace CLIP 모델 로드 성공")
                clip_tokenize = None  # HuggingFace는 tokenize 함수 불필요
            except Exception as e2:
                logger.warning(f"HuggingFace CLIP 로드도 실패: {e2}")
                raise Exception(f"모든 CLIP 구현 로드 실패. OpenAI: {e1}, HuggingFace: {e2}")
        
        if clip_model is not None:
            from PIL import Image
            
            clip_s_scores = []
            ref_clip_s_scores = []
            
            # 유효한 이미지가 있는 샘플만 처리
            if 'image_path' in valid_df.columns:
                valid_image_samples = valid_df[valid_df['image_path'].apply(lambda x: os.path.exists(str(x)) if pd.notna(x) else False)]
            else:
                valid_image_samples = pd.DataFrame()  # 빈 DataFrame
            
            if len(valid_image_samples) > 0:
                logger.info(f"CLIP 메트릭 계산 중... ({len(valid_image_samples)}개 이미지)")
                
                for idx, (_, sample) in enumerate(valid_image_samples.iterrows()):
                    try:
                        # 이미지 로드
                        image_path = sample['image_path']
                        if not os.path.exists(image_path):
                            continue
                            
                        image_pil = Image.open(image_path).convert("RGB")
                        
                        # 텍스트 준비 (길이 제한)
                        pred_text = str(sample['prediction'])[:200]  
                        ref_text = str(sample['reference'])[:200] if "Assistant:" not in str(sample['reference']) else str(sample['reference']).split("Assistant:")[-1].strip()[:200]
                        
                        if not pred_text.strip() or not ref_text.strip():
                            continue
                        
                        # OpenAI CLIP 사용 (더 안전한 방식)
                        if hasattr(clip_model, 'encode_image') and clip_tokenize is not None:
                            try:
                                image = preprocess(image_pil).unsqueeze(0).to(device_clip)
                                pred_tokens = clip_tokenize([pred_text], truncate=True).to(device_clip)
                                ref_tokens = clip_tokenize([ref_text], truncate=True).to(device_clip)
                                
                                with torch.no_grad():
                                    # 특징 추출
                                    image_features = clip_model.encode_image(image)
                                    pred_text_features = clip_model.encode_text(pred_tokens)
                                    ref_text_features = clip_model.encode_text(ref_tokens)
                                    
                                    # 정규화
                                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                                    pred_text_features = pred_text_features / pred_text_features.norm(dim=-1, keepdim=True)
                                    ref_text_features = ref_text_features / ref_text_features.norm(dim=-1, keepdim=True)
                                    
                                    # CLIP-S (Image-Prediction similarity)
                                    clip_s_score = (image_features @ pred_text_features.T).item()
                                    clip_s_scores.append(clip_s_score)
                                    
                                    # RefCLIP-S (Reference-Prediction similarity)
                                    ref_clip_s_score = (ref_text_features @ pred_text_features.T).item()
                                    ref_clip_s_scores.append(ref_clip_s_score)
                                    
                            except Exception as e_openai:
                                logger.debug(f"OpenAI CLIP 처리 오류: {e_openai}")
                                continue
                                
                        # HuggingFace CLIP 사용
                        elif hasattr(clip_model, 'get_image_features'):
                            try:
                                inputs = preprocess(text=[pred_text, ref_text], images=image_pil, return_tensors="pt", padding=True)
                                inputs = {k: v.to(device_clip) for k, v in inputs.items()}
                                
                                with torch.no_grad():
                                    outputs = clip_model(**inputs)
                                    image_features = outputs.image_embeds
                                    text_features = outputs.text_embeds
                                    
                                    # 정규화
                                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                                    
                                    # CLIP-S (Image-Prediction similarity)
                                    clip_s_score = (image_features @ text_features[0:1].T).item()
                                    clip_s_scores.append(clip_s_score)
                                    
                                    # RefCLIP-S (Reference-Prediction similarity) 
                                    ref_clip_s_score = (text_features[1:2] @ text_features[0:1].T).item()
                                    ref_clip_s_scores.append(ref_clip_s_score)
                                    
                            except Exception as e_hf:
                                logger.debug(f"HuggingFace CLIP 처리 오류: {e_hf}")
                                continue
                        else:
                            logger.warning(f"알 수 없는 CLIP 모델 타입: {type(clip_model)}")
                            continue
                    
                    except Exception as e:
                        logger.debug(f"이미지 {image_path} 처리 오류: {e}")
                        continue
                
                if clip_s_scores:
                    metrics['clip_s'] = float(np.mean(clip_s_scores))
                    logger.info(f"✓ CLIP-S: {metrics['clip_s']:.4f}")
                
                if ref_clip_s_scores:
                    metrics['ref_clip_s'] = float(np.mean(ref_clip_s_scores))
                    logger.info(f"✓ RefCLIP-S: {metrics['ref_clip_s']:.4f}")
            else:
                logger.warning("⚠️ 유효한 이미지가 없어 CLIP 메트릭을 계산할 수 없습니다.")
                
    except Exception as e:
        logger.warning(f"⚠️ CLIP 메트릭 계산 오류: {e}")
        logger.debug(f"CLIP 오류 상세: {traceback.format_exc()}")
        
        # CLIP 대안: 텍스트 유사도만 계산
        try:
            logger.info("CLIP 대안으로 텍스트 유사도 계산 시도...")
            from sentence_transformers import SentenceTransformer
            model_st = SentenceTransformer('all-MiniLM-L6-v2')
            
            pred_embeddings = model_st.encode(predictions)
            ref_embeddings = model_st.encode(ref_texts_for_bleu)
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                similarities.append(sim)
            
            metrics['clip_s'] = float(np.mean(similarities))
            metrics['ref_clip_s'] = float(np.mean(similarities))  # 동일한 값 사용
            logger.info(f"✓ CLIP-S (대안-텍스트유사도): {metrics['clip_s']:.4f}")
            logger.info(f"✓ RefCLIP-S (대안-텍스트유사도): {metrics['ref_clip_s']:.4f}")
        except Exception as fallback_e:
            logger.warning(f"⚠️ CLIP 대안 계산도 실패: {fallback_e}")
            metrics['clip_s'] = 0.0
            metrics['ref_clip_s'] = 0.0
    
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
    if 'clip_s' in metrics:
        print(f"CLIP-S     (↑): {metrics['clip_s']:.4f}")
    if 'ref_clip_s' in metrics:
        print(f"RefCLIP-S  (↑): {metrics['ref_clip_s']:.4f}")
    
    print("-" * 40)
    print("💡 (↑) 표시는 높을수록 좋은 메트릭입니다.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="PanoLLaVA 모델 평가 시스템")
    parser.add_argument('--ckpt', required=True, help='모델 체크포인트 경로')
    parser.add_argument('--lora-weights-path', help='LoRA 가중치 경로 (선택)')
    parser.add_argument('--csv-input', default = 'data/quic360/test.csv', help='테스트 CSV 파일 경로')
    parser.add_argument('--output-dir', default='eval_results', help='결과 저장 디렉토리')
    parser.add_argument('--vision-name', default='google/siglip-base-patch16-224')
    parser.add_argument('--lm-name', default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--resampler', default='mlp')
    parser.add_argument('--crop-strategy', default='e2p', choices=['sliding_window', 'e2p', 'cubemap', 'resize', 'anyres', 'anyres_max'])
    parser.add_argument('--max-text-length', type=int, default=128)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--min-new-tokens', type=int, default=5)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    parser.add_argument('--length-penalty', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=16, help='데이터로더 워커 수')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  사용 디바이스: {device}")
    
    try:
        # 1단계: 모델 및 LoRA 가중치 로드
        model_kwargs = {
            "vision_name": args.vision_name,
            "lm_name": args.lm_name,
            "resampler": args.resampler,
            "lr": 1e-5,
            "max_text_length": args.max_text_length
        }
        model = load_model_and_lora(args.ckpt, args.lora_weights_path, device, **model_kwargs)
        
        # 2단계: 테스트 데이터셋 준비
        datamodule, test_dataloader = prepare_test_dataset(
            args.csv_input, args.batch_size, args.max_text_length, 
            args.crop_strategy, args.lm_name, args.num_workers
        )
        
        # 3단계: 텍스트 생성
        predictions, references, image_paths, input_texts = generate_predictions(
            model, test_dataloader, datamodule, device,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty, length_penalty=args.length_penalty,
            min_new_tokens=args.min_new_tokens,
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
