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
    python eval_comprehensive.py --stage finetune --ckpt runs/vlm_finetune/checkpoints/best.ckpt --csv-val data/test.csv
    
    # Resampler 모델 평가  
    python eval_comprehensive.py --stage resampler --ckpt runs/vlm_resampler/checkpoints/best.ckpt --csv-val data/test.csv
    
    # 두 모델 비교 평가
    python eval_comprehensive.py --stage both --finetune-ckpt runs/vlm_finetune/checkpoints/best.ckpt --resampler-ckpt runs/vlm_resampler/checkpoints/best.ckpt --csv-val data/test.csv
"""

import argparse
import torch
import json
import logging
import sys
import time
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

class ComprehensiveMetrics:
    """종합 평가 메트릭 계산 클래스"""
    
    def __init__(self):
        self.available_metrics = []
        self._check_dependencies()
    
    def _check_dependencies(self):
        """평가에 필요한 패키지 확인"""
        # NLTK 확인
        try:
            import nltk
            self.available_metrics.extend(['bleu', 'meteor'])
            logger.info("✓ NLTK available - BLEU, METEOR metrics enabled")
        except ImportError:
            logger.warning("✗ NLTK not available. Install with: pip install nltk")
        
        # ROUGE 확인
        try:
            from rouge_score import rouge_scorer
            self.available_metrics.append('rouge')
            logger.info("✓ rouge-score available - ROUGE metrics enabled")
        except ImportError:
            logger.warning("✗ rouge-score not available. Install with: pip install rouge-score")
        
        # CIDEr 확인
        try:
            from pycocoevalcap.cider.cider import Cider
            self.available_metrics.append('cider')
            logger.info("✓ pycocoevalcap available - CIDEr metric enabled")
        except ImportError:
            logger.info("✗ pycocoevalcap not available. CIDEr metric skipped.")
        
        # CLIP Score 확인
        try:
            import clip
            self.available_metrics.append('clip_score')
            logger.info("✓ CLIP available - CLIP Score metric enabled")
        except ImportError:
            logger.info("✗ CLIP not available. CLIP Score metric skipped.")
    
    def compute_text_metrics(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """모든 텍스트 평가 메트릭 계산"""
        metrics = {}
        
        if not references or not predictions:
            logger.warning("Empty references or predictions")
            return metrics
        
        # 유효한 샘플만 필터링
        valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) 
                      if ref.strip() and pred.strip()]
        
        if not valid_pairs:
            logger.warning("No valid reference-prediction pairs found")
            return metrics
        
        valid_refs, valid_preds = zip(*valid_pairs)
        logger.info(f"Computing metrics for {len(valid_pairs)} valid samples")
        
        # BLEU scores
        if 'bleu' in self.available_metrics:
            metrics.update(self._compute_bleu(valid_refs, valid_preds))
        
        # ROUGE scores
        if 'rouge' in self.available_metrics:
            metrics.update(self._compute_rouge(valid_refs, valid_preds))
        
        # METEOR
        if 'meteor' in self.available_metrics:
            metrics.update(self._compute_meteor(valid_refs, valid_preds))
        
        # CIDEr
        if 'cider' in self.available_metrics:
            metrics.update(self._compute_cider(valid_refs, valid_preds))
        
        # 기본 통계
        metrics.update(self._compute_basic_stats(valid_refs, valid_preds))
        
        return metrics
    
    def _compute_bleu(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """BLEU 점수 계산"""
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            
            ref_tokens = [[ref.split()] for ref in references]
            pred_tokens = [pred.split() for pred in predictions]
            smoothing = SmoothingFunction().method1
            
            return {
                'bleu1': corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing),
                'bleu2': corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing),
                'bleu3': corpus_bleu(ref_tokens, pred_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing),
                'bleu4': corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            }
        except Exception as e:
            logger.error(f"Error computing BLEU: {e}")
            return {}
    
    def _compute_rouge(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """ROUGE 점수 계산"""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
            
            for ref, pred in zip(references, predictions):
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            return {
                'rouge1': np.mean(rouge1_scores),
                'rouge2': np.mean(rouge2_scores),
                'rougeL': np.mean(rougeL_scores)
            }
        except Exception as e:
            logger.error(f"Error computing ROUGE: {e}")
            return {}
    
    def _compute_meteor(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """METEOR 점수 계산"""
        try:
            from nltk.translate.meteor_score import meteor_score
            
            meteor_scores = []
            for ref, pred in zip(references, predictions):
                try:
                    score = meteor_score([ref.split()], pred.split())
                    meteor_scores.append(score)
                except:
                    meteor_scores.append(0.0)
            
            return {'meteor': np.mean(meteor_scores)}
        except Exception as e:
            logger.error(f"Error computing METEOR: {e}")
            return {}
    
    def _compute_cider(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """CIDEr 점수 계산"""
        try:
            from pycocoevalcap.cider.cider import Cider
            
            if len(references) > 1:
                cider_scorer = Cider()
                gts = {str(i): [ref] for i, ref in enumerate(references)}
                res = {str(i): [pred] for i, pred in enumerate(predictions)}
                cider_score, _ = cider_scorer.compute_score(gts, res)
                return {'cider': cider_score}
        except Exception as e:
            logger.warning(f"Error computing CIDEr: {e}")
        return {}
    
    def _compute_basic_stats(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """기본 통계 계산"""
        ref_lengths = [len(ref.split()) for ref in references]
        pred_lengths = [len(pred.split()) for pred in predictions]
        
        return {
            'avg_ref_length': np.mean(ref_lengths),
            'avg_pred_length': np.mean(pred_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0,
            'empty_predictions': sum(1 for pred in predictions if not pred.strip()) / len(predictions)
        }

def load_model(checkpoint_path: str, stage: str, device: torch.device, **kwargs) -> VLMModule:
    """모델 로드"""
    logger.info(f"Loading {stage} model from: {checkpoint_path}")
    
    # 체크포인트 검증
    checkpoint = safe_load_checkpoint(checkpoint_path)
    if not checkpoint:
        raise ValueError(f"Failed to load checkpoint: {checkpoint_path}")
    
    # 모델 로드
    model = VLMModule.load_from_checkpoint(
        checkpoint_path,
        stage=stage,
        map_location=device,
        **kwargs
    )
    model.eval()
    model = model.to(device)
    logger.info(f"✓ {stage.capitalize()} model loaded successfully on {device}")
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
                
                # Ground truth 추출
                gt_texts = []
                if "text" in batch:
                    gt_texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
                elif "labels" in batch:
                    # labels가 텐서인 경우 디코딩
                    labels = batch["labels"]
                    if torch.is_tensor(labels):
                        # datamodule의 tokenizer 사용
                        try:
                            gt_texts = datamodule.tokenizer.batch_decode(labels, skip_special_tokens=True)
                        except:
                            gt_texts = [f"label_{i}" for i in range(batch_size)]
                    else:
                        gt_texts = labels if isinstance(labels, list) else [labels]
                else:
                    gt_texts = [f"sample_{i}" for i in range(batch_size)]
                
                # 이미지 경로 추출
                image_paths = batch.get("image_path", [f"batch_{batch_idx}_sample_{i}" for i in range(batch_size)])
                
                # 모델 추론 (generate 모드)
                # PanoramaVLM은 pixel_values, max_new_tokens, temperature만 지원
                generation_params_batch = {
                    "pixel_values": pixel_values,
                    "max_new_tokens": generation_params.get("max_new_tokens", 64),
                    "temperature": generation_params.get("temperature", 0.7)
                }
                
                output = model.model.generate(**generation_params_batch)
                batch_predictions = output["text"] if isinstance(output, dict) else output
                
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
                                    "max_new_tokens": generation_params.get("max_new_tokens", 64),
                                    "temperature": generation_params.get("temperature", 0.7)
                                }
                                
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

def save_results(predictions: List[str], references: List[str], metadata: List[Dict], 
                metrics: Dict[str, float], output_dir: Path, stage: str, timestamp: str):
    """결과 저장"""
    # 상세 결과 저장
    results_file = output_dir / f"{stage}_evaluation_{timestamp}.jsonl"
    with open(results_file, "w", encoding="utf-8") as f:
        for pred, ref, meta in zip(predictions, references, metadata):
            record = {
                "prediction": pred,
                "reference": ref,
                "metadata": meta
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # 메트릭 저장
    metrics_file = output_dir / f"{stage}_metrics_{timestamp}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ {stage.capitalize()} results saved:")
    logger.info(f"  Details: {results_file}")
    logger.info(f"  Metrics: {metrics_file}")

def print_results_summary(stage: str, predictions: List[str], references: List[str], 
                         metrics: Dict[str, float], error_count: int, checkpoint_path: str):
    """결과 요약 출력"""
    logger.info(f"\n{'='*60}")
    logger.info(f"{stage.upper()} MODEL EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {checkpoint_path}")
    logger.info(f"Total samples: {len(predictions)}")
    logger.info(f"Successful predictions: {len([p for p in predictions if p.strip()])}")
    logger.info(f"Empty predictions: {len([p for p in predictions if not p.strip()])}")
    logger.info(f"Errors: {error_count}")
    
    # 메트릭 출력
    if metrics:
        logger.info(f"\nEVALUATION METRICS:")
        
        # BLEU 메트릭
        bleu_metrics = {k: v for k, v in metrics.items() if k.startswith('bleu')}
        if bleu_metrics:
            logger.info(f"  BLEU Scores:")
            for metric, value in bleu_metrics.items():
                logger.info(f"    {metric.upper()}: {value:.4f}")
        
        # ROUGE 메트릭
        rouge_metrics = {k: v for k, v in metrics.items() if k.startswith('rouge')}
        if rouge_metrics:
            logger.info(f"  ROUGE Scores:")
            for metric, value in rouge_metrics.items():
                logger.info(f"    {metric.upper()}: {value:.4f}")
        
        # 기타 메트릭
        other_metrics = {k: v for k, v in metrics.items() 
                        if not k.startswith('bleu') and not k.startswith('rouge')}
        if other_metrics:
            logger.info(f"  Other Metrics:")
            for metric, value in other_metrics.items():
                logger.info(f"    {metric.upper()}: {value:.4f}")

def main():
    parser = argparse.ArgumentParser(description="PanoLLaVA Comprehensive Model Evaluation")
    
    # 평가 단계 선택
    parser.add_argument('--stage', choices=['finetune', 'resampler', 'both'], required=True,
                       help='Evaluation stage: finetune, resampler, or both')
    
    # 체크포인트 경로
    parser.add_argument('--ckpt', help='Model checkpoint for single stage evaluation')
    parser.add_argument('--finetune-ckpt', help='Finetune model checkpoint for comparison')
    parser.add_argument('--resampler-ckpt', help='Resampler model checkpoint for comparison')
    
    # 필수 인자
    parser.add_argument('--csv-val', required=True, help='Validation CSV file')
    
    # 모델 설정
    parser.add_argument('--vision-name', default='google/siglip-base-patch16-224')
    parser.add_argument('--lm-name', default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--resampler', default='mlp')
    
    # 데이터 설정
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-txt-len', type=int, default=512)
    
    # 생성 설정
    parser.add_argument('--max-new-tokens', type=int, default=64, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    
    # 출력 설정
    parser.add_argument('--output-dir', default='comprehensive_eval_results', help='Output directory')
    parser.add_argument('--save-samples', type=int, default=20, help='Number of samples to save')
    
    args = parser.parse_args()
    
    # 인자 검증
    if args.stage == 'both':
        if not args.finetune_ckpt or not args.resampler_ckpt:
            raise ValueError("Both --finetune-ckpt and --resampler-ckpt required for comparison")
    else:
        if not args.ckpt:
            raise ValueError(f"--ckpt required for {args.stage} evaluation")
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        logger.info(f"GPU Memory: {gpu_info['free']:.1f}GB free / {gpu_info['total']:.1f}GB total")
        
        # GPU 메모리에 따른 배치 크기 조정
        if gpu_info['free'] < 4 and args.batch_size > 1:
            suggested_batch_size = 1
            logger.warning(f"Low GPU memory. Reducing batch size from {args.batch_size} to {suggested_batch_size}")
            args.batch_size = suggested_batch_size
    
    # 데이터 로드
    logger.info(f"Loading validation data: {args.csv_val}")
    try:
        datamodule = VLMDataModule(
            csv_train=args.csv_val,  # dummy
            csv_val=args.csv_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tokenizer_name=args.lm_name,
            max_txt_len=args.max_txt_len
        )
        datamodule.setup()
        logger.info(f"Dataset loaded: {len(datamodule.val_ds)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # 평가 메트릭 초기화
    metrics_calculator = ComprehensiveMetrics()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 생성 파라미터 설정 (PanoramaVLM 지원 파라미터만)
    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature
    }
    
    # 모델별 평가 실행
    model_kwargs = {
        "vision_name": args.vision_name,
        "lm_name": args.lm_name,
        "resampler": args.resampler,
        "lr": 1e-5
    }
    
    results = {}
    
    if args.stage in ['finetune', 'both']:
        ckpt_path = args.finetune_ckpt if args.stage == 'both' else args.ckpt
        model = load_model(ckpt_path, 'finetune', device, **model_kwargs)
        
        predictions, references, metadata, error_count = evaluate_model(
            model, datamodule.val_dataloader(), 'finetune', device, generation_params, datamodule
        )
        
        # 메트릭 계산
        metrics = metrics_calculator.compute_text_metrics(references, predictions)
        
        # 결과 저장
        save_results(predictions, references, metadata, metrics, output_dir, 'finetune', timestamp)
        print_results_summary('finetune', predictions, references, metrics, error_count, ckpt_path)
        
        results['finetune'] = {
            'predictions': predictions,
            'references': references,
            'metrics': metrics,
            'error_count': error_count
        }
        
        del model
        torch.cuda.empty_cache()
    
    if args.stage in ['resampler', 'both']:
        ckpt_path = args.resampler_ckpt if args.stage == 'both' else args.ckpt
        model = load_model(ckpt_path, 'resampler', device, **model_kwargs)
        
        predictions, references, metadata, error_count = evaluate_model(
            model, datamodule.val_dataloader(), 'resampler', device, generation_params, datamodule
        )
        
        # 메트릭 계산
        metrics = metrics_calculator.compute_text_metrics(references, predictions)
        
        # 결과 저장
        save_results(predictions, references, metadata, metrics, output_dir, 'resampler', timestamp)
        print_results_summary('resampler', predictions, references, metrics, error_count, ckpt_path)
        
        results['resampler'] = {
            'predictions': predictions,
            'references': references,
            'metrics': metrics,
            'error_count': error_count
        }
        
        del model
        torch.cuda.empty_cache()
    
    # 비교 결과 출력 (both 모드)
    if args.stage == 'both' and len(results) == 2:
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL COMPARISON SUMMARY")
        logger.info(f"{'='*60}")
        
        for stage, result in results.items():
            logger.info(f"\n{stage.upper()} MODEL:")
            for metric, value in result['metrics'].items():
                logger.info(f"  {metric.upper()}: {value:.4f}")
        
        # 승리 카운트
        finetune_wins = 0
        resampler_wins = 0
        
        for metric in results['finetune']['metrics']:
            if metric in results['resampler']['metrics']:
                f_val = results['finetune']['metrics'][metric]
                r_val = results['resampler']['metrics'][metric]
                if f_val > r_val:
                    finetune_wins += 1
                elif r_val > f_val:
                    resampler_wins += 1
        
        logger.info(f"\nMETRIC COMPARISON:")
        logger.info(f"  Finetune wins: {finetune_wins}")
        logger.info(f"  Resampler wins: {resampler_wins}")
        
        if finetune_wins > resampler_wins:
            logger.info(f"  🏆 Overall winner: FINETUNE model")
        elif resampler_wins > finetune_wins:
            logger.info(f"  🏆 Overall winner: RESAMPLER model")
        else:
            logger.info(f"  🤝 Performance tie")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Comprehensive evaluation completed!")
    logger.info(f"Results saved in: {output_dir}")
    logger.info(f"{'='*60}")

if __name__ == '__main__':
    main()