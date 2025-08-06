#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PanoLLaVA 평가 결과 분석 도구
================================
CSV 평가 결과 파일을 분석하여 다양한 지표를 계산하고 시각화합니다.

사용법:
    python analyze_eval_results.py --csv-file lora_finetune_eval_results/finetune_detailed_results_20250803_090307.csv
    python analyze_eval_results.py --csv-file results.csv --save-plots --output-dir analysis_output
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (matplotlib) - 서버 환경에 맞게 수정
try:
    plt.rcParams['font.family'] = ['DejaVu Sans']
except:
    plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class EvaluationAnalyzer:
    """평가 결과 분석 클래스"""
    
    def __init__(self, csv_file: str):
        """
        Args:
            csv_file: 평가 결과 CSV 파일 경로
        """
        self.csv_file = Path(csv_file)
        self.df = None
        self.metrics = {}
        self.load_data()
    
    def load_data(self):
        """CSV 데이터 로드 및 전처리"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✓ 데이터 로드 완료: {len(self.df)} 샘플")
            print(f"  컬럼: {list(self.df.columns)}")
            
            # 기본 통계
            print(f"\n=== 기본 통계 ===")
            print(f"전체 샘플 수: {len(self.df)}")
            print(f"스테이지: {self.df['stage'].unique()}")
            print(f"빈 예측: {self.df['is_empty_prediction'].sum()}")
            print(f"개별 처리: {self.df['individual_processing'].sum()}")
            
            # 데이터 정리
            self.df['prediction'] = self.df['prediction'].fillna('')
            self.df['reference'] = self.df['reference'].fillna('')
            
        except Exception as e:
            raise ValueError(f"CSV 파일 로드 실패: {e}")
    
    def compute_text_metrics(self) -> Dict[str, float]:
        """텍스트 평가 메트릭 계산"""
        print(f"\n=== 텍스트 메트릭 계산 ===")
        
        metrics = {}
        valid_samples = []
        valid_predictions = []
        
        # 유효한 샘플 필터링 (참조 텍스트가 있는 경우)
        for idx, row in self.df.iterrows():
            pred = str(row['prediction']).strip()
            ref = str(row['reference']).strip()
            
            if pred and ref:  # 둘 다 비어있지 않은 경우
                valid_samples.append((pred, ref))
            elif pred:  # 예측만 있는 경우 (참조 없는 메트릭용)
                valid_predictions.append(pred)
        
        print(f"참조가 있는 유효한 샘플: {len(valid_samples)}/{len(self.df)}")
        print(f"예측만 있는 샘플: {len(valid_predictions)}/{len(self.df)}")
        
        # 참조가 있는 경우 BLEU, ROUGE 등 계산
        if len(valid_samples) > 0:
            predictions, references = zip(*valid_samples)
        
        # 참조가 있는 경우 BLEU, ROUGE 등 계산
        if len(valid_samples) > 0:
            predictions, references = zip(*valid_samples)
            
            # BLEU 점수 계산
            try:
                import nltk
                from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
                
                nltk.download('punkt', quiet=True)
                
                ref_tokens = [[ref.split()] for ref in references]
                pred_tokens = [pred.split() for pred in predictions]
                smoothing = SmoothingFunction().method1
                
                metrics.update({
                    'bleu1': corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing),
                    'bleu2': corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing),
                    'bleu3': corpus_bleu(ref_tokens, pred_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing),
                    'bleu4': corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                })
                print("✓ BLEU 메트릭 계산 완료")
                
            except ImportError:
                print("✗ NLTK 없음 - BLEU 건너뜀")
            except Exception as e:
                print(f"✗ BLEU 계산 오류: {e}")
            
            # ROUGE 점수 계산
            try:
                from rouge_score import rouge_scorer
                
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
                
                for ref, pred in zip(references, predictions):
                    scores = scorer.score(ref, pred)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                
                metrics.update({
                    'rouge1': np.mean(rouge1_scores),
                    'rouge2': np.mean(rouge2_scores),
                    'rougeL': np.mean(rougeL_scores)
                })
                print("✓ ROUGE 메트릭 계산 완료")
                
            except ImportError:
                print("✗ rouge-score 없음 - ROUGE 건너뜀")
            except Exception as e:
                print(f"✗ ROUGE 계산 오류: {e}")
            
            # METEOR 점수 계산
            try:
                from nltk.translate.meteor_score import meteor_score
                
                meteor_scores = []
                for ref, pred in zip(references, predictions):
                    try:
                        score = meteor_score([ref.split()], pred.split())
                        meteor_scores.append(score)
                    except:
                        meteor_scores.append(0.0)
                
                metrics['meteor'] = np.mean(meteor_scores)
                print("✓ METEOR 메트릭 계산 완료")
                
            except ImportError:
                print("✗ NLTK METEOR 없음 - METEOR 건너뜀")
            except Exception as e:
                print(f"✗ METEOR 계산 오류: {e}")
        else:
            print("📌 참조 텍스트가 없어 BLEU/ROUGE/METEOR 계산을 건너뜁니다.")

        # CLIP Score 계산 (이미지-텍스트 유사도)
        if len(valid_predictions) > 0:
            print("📌 CLIP Score 계산 시도...")
            try:
                import torch
                import clip
                from PIL import Image
                
                # CLIP 모델 로드
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load("ViT-B/32", device=device)
                
                clip_scores = []
                processed_images = 0
                
                for idx, row in self.df.iterrows():
                    if str(row['prediction']).strip():
                        try:
                            # 이미지 로드
                            image_path = row['image_path']
                            if not Path(image_path).exists():
                                # 상대 경로로 다시 시도
                                image_path = Path("/data/1_personal/4_SWWOO/panollava") / image_path
                            
                            if Path(image_path).exists():
                                image = Image.open(image_path).convert('RGB')
                                
                                # 전처리
                                image_input = preprocess(image).unsqueeze(0).to(device)
                                text_input = clip.tokenize([row['prediction']], truncate=True).to(device)
                                
                                # CLIP 스코어 계산
                                with torch.no_grad():
                                    image_features = model.encode_image(image_input)
                                    text_features = model.encode_text(text_input)
                                    
                                    # 코사인 유사도
                                    similarity = torch.cosine_similarity(image_features, text_features).item()
                                    clip_scores.append(similarity)
                                    processed_images += 1
                            
                            # 너무 많으면 샘플링 (처리 시간 단축)
                            if processed_images >= 100:
                                break
                                
                        except Exception as e:
                            continue
                
                if clip_scores:
                    metrics['clip_score'] = np.mean(clip_scores)
                    metrics['clip_score_std'] = np.std(clip_scores)
                    print(f"✓ CLIP Score 계산 완료: {processed_images}개 이미지 처리")
                else:
                    print("✗ CLIP Score 계산 실패: 처리된 이미지 없음")
                
            except ImportError:
                print("✗ CLIP 라이브러리 없음 - CLIP Score 건너뜀")
                print("  설치: pip install git+https://github.com/openai/CLIP.git")
            except Exception as e:
                print(f"✗ CLIP Score 계산 오류: {e}")

        # 참조 없는 텍스트 품질 메트릭
        if len(valid_predictions) > 0:
            print("📌 참조 없는 텍스트 품질 메트릭 계산...")
            
            # 어휘 다양성 (Vocabulary Diversity)
            all_words = []
            for pred in valid_predictions:
                all_words.extend(pred.lower().split())
            
            if all_words:
                unique_words = set(all_words)
                metrics['vocabulary_diversity'] = len(unique_words) / len(all_words)
                metrics['total_unique_words'] = len(unique_words)
                metrics['total_words'] = len(all_words)
            
            # 평균 문장 길이와 복잡성
            sentence_lengths = [len(pred.split()) for pred in valid_predictions]
            metrics['avg_sentence_length'] = np.mean(sentence_lengths)
            metrics['sentence_length_std'] = np.std(sentence_lengths)
            
            # 반복성 측정 (같은 구문의 반복)
            pred_set = set(valid_predictions)
            metrics['prediction_uniqueness'] = len(pred_set) / len(valid_predictions)
            
        
        # 기본 통계 (참조가 있는 경우)
        if len(valid_samples) > 0:
            predictions_with_ref, references = zip(*valid_samples)
            ref_lengths = [len(ref.split()) for ref in references]
            pred_lengths_with_ref = [len(pred.split()) for pred in predictions_with_ref]
            
            metrics.update({
                'avg_ref_length': np.mean(ref_lengths),
                'avg_pred_length_with_ref': np.mean(pred_lengths_with_ref),
                'length_ratio': np.mean(pred_lengths_with_ref) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0,
            })
        
        # 전체 예측에 대한 기본 통계
        all_pred_lengths = [len(str(pred).split()) for pred in self.df['prediction'] if str(pred).strip()]
        if all_pred_lengths:
            metrics.update({
                'total_predictions': len(all_pred_lengths),
                'avg_pred_length_all': np.mean(all_pred_lengths),
                'empty_predictions_ratio': self.df['is_empty_prediction'].mean()
            })
        
        self.metrics = metrics
        return metrics
    
    def _convert_numpy_types(self, obj):
        """numpy 타입을 JSON 호환 타입으로 변환"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _json_serializer(self, obj):
        """JSON 직렬화를 위한 커스텀 serializer"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return str(obj)
    
    def _deep_convert_to_json_compatible(self, obj):
        """깊은 변환으로 모든 numpy 타입을 JSON 호환 타입으로 변환"""
        if isinstance(obj, dict):
            return {str(k): self._deep_convert_to_json_compatible(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._deep_convert_to_json_compatible(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif obj is None:
            return None
        else:
            try:
                # JSON으로 직렬화 가능한지 테스트
                json.dumps(obj)
                return obj
            except:
                return str(obj)
    
    def analyze_length_distribution(self) -> Dict[str, any]:
        """텍스트 길이 분포 분석"""
        print(f"\n=== 길이 분포 분석 ===")
        
        analysis = {}
        
        # 예측 길이 분석
        pred_lengths = self.df['prediction_length'].values
        ref_lengths = self.df['reference_length'].values
        
        analysis['prediction_lengths'] = {
            'mean': np.mean(pred_lengths),
            'std': np.std(pred_lengths),
            'min': np.min(pred_lengths),
            'max': np.max(pred_lengths),
            'median': np.median(pred_lengths),
            'q25': np.percentile(pred_lengths, 25),
            'q75': np.percentile(pred_lengths, 75)
        }
        
        analysis['reference_lengths'] = {
            'mean': np.mean(ref_lengths),
            'std': np.std(ref_lengths),
            'min': np.min(ref_lengths),
            'max': np.max(ref_lengths),
            'median': np.median(ref_lengths),
            'q25': np.percentile(ref_lengths, 25),
            'q75': np.percentile(ref_lengths, 75)
        }
        
        print(f"예측 길이: 평균 {analysis['prediction_lengths']['mean']:.1f} ± {analysis['prediction_lengths']['std']:.1f}")
        print(f"참조 길이: 평균 {analysis['reference_lengths']['mean']:.1f} ± {analysis['reference_lengths']['std']:.1f}")
        
        return analysis
    
    def analyze_error_patterns(self) -> Dict[str, any]:
        """오류 패턴 분석"""
        print(f"\n=== 오류 패턴 분석 ===")
        
        analysis = {}
        
        # 빈 예측 분석
        empty_pred_ratio = self.df['is_empty_prediction'].mean()
        analysis['empty_predictions'] = {
            'count': self.df['is_empty_prediction'].sum(),
            'ratio': empty_pred_ratio,
            'percentage': empty_pred_ratio * 100
        }
        
        # 개별 처리 분석
        individual_proc_ratio = self.df['individual_processing'].mean()
        analysis['individual_processing'] = {
            'count': self.df['individual_processing'].sum(),
            'ratio': individual_proc_ratio,
            'percentage': individual_proc_ratio * 100
        }
        
        # 배치별 분석
        batch_analysis = self.df.groupby('batch_idx').agg({
            'is_empty_prediction': 'mean',
            'individual_processing': 'mean',
            'prediction_length': 'mean'
        }).reset_index()
        
        analysis['batch_performance'] = {
            'total_batches': len(batch_analysis),
            'avg_empty_per_batch': batch_analysis['is_empty_prediction'].mean(),
            'avg_individual_per_batch': batch_analysis['individual_processing'].mean()
        }
        
        print(f"빈 예측: {analysis['empty_predictions']['count']} ({analysis['empty_predictions']['percentage']:.1f}%)")
        print(f"개별 처리: {analysis['individual_processing']['count']} ({analysis['individual_processing']['percentage']:.1f}%)")
        
        return analysis
    
    def create_visualizations(self, output_dir: Optional[Path] = None):
        """시각화 생성"""
        print(f"\n=== 시각화 생성 ===")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        try:
            # 1. 길이 분포 히스토그램
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Text Length Distribution Analysis', fontsize=16)
            
            # 예측 길이 분포
            pred_lengths = self.df['prediction_length'].dropna()
            if len(pred_lengths) > 0:
                axes[0, 0].hist(pred_lengths, bins=min(30, len(pred_lengths)//2), alpha=0.7, color='blue')
                axes[0, 0].set_title('Prediction Length Distribution')
                axes[0, 0].set_xlabel('Length (words)')
                axes[0, 0].set_ylabel('Frequency')
            
            # 참조 길이 분포
            ref_lengths = self.df['reference_length'].dropna()
            if len(ref_lengths) > 0 and ref_lengths.max() > 0:
                axes[0, 1].hist(ref_lengths, bins=min(30, len(ref_lengths)//2), alpha=0.7, color='green')
                axes[0, 1].set_title('Reference Length Distribution')
                axes[0, 1].set_xlabel('Length (words)')
                axes[0, 1].set_ylabel('Frequency')
            else:
                axes[0, 1].text(0.5, 0.5, 'No reference data\n(all lengths = 0)', 
                              ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Reference Length Distribution')
            
            # 길이 비교 산점도
            valid_refs = self.df[self.df['reference_length'] > 0]
            if len(valid_refs) > 0:
                axes[1, 0].scatter(valid_refs['reference_length'], valid_refs['prediction_length'], alpha=0.5)
                max_len = max(valid_refs['reference_length'].max(), valid_refs['prediction_length'].max())
                axes[1, 0].plot([0, max_len], [0, max_len], 'r--')
                axes[1, 0].set_title('Prediction vs Reference Length')
                axes[1, 0].set_xlabel('Reference Length')
                axes[1, 0].set_ylabel('Prediction Length')
            else:
                # 참조 길이가 모두 0인 경우 - 예측 길이만 히스토그램으로 표시
                axes[1, 0].hist(self.df['prediction_length'], bins=20, alpha=0.7, color='orange')
                axes[1, 0].set_title('Prediction Length Only (No Reference)')
                axes[1, 0].set_xlabel('Prediction Length')
                axes[1, 0].set_ylabel('Frequency')
            
            # 배치별 성능
            batch_stats = self.df.groupby('batch_idx')['prediction_length'].mean()
            if len(batch_stats) > 0:
                axes[1, 1].plot(batch_stats.index, batch_stats.values, 'o-')
                axes[1, 1].set_title('Average Prediction Length by Batch')
                axes[1, 1].set_xlabel('Batch Index')
                axes[1, 1].set_ylabel('Average Length')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_dir / 'length_distribution.png', dpi=300, bbox_inches='tight')
                print(f"✓ 길이 분포 차트 저장: {output_dir / 'length_distribution.png'}")
            else:
                plt.show()
            
            # 2. 오류 패턴 분석
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle('Error Pattern Analysis', fontsize=16)
            
            # 빈 예측 비율
            empty_counts = self.df['is_empty_prediction'].value_counts()
            if len(empty_counts) == 2:
                labels = ['Valid Prediction', 'Empty Prediction']
                values = [empty_counts.get(False, 0), empty_counts.get(True, 0)]
            else:
                # 모든 예측이 유효한 경우
                labels = ['Valid Prediction']
                values = [len(self.df)]
            
            axes[0].pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[0].set_title('Empty Predictions Ratio')
            
            # 개별 처리 비율
            individual_counts = self.df['individual_processing'].value_counts()
            if len(individual_counts) == 2:
                ind_labels = ['Batch Processing', 'Individual Processing']
                ind_values = [individual_counts.get(False, 0), individual_counts.get(True, 0)]
            else:
                # 모든 처리가 배치 처리인 경우
                ind_labels = ['Batch Processing']
                ind_values = [len(self.df)]
                
            axes[1].pie(ind_values, labels=ind_labels, autopct='%1.1f%%', startangle=90)
            axes[1].set_title('Individual Processing Ratio')
            
            # 배치별 오류율
            batch_error_rates = self.df.groupby('batch_idx')['is_empty_prediction'].mean()
            if len(batch_error_rates) > 0:
                batch_indices = list(range(len(batch_error_rates)))
                axes[2].bar(batch_indices, batch_error_rates.values)
                axes[2].set_title('Error Rate by Batch')
                axes[2].set_xlabel('Batch Index')
                axes[2].set_ylabel('Empty Prediction Rate')
                
                # x축 틱 설정 (너무 많으면 간격 조정)
                if len(batch_indices) > 20:
                    step = len(batch_indices) // 10
                    axes[2].set_xticks(batch_indices[::step])
                    axes[2].set_xticklabels([f'{i}' for i in batch_error_rates.index[::step]])
            else:
                axes[2].text(0.5, 0.5, 'No batch data', ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title('Error Rate by Batch')
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_dir / 'error_patterns.png', dpi=300, bbox_inches='tight')
                print(f"✓ 오류 패턴 차트 저장: {output_dir / 'error_patterns.png'}")
            else:
                plt.show()
            
            plt.close('all')
            
        except Exception as e:
            print(f"⚠️ 시각화 생성 중 오류 발생: {e}")
            print("시각화를 건너뛰고 계속 진행합니다.")
            plt.close('all')
    
    def find_best_worst_samples(self, n_samples: int = 10) -> Dict[str, List]:
        """최고/최악 샘플 찾기"""
        print(f"\n=== 최고/최악 샘플 분석 ===")
        
        # 길이 기반 분석 (참조가 없으므로)
        valid_samples = self.df[~self.df['is_empty_prediction']].copy()
        
        if len(valid_samples) == 0:
            print("유효한 샘플이 없습니다.")
            return {'best': [], 'worst': []}
        
        # 예측 길이로 정렬
        best_samples = valid_samples.nlargest(n_samples, 'prediction_length')
        worst_samples = valid_samples.nsmallest(n_samples, 'prediction_length')
        
        best_list = []
        for _, row in best_samples.iterrows():
            best_list.append({
                'sample_id': row['sample_id'],
                'prediction': row['prediction'],
                'length': row['prediction_length'],
                'image_path': row['image_path']
            })
        
        worst_list = []
        for _, row in worst_samples.iterrows():
            worst_list.append({
                'sample_id': row['sample_id'],
                'prediction': row['prediction'],
                'length': row['prediction_length'],
                'image_path': row['image_path']
            })
        
        print(f"최고 샘플 (길이 기준): {len(best_list)}개")
        print(f"최악 샘플 (길이 기준): {len(worst_list)}개")
        
        return {'best': best_list, 'worst': worst_list}
    
    def generate_report(self, output_dir: Optional[Path] = None) -> str:
        """종합 분석 리포트 생성"""
        print(f"\n=== 종합 리포트 생성 ===")
        
        # 모든 분석 실행
        metrics = self.compute_text_metrics()
        length_analysis = self.analyze_length_distribution()
        error_analysis = self.analyze_error_patterns()
        sample_analysis = self.find_best_worst_samples()
        
        # 리포트 작성
        report = []
        report.append("# PanoLLaVA 평가 결과 분석 리포트")
        report.append("=" * 50)
        report.append(f"파일: {self.csv_file}")
        report.append(f"분석 시간: {pd.Timestamp.now()}")
        report.append("")
        
        # 기본 통계
        report.append("## 1. 기본 통계")
        report.append(f"- 전체 샘플 수: {len(self.df)}")
        report.append(f"- 스테이지: {', '.join(self.df['stage'].unique())}")
        report.append(f"- 빈 예측 수: {error_analysis['empty_predictions']['count']} ({error_analysis['empty_predictions']['percentage']:.1f}%)")
        report.append(f"- 개별 처리 수: {error_analysis['individual_processing']['count']} ({error_analysis['individual_processing']['percentage']:.1f}%)")
        report.append("")
        
        # 텍스트 메트릭
        if metrics:
            report.append("## 2. 텍스트 평가 메트릭")
            
            # 참조 기반 메트릭 (BLEU, ROUGE, METEOR)
            reference_metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge1', 'rouge2', 'rougeL', 'meteor']
            ref_metrics_found = any(metric in metrics for metric in reference_metrics)
            
            if ref_metrics_found:
                report.append("### 참조 기반 메트릭")
                for metric in reference_metrics:
                    if metric in metrics:
                        report.append(f"- {metric.upper()}: {metrics[metric]:.4f}")
                report.append("")
            
            # CLIP Score
            if 'clip_score' in metrics:
                report.append("### 이미지-텍스트 유사도")
                report.append(f"- CLIP Score: {metrics['clip_score']:.4f} ± {metrics.get('clip_score_std', 0):.4f}")
                report.append("")
            
            # 텍스트 품질 메트릭
            quality_metrics = ['vocabulary_diversity', 'prediction_uniqueness', 'avg_sentence_length']
            quality_metrics_found = any(metric in metrics for metric in quality_metrics)
            
            if quality_metrics_found:
                report.append("### 텍스트 품질 메트릭")
                if 'vocabulary_diversity' in metrics:
                    report.append(f"- 어휘 다양성: {metrics['vocabulary_diversity']:.4f}")
                if 'prediction_uniqueness' in metrics:
                    report.append(f"- 예측 고유성: {metrics['prediction_uniqueness']:.4f}")
                if 'avg_sentence_length' in metrics:
                    report.append(f"- 평균 문장 길이: {metrics['avg_sentence_length']:.1f} 단어")
                if 'total_unique_words' in metrics:
                    report.append(f"- 총 고유 단어 수: {metrics['total_unique_words']}")
                report.append("")
            
            # 기타 통계
            other_metrics = [k for k in metrics.keys() if k not in reference_metrics + quality_metrics + ['clip_score', 'clip_score_std']]
            if other_metrics:
                report.append("### 기타 통계")
                for metric in other_metrics:
                    if isinstance(metrics[metric], (int, float)):
                        if 'ratio' in metric or 'length' in metric:
                            report.append(f"- {metric}: {metrics[metric]:.4f}")
                        else:
                            report.append(f"- {metric}: {metrics[metric]}")
                report.append("")
        
        # 길이 분석
        report.append("## 3. 텍스트 길이 분석")
        pred_stats = length_analysis['prediction_lengths']
        report.append(f"- 예측 텍스트 길이: {pred_stats['mean']:.1f} ± {pred_stats['std']:.1f} 단어")
        report.append(f"  - 최소/최대: {pred_stats['min']}/{pred_stats['max']} 단어")
        report.append(f"  - 중간값: {pred_stats['median']:.1f} 단어")
        report.append("")
        
        # 샘플 예시
        report.append("## 4. 샘플 예시")
        report.append("### 가장 긴 예측 (상위 3개)")
        for i, sample in enumerate(sample_analysis['best'][:3]):
            report.append(f"**샘플 {sample['sample_id']} ({sample['length']} 단어):**")
            report.append(f"```")
            report.append(f"{sample['prediction']}")
            report.append(f"```")
            report.append("")
        
        # 권장사항
        report.append("## 5. 분석 결과 및 권장사항")
        
        if error_analysis['empty_predictions']['percentage'] > 10:
            report.append("⚠️ **빈 예측 비율이 높습니다** (>10%)")
            report.append("  - 모델 inference 파이프라인 점검 필요")
            report.append("  - generate 파라미터 조정 고려")
            report.append("")
        
        if error_analysis['individual_processing']['percentage'] > 5:
            report.append("⚠️ **개별 처리 비율이 높습니다** (>5%)")
            report.append("  - 배치 처리 시 메모리 부족 가능성")
            report.append("  - 배치 크기 감소 고려")
            report.append("")
        
        if pred_stats['mean'] < 10:
            report.append("⚠️ **예측 텍스트가 너무 짧습니다** (<10 단어)")
            report.append("  - max_new_tokens 증가 고려")
            report.append("  - temperature 조정 고려")
            report.append("")
        
        report_text = "\n".join(report)
        
        # 파일 저장
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            report_file = output_dir / 'analysis_report.md'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            # JSON으로도 저장
            json_data = {
                'basic_stats': {
                    'total_samples': int(len(self.df)),
                    'stages': list(self.df['stage'].unique()),
                    'empty_predictions': {
                        'count': int(error_analysis['empty_predictions']['count']),
                        'ratio': float(error_analysis['empty_predictions']['ratio']),
                        'percentage': float(error_analysis['empty_predictions']['percentage'])
                    },
                    'individual_processing': {
                        'count': int(error_analysis['individual_processing']['count']),
                        'ratio': float(error_analysis['individual_processing']['ratio']),
                        'percentage': float(error_analysis['individual_processing']['percentage'])
                    }
                },
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in metrics.items()},
                'length_analysis': self._convert_numpy_types(length_analysis),
                'error_analysis': self._convert_numpy_types(error_analysis),
                'sample_analysis': sample_analysis
            }
            
            json_file = output_dir / 'analysis_data.json'
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
                print(f"✓ 데이터 저장: {json_file}")
            except Exception as e:
                print(f"⚠️ JSON 저장 중 오류: {e}")
                # 대안으로 str() 변환해서 저장
                try:
                    json_data_str = self._deep_convert_to_json_compatible(json_data)
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(json_data_str, f, indent=2, ensure_ascii=False)
                    print(f"✓ 데이터 저장 (타입 변환됨): {json_file}")
                except Exception as e2:
                    print(f"✗ JSON 저장 실패: {e2}")
            
            print(f"✓ 리포트 저장: {report_file}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description='PanoLLaVA 평가 결과 분석 도구')
    
    parser.add_argument('--csv-file', required=True, help='분석할 CSV 파일 경로')
    parser.add_argument('--output-dir', help='출력 디렉토리 (기본값: csv 파일과 같은 디렉토리)')
    parser.add_argument('--save-plots', action='store_true', help='차트를 파일로 저장')
    parser.add_argument('--no-display', action='store_true', help='차트를 화면에 표시하지 않음')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.csv_file).parent / 'analysis_output'
    
    if args.save_plots:
        output_dir.mkdir(exist_ok=True)
    
    # 분석 실행
    analyzer = EvaluationAnalyzer(args.csv_file)
    
    print(f"\n{'='*60}")
    print(f"PanoLLaVA 평가 결과 분석")
    print(f"{'='*60}")
    
    # 시각화 생성
    if args.save_plots or not args.no_display:
        analyzer.create_visualizations(output_dir if args.save_plots else None)
    
    # 리포트 생성
    report = analyzer.generate_report(output_dir if args.save_plots else None)
    
    if not args.save_plots:
        print(f"\n{report}")
    
    print(f"\n{'='*60}")
    print(f"분석 완료!")
    if args.save_plots:
        print(f"결과가 {output_dir}에 저장되었습니다.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
