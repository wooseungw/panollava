#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터셋 배치 출력 시각화 및 분석 테스트
===========================================

새로 구성된 ChatPanoDataset의 한 배치 출력을 상세히 분석하고 시각화하는 테스트 코드입니다.

주요 기능:
1. Train/Eval 모드별 배치 출력 분석
2. 토큰화 결과 및 라벨 시각화
3. 이미지 처리 결과 확인
4. 배치 구조 및 메타데이터 분석
5. 텍스트 포맷팅 결과 확인
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd

# 프로젝트 루트를 Python path에 추가
sys.path.append('/data/1_personal/4_SWWOO/panollava')

from panovlm.dataset import ChatPanoDataset, custom_collate_fn
from panovlm.processors.pano_llava_processor import PanoLLaVAProcessor
from panovlm.processors.image import PanoramaImageProcessor
from panovlm.processors.universal_text_formatter import UniversalTextFormatter
from transformers import AutoTokenizer

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_test_environment():
    """테스트 환경 설정"""
    print("=" * 80)
    print("🔧 데이터셋 배치 시각화 테스트 시작")
    print("=" * 80)
    
    # 설정값
    config = {
        'csv_path': "data/quic360/downtest.csv",
        'vision_model': "google/siglip-base-patch16-224",
        'language_model': "Qwen/Qwen2.5-0.5B-Instruct",
        'batch_size': 16,
        'max_text_length': 256,
        'image_size': (224, 224),
        'crop_strategy': "e2p",
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print(f"📁 CSV 경로: {config['csv_path']}")
    print(f"💻 디바이스: {config['device']}")
    print(f"🏗️  배치 크기: {config['batch_size']}")
    print(f"📏 최대 텍스트 길이: {config['max_text_length']}")
    
    # CSV 파일 존재 확인
    if not Path(config['csv_path']).exists():
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {config['csv_path']}")
    
    return config

def create_processors(config):
    """프로세서 및 토크나이저 생성"""
    print("\n📊 프로세서 초기화 중...")
    
    # 이미지 프로세서
    img_proc = PanoramaImageProcessor(
        image_size=config['image_size'],
        crop_strategy=config['crop_strategy']
    )
    
    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(config['language_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    
    # 통합 프로세서
    processor = PanoLLaVAProcessor(
        img_proc=img_proc,
        max_length=config['max_text_length']
    )
    
    print(f"✅ 프로세서 준비 완료")
    print(f"   - 이미지 크기: {config['image_size']}")
    print(f"   - 크롭 전략: {config['crop_strategy']}")
    print(f"   - 뷰 개수: {img_proc.num_views}")
    print(f"   - 토크나이저: {config['language_model']}")
    
    return processor, tokenizer

def analyze_dataset_info(csv_path):
    """데이터셋 기본 정보 분석"""
    print("\n📊 데이터셋 기본 정보:")
    
    df = pd.read_csv(csv_path)
    print(f"   - 총 샘플 수: {len(df)}")
    print(f"   - 컬럼: {list(df.columns)}")
    
    # 각 컬럼 통계
    for col in df.columns:
        if col == 'url':
            print(f"   - {col}: 이미지 경로 ({df[col].nunique()}개 고유 경로)")
        elif col == 'query':
            avg_len = df[col].str.len().mean()
            print(f"   - {col}: 평균 길이 {avg_len:.1f}자")
        elif col == 'annotation':
            avg_len = df[col].str.len().mean()
            null_count = df[col].isnull().sum()
            print(f"   - {col}: 평균 길이 {avg_len:.1f}자, 결측값 {null_count}개")
    
    return df

def create_datasets(csv_path, processor, tokenizer):
    """Train/Eval 모드 데이터셋 생성"""
    print("\n🏗️  데이터셋 생성 중...")
    
    # Train 모드 데이터셋
    train_dataset = ChatPanoDataset(
        csv_path=csv_path,
        processor=processor,
        tokenizer=tokenizer,
        mode="train",
        include_reference=True
    )
    
    # Eval 모드 데이터셋
    eval_dataset = ChatPanoDataset(
        csv_path=csv_path,
        processor=processor,
        tokenizer=tokenizer,
        mode="eval",
        include_reference=True
    )
    
    print(f"✅ 데이터셋 준비 완료")
    print(f"   - Train 샘플: {len(train_dataset)}")
    print(f"   - Eval 샘플: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

def visualize_single_sample(dataset, sample_idx=0, mode_name=""):
    """단일 샘플 상세 분석"""
    print(f"\n🔍 {mode_name} 모드 샘플 #{sample_idx} 분석:")
    
    try:
        sample = dataset[sample_idx]
        
        # 기본 정보
        print(f"   - 이미지 경로: {sample.get('image_path', 'N/A')}")
        print(f"   - 샘플 ID: {sample.get('sample_id', 'N/A')}")
        
        # 텐서 정보
        if 'pixel_values' in sample:
            pv_shape = sample['pixel_values'].shape
            print(f"   - 이미지 텐서: {pv_shape}")
            print(f"     * 뷰 수: {pv_shape[0] if len(pv_shape) >= 4 else 'N/A'}")
            print(f"     * 채널: {pv_shape[-3] if len(pv_shape) >= 3 else 'N/A'}")
            print(f"     * 크기: {pv_shape[-2:] if len(pv_shape) >= 2 else 'N/A'}")
        
        # 텍스트 정보
        if 'input_ids' in sample:
            input_len = sample['input_ids'].shape[-1]
            print(f"   - 입력 토큰 길이: {input_len}")
            
            # 토큰 디코딩 (처음 50개와 마지막 20개만)
            input_ids = sample['input_ids']
            if input_ids.dim() > 1:
                input_ids = input_ids.squeeze(0)
            
            # 전체 텍스트 디코딩
            try:
                decoded_text = dataset.tokenizer.decode(input_ids, skip_special_tokens=False)
                print(f"   - 입력 텍스트 : {decoded_text[:]}")
                
                # 특수 토큰 분석
                special_tokens = []
                for token_id in input_ids[:].tolist():  # 첫 20개 토큰만
                    token = dataset.tokenizer.decode([token_id])
                    if token in dataset.tokenizer.special_tokens_map.values():
                        special_tokens.append(f"'{token}'")
                
                if special_tokens:
                    print(f"   - 특수 토큰: {', '.join(special_tokens)}")
                
            except Exception as e:
                print(f"   - 텍스트 디코딩 오류: {e}")
        
        # 라벨 정보
        if 'labels' in sample and sample['labels'] is not None:
            labels = sample['labels']
            if labels.dim() > 1:
                labels = labels.squeeze(0)
            
            # IGNORE_INDEX가 아닌 라벨만 계산
            valid_labels = labels[labels != dataset.IGNORE_INDEX]
            print(f"   - 라벨 길이: {len(labels)} (유효: {len(valid_labels)})")
            
            if len(valid_labels) > 0:
                try:
                    # 유효한 라벨만 디코딩
                    decoded_labels = dataset.tokenizer.decode(valid_labels, skip_special_tokens=False)
                    print(f"   - 라벨 텍스트: {decoded_labels[:]}")
                except Exception as e:
                    print(f"   - 라벨 디코딩 오류: {e}")
        else:
            print(f"   - 라벨: 없음 (mode={dataset.mode})")
        
        # 참조 텍스트
        if 'reference' in sample:
            ref_text = sample['reference']
            if ref_text:
                print(f"   - 참조 텍스트: {ref_text[:]}")
            else:
                print(f"   - 참조 텍스트: 없음")
        
        # 포맷팅된 입력 텍스트
        if 'input_text' in sample:
            input_text = sample['input_text']
            print(f"   - 포맷팅된 입력: {input_text[:]}")
        
        return sample
        
    except Exception as e:
        print(f"❌ 샘플 {sample_idx} 분석 오류: {e}")
        return None

def visualize_batch(dataloader, mode_name="", max_samples=2):
    """배치 분석 및 시각화"""
    print(f"\n📦 {mode_name} 모드 배치 분석:")
    
    try:
        # 첫 번째 배치 가져오기
        batch = next(iter(dataloader))
        
        # 배치 기본 정보
        batch_size = len(batch.get('input_ids', []))
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - 배치 키: {list(batch.keys())}")
        
        # 각 키별 상세 정보
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list):
                print(f"   - {key}: 리스트 (길이: {len(value)})")
                if value and isinstance(value[0], str):
                    # 문자열 리스트인 경우 길이 정보
                    lengths = [len(s) for s in value]
                    print(f"     * 문자열 길이: {lengths}")
            else:
                print(f"   - {key}: {type(value)}")
        
        # 개별 샘플 분석 (최대 max_samples개)
        for i in range(min(batch_size, max_samples)):
            print(f"\n   📋 배치 내 샘플 #{i}:")
            
            # 텍스트 관련
            if 'input_text' in batch:
                text = batch['input_text'][i] if i < len(batch['input_text']) else 'N/A'
                print(f"      - 입력 텍스트: {text[:]}")
            
            if 'reference' in batch:
                ref = batch['reference'][i] if i < len(batch['reference']) else 'N/A'
                if ref:
                    print(f"      - 참조 텍스트: {ref[:]}")
                else:
                    print(f"      - 참조 텍스트: 없음")
            
            # 토큰 길이
            if 'input_ids' in batch and isinstance(batch['input_ids'], torch.Tensor):
                token_len = batch['input_ids'][i].shape[-1] if i < batch['input_ids'].shape[0] else 0
                print(f"      - 토큰 길이: {token_len}")
                
                # 유효한 토큰 개수 (패딩 제외)
                if 'attention_mask' in batch and isinstance(batch['attention_mask'], torch.Tensor):
                    valid_tokens = batch['attention_mask'][i].sum().item() if i < batch['attention_mask'].shape[0] else 0
                    print(f"      - 유효 토큰: {valid_tokens}")
            
            # 라벨 정보
            if 'labels' in batch and batch['labels'] is not None:
                if isinstance(batch['labels'], torch.Tensor) and i < batch['labels'].shape[0]:
                    labels = batch['labels'][i]
                    valid_labels = (labels != -100).sum().item()
                    print(f"      - 유효 라벨: {valid_labels}")
            
            # 이미지 정보
            if 'pixel_values' in batch and isinstance(batch['pixel_values'], torch.Tensor):
                if i < batch['pixel_values'].shape[0]:
                    img_shape = batch['pixel_values'][i].shape
                    print(f"      - 이미지 shape: {img_shape}")
        
        return batch
        
    except Exception as e:
        print(f"❌ 배치 분석 오류: {e}")
        return None

def create_visualization_plot(train_sample, eval_sample, train_batch, eval_batch):
    """시각화 플롯 생성"""
    print(f"\n🎨 시각화 생성 중...")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ChatPanoDataset 배치 분석', fontsize=16, fontweight='bold')
        
        # 1. 이미지 뷰 시각화 (Train)
        ax = axes[0, 0]
        if train_sample and 'pixel_values' in train_sample:
            pv = train_sample['pixel_values']
            if pv.shape[0] > 0:  # 첫 번째 뷰 표시
                img_tensor = pv[0]  # (C, H, W)
                # 정규화 해제 (대략적)
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np * 0.229 + 0.485, 0, 1)  # 대략적 역정규화
                ax.imshow(img_np)
                ax.set_title(f'Train - 첫 번째 뷰\n{img_tensor.shape}')
        ax.axis('off')
        
        # 2. 이미지 뷰 시각화 (Eval)
        ax = axes[0, 1]
        if eval_sample and 'pixel_values' in eval_sample:
            pv = eval_sample['pixel_values']
            if pv.shape[0] > 0:
                img_tensor = pv[0]
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np * 0.229 + 0.485, 0, 1)
                ax.imshow(img_np)
                ax.set_title(f'Eval - 첫 번째 뷰\n{img_tensor.shape}')
        ax.axis('off')
        
        # 3. 텍스트 길이 비교
        ax = axes[0, 2]
        modes = ['Train', 'Eval']
        input_lengths = []
        label_lengths = []
        
        for sample in [train_sample, eval_sample]:
            if sample:
                # Input 길이
                if 'input_ids' in sample:
                    input_len = sample['input_ids'].shape[-1]
                    input_lengths.append(input_len)
                else:
                    input_lengths.append(0)
                
                # Label 길이 (유효한 것만)
                if 'labels' in sample and sample['labels'] is not None:
                    labels = sample['labels']
                    if labels.dim() > 1:
                        labels = labels.squeeze(0)
                    valid_labels = (labels != -100).sum().item()
                    label_lengths.append(valid_labels)
                else:
                    label_lengths.append(0)
            else:
                input_lengths.append(0)
                label_lengths.append(0)
        
        x = np.arange(len(modes))
        width = 0.35
        ax.bar(x - width/2, input_lengths, width, label='Input 토큰', alpha=0.8)
        ax.bar(x + width/2, label_lengths, width, label='Label 토큰', alpha=0.8)
        ax.set_xlabel('모드')
        ax.set_ylabel('토큰 수')
        ax.set_title('토큰 길이 비교')
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 4. 배치 크기 비교
        ax = axes[1, 0]
        batch_info = []
        batch_names = ['Train Batch', 'Eval Batch']
        
        for batch in [train_batch, eval_batch]:
            if batch and 'input_ids' in batch:
                batch_size = batch['input_ids'].shape[0] if isinstance(batch['input_ids'], torch.Tensor) else len(batch['input_ids'])
                seq_len = batch['input_ids'].shape[1] if isinstance(batch['input_ids'], torch.Tensor) else 0
                batch_info.append((batch_size, seq_len))
            else:
                batch_info.append((0, 0))
        
        batch_sizes = [info[0] for info in batch_info]
        seq_lens = [info[1] for info in batch_info]
        
        x = np.arange(len(batch_names))
        ax.bar(x, batch_sizes, alpha=0.8, color='skyblue')
        ax.set_xlabel('배치')
        ax.set_ylabel('배치 크기')
        ax.set_title('배치 크기')
        ax.set_xticks(x)
        ax.set_xticklabels(batch_names)
        
        # 배치 크기를 막대 위에 표시
        for i, v in enumerate(batch_sizes):
            ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
        
        # 5. 시퀀스 길이
        ax = axes[1, 1]
        ax.bar(x, seq_lens, alpha=0.8, color='lightcoral')
        ax.set_xlabel('배치')
        ax.set_ylabel('시퀀스 길이')
        ax.set_title('시퀀스 길이')
        ax.set_xticks(x)
        ax.set_xticklabels(batch_names)
        
        for i, v in enumerate(seq_lens):
            ax.text(i, v + 5, str(v), ha='center', va='bottom')
        
        # 6. 데이터 구성 요약
        ax = axes[1, 2]
        ax.axis('off')
        
        # 텍스트 요약 정보
        summary_text = "📊 데이터셋 구성 요약\n\n"
        
        if train_sample:
            summary_text += "🔹 Train 모드:\n"
            summary_text += f"  • Labels: {'있음' if 'labels' in train_sample and train_sample['labels'] is not None else '없음'}\n"
            summary_text += f"  • Reference: {'있음' if 'reference' in train_sample and train_sample['reference'] else '없음'}\n"
            summary_text += f"  • 이미지 뷰: {train_sample['pixel_values'].shape[0] if 'pixel_values' in train_sample else 0}개\n\n"
        
        if eval_sample:
            summary_text += "🔹 Eval 모드:\n"
            summary_text += f"  • Labels: {'있음' if 'labels' in eval_sample and eval_sample['labels'] is not None else '없음'}\n"
            summary_text += f"  • Reference: {'있음' if 'reference' in eval_sample and eval_sample['reference'] else '없음'}\n"
            summary_text += f"  • 이미지 뷰: {eval_sample['pixel_values'].shape[0] if 'pixel_values' in eval_sample else 0}개\n\n"
        
        summary_text += "🔹 주요 차이점:\n"
        summary_text += "  • Train: 학습용 라벨 포함\n"
        summary_text += "  • Eval: 생성용, 라벨 제외\n"
        summary_text += "  • 둘 다: 참조 텍스트 제공"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # 저장
        output_path = "/data/1_personal/4_SWWOO/panollava/dataset_batch_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 시각화 저장됨: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ 시각화 오류: {e}")

def main():
    """메인 실행 함수"""
    try:
        # 1. 환경 설정
        config = setup_test_environment()
        
        # 2. 데이터셋 정보 분석
        df_info = analyze_dataset_info(config['csv_path'])
        
        # 3. 프로세서 생성
        processor, tokenizer = create_processors(config)
        
        # 4. 데이터셋 생성
        train_dataset, eval_dataset = create_datasets(
            config['csv_path'], processor, tokenizer
        )
        
        # 5. 단일 샘플 분석
        train_sample = visualize_single_sample(train_dataset, 0, "Train")
        eval_sample = visualize_single_sample(eval_dataset, 0, "Eval")
        
        # 6. DataLoader 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            collate_fn=custom_collate_fn,
            shuffle=False  # 테스트를 위해 순서 고정
        )
        
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=config['batch_size'], 
            collate_fn=custom_collate_fn,
            shuffle=False
        )
        
        # 7. 배치 분석
        train_batch = visualize_batch(train_loader, "Train", max_samples=config['batch_size'])
        eval_batch = visualize_batch(eval_loader, "Eval", max_samples=config['batch_size'])
        
        # 8. 시각화 생성
        create_visualization_plot(train_sample, eval_sample, train_batch, eval_batch)
        
        # 9. 요약 리포트
        print("\n" + "="*80)
        print("📋 최종 요약 리포트")
        print("="*80)
        print(f"✅ Train 데이터셋: {len(train_dataset)} 샘플")
        print(f"✅ Eval 데이터셋: {len(eval_dataset)} 샘플")
        print(f"✅ 배치 크기: {config['batch_size']}")
        print(f"✅ 이미지 뷰 수: {processor.img_proc.num_views}")
        print(f"✅ 최대 토큰 길이: {config['max_text_length']}")
        
        if train_sample:
            print(f"✅ Train 샘플 구성: 이미지 + 텍스트 + 라벨")
        if eval_sample:
            print(f"✅ Eval 샘플 구성: 이미지 + 텍스트 (라벨 없음)")
        
        print("\n🎯 데이터셋 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
