#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
평가 결과 CSV에 reference 추가 도구
==================================
기존 평가 결과 CSV 파일에 원본 테스트 데이터의 annotation을 reference로 추가합니다.

사용법:
    python add_references_to_results.py \
        --results-csv lora_finetune_eval_results/finetune_detailed_results_20250803_090307.csv \
        --test-csv data/quic360/test.csv \
        --output-csv lora_finetune_eval_results/finetune_detailed_results_with_ref.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_references_to_results(results_csv: str, test_csv: str, output_csv: str):
    """평가 결과에 reference 추가"""
    
    # 1. 평가 결과 로드
    logger.info(f"평가 결과 로드: {results_csv}")
    results_df = pd.read_csv(results_csv)
    logger.info(f"평가 결과 샘플 수: {len(results_df)}")
    logger.info(f"평가 결과 컬럼: {list(results_df.columns)}")
    
    # 2. 원본 테스트 데이터 로드
    logger.info(f"테스트 데이터 로드: {test_csv}")
    test_df = pd.read_csv(test_csv)
    logger.info(f"테스트 데이터 샘플 수: {len(test_df)}")
    logger.info(f"테스트 데이터 컬럼: {list(test_df.columns)}")
    
    # 3. 이미지 경로 정규화 (결과에서)
    results_df['normalized_image_path'] = results_df['image_path'].str.replace(r'^.*/data/quic360/', 'data/quic360/', regex=True)
    
    # 4. 테스트 데이터에서 이미지 경로로 매칭
    test_df['normalized_image_path'] = test_df['url']
    
    # 5. sample_id를 기반으로 매칭 (sample_id가 테스트 데이터의 인덱스라고 가정)
    logger.info("Reference 매칭 시작...")
    
    # 결과 DataFrame에 reference 컬럼 추가
    results_df['reference'] = ''
    results_df['reference_length'] = 0
    
    matched_count = 0
    unmatched_count = 0
    
    for idx, row in results_df.iterrows():
        sample_id = row['sample_id']
        
        # sample_id가 테스트 데이터의 인덱스인지 확인
        if sample_id < len(test_df):
            test_row = test_df.iloc[sample_id]
            
            # 이미지 경로가 일치하는지 확인 (추가 검증)
            if row['normalized_image_path'] == test_row['normalized_image_path']:
                annotation = str(test_row['annotation']).strip()
                results_df.at[idx, 'reference'] = annotation
                results_df.at[idx, 'reference_length'] = len(annotation.split()) if annotation else 0
                matched_count += 1
            else:
                # 이미지 경로가 다르면 경로로 매칭 시도
                matching_rows = test_df[test_df['normalized_image_path'] == row['normalized_image_path']]
                if len(matching_rows) > 0:
                    # 첫 번째 매칭되는 annotation 사용
                    annotation = str(matching_rows.iloc[0]['annotation']).strip()
                    results_df.at[idx, 'reference'] = annotation
                    results_df.at[idx, 'reference_length'] = len(annotation.split()) if annotation else 0
                    matched_count += 1
                else:
                    unmatched_count += 1
                    logger.debug(f"매칭 실패 - sample_id: {sample_id}, image: {row['normalized_image_path']}")
        else:
            unmatched_count += 1
            logger.debug(f"sample_id 범위 초과: {sample_id} >= {len(test_df)}")
    
    logger.info(f"매칭 완료: {matched_count}개 성공, {unmatched_count}개 실패")
    
    # 6. 결과 저장
    # 불필요한 컬럼 제거
    if 'normalized_image_path' in results_df.columns:
        results_df = results_df.drop('normalized_image_path', axis=1)
    
    # 기존 reference_length 컬럼 업데이트
    results_df['reference_length'] = results_df['reference'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() else 0
    )
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    logger.info(f"결과 저장: {output_path}")
    logger.info(f"최종 샘플 수: {len(results_df)}")
    logger.info(f"Reference가 있는 샘플: {(results_df['reference_length'] > 0).sum()}")
    
    # 샘플 확인
    logger.info("\n=== 샘플 확인 ===")
    for i in range(min(3, len(results_df))):
        row = results_df.iloc[i]
        logger.info(f"샘플 {i}:")
        logger.info(f"  Prediction: {row['prediction']}")
        logger.info(f"  Reference: {row['reference']}")
        logger.info(f"  Lengths: pred={row['prediction_length']}, ref={row['reference_length']}")
        logger.info("")

def main():
    parser = argparse.ArgumentParser(description='평가 결과에 reference 추가')
    
    parser.add_argument('--results-csv', required=True, help='평가 결과 CSV 파일')
    parser.add_argument('--test-csv', required=True, help='원본 테스트 CSV 파일')
    parser.add_argument('--output-csv', required=True, help='출력 CSV 파일')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.results_csv).exists():
        raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {args.results_csv}")
    
    if not Path(args.test_csv).exists():
        raise FileNotFoundError(f"테스트 파일을 찾을 수 없습니다: {args.test_csv}")
    
    add_references_to_results(args.results_csv, args.test_csv, args.output_csv)
    
    print(f"\n{'='*60}")
    print(f"Reference 추가 완료!")
    print(f"출력 파일: {args.output_csv}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
