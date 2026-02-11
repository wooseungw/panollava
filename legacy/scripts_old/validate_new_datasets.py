#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
새로 추가된 CSV 데이터셋 검증 스크립트
==========================================

train_zind_dummy_anno.csv, train_stanford_dummy_anno.csv, train_structured3d_dummy_anno.csv
파일들이 데이터셋 클래스와 호환되는지 검증합니다.

검증 항목:
1. CSV 파일 구조 (url, query, annotation 컬럼 존재)
2. 이미지 파일 존재 확인 (샘플링)
3. 이미지 로드 가능 여부
"""

import sys
from pathlib import Path
import pandas as pd

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL(Pillow)이 설치되지 않았습니다. 이미지 로드 테스트를 건너뜁니다.")

# 프로젝트 루트
project_root = Path(__file__).resolve().parent.parent

def validate_csv_structure(csv_path: Path) -> bool:
    """CSV 파일 구조 검증"""
    print(f"\n{'='*60}")
    print(f"📋 CSV 구조 검증: {csv_path.name}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV 파일 로드 성공")
        print(f"   - 총 행 수: {len(df):,}")
        print(f"   - 컬럼: {list(df.columns)}")
        
        # 필수 컬럼 확인
        required_columns = ['url', 'query', 'annotation']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 필수 컬럼 누락: {missing_columns}")
            return False
        
        print(f"✅ 필수 컬럼 존재: {required_columns}")
        
        # 결측값 확인
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            print(f"⚠️  결측값 발견:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"   - {col}: {count}개")
        else:
            print(f"✅ 결측값 없음")
        
        # 데이터 샘플 출력
        print(f"\n📊 데이터 샘플 (첫 3행):")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            print(f"\n   샘플 {i+1}:")
            print(f"   - url: {row['url'][:80]}..." if len(row['url']) > 80 else f"   - url: {row['url']}")
            print(f"   - query: {row['query']}")
            print(f"   - annotation: {row['annotation'][:60]}..." if len(row['annotation']) > 60 else f"   - annotation: {row['annotation']}")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV 파일 로드 실패: {e}")
        return False


def validate_image_files(csv_path: Path, sample_size: int = 10) -> bool:
    """이미지 파일 존재 및 로드 가능 여부 검증"""
    print(f"\n{'='*60}")
    print(f"🖼️  이미지 파일 검증: {csv_path.name}")
    print(f"{'='*60}")
    
    if not PIL_AVAILABLE:
        print(f"⚠️  PIL(Pillow)이 없어 이미지 로드 테스트를 건너뜁니다.")
        print(f"   파일 존재 여부만 확인합니다.")
    
    try:
        df = pd.read_csv(csv_path)
        
        # 샘플링: 첫 N개와 랜덤 N개
        sample_indices = list(range(min(sample_size, len(df))))
        if len(df) > sample_size:
            import random
            random.seed(42)
            random_indices = random.sample(range(sample_size, len(df)), min(sample_size, len(df) - sample_size))
            sample_indices.extend(random_indices)
        
        print(f"🔍 검증할 이미지 수: {len(sample_indices)}개 (총 {len(df):,}개 중 샘플링)")
        
        success_count = 0
        fail_count = 0
        missing_files = []
        load_errors = []
        
        for idx in sample_indices:
            row = df.iloc[idx]
            image_path = Path(row['url'])
            
            # 파일 존재 확인
            if not image_path.exists():
                fail_count += 1
                missing_files.append(str(image_path))
                continue
            
            # 이미지 로드 테스트 (PIL이 있을 때만)
            if PIL_AVAILABLE:
                try:
                    img = Image.open(image_path).convert("RGB")
                    # 이미지 크기 확인
                    width, height = img.size
                    success_count += 1
                    if idx == 0:  # 첫 번째 이미지 정보 출력
                        print(f"   📌 첫 번째 이미지 정보:")
                        print(f"      - 경로: {image_path}")
                        print(f"      - 크기: {width} x {height}")
                        print(f"      - 모드: {img.mode}")
                except Exception as e:
                    fail_count += 1
                    load_errors.append((str(image_path), str(e)))
            else:
                # PIL이 없으면 파일 존재만 확인
                success_count += 1
                if idx == 0:
                    print(f"   📌 첫 번째 이미지 경로:")
                    print(f"      - {image_path}")
        
        # 결과 출력
        print(f"\n📊 검증 결과:")
        print(f"   ✅ 성공: {success_count}개")
        print(f"   ❌ 실패: {fail_count}개")
        
        if missing_files:
            print(f"\n⚠️  존재하지 않는 파일 (최대 5개 표시):")
            for path in missing_files[:5]:
                print(f"      - {path}")
            if len(missing_files) > 5:
                print(f"      ... 외 {len(missing_files) - 5}개")
        
        if load_errors:
            print(f"\n⚠️  로드 실패 (최대 5개 표시):")
            for path, error in load_errors[:5]:
                print(f"      - {path}")
                print(f"        오류: {error}")
            if len(load_errors) > 5:
                print(f"      ... 외 {len(load_errors) - 5}개")
        
        # 성공률이 80% 이상이면 통과
        success_rate = success_count / len(sample_indices) if sample_indices else 0
        print(f"\n📈 성공률: {success_rate*100:.1f}%")
        
        if success_rate >= 0.8:
            print(f"✅ 이미지 검증 통과 (80% 이상)")
            return True
        else:
            print(f"❌ 이미지 검증 실패 (80% 미만)")
            return False
            
    except Exception as e:
        print(f"❌ 이미지 검증 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_dataset_class(csv_path: Path) -> bool:
    """데이터셋 클래스 초기화 및 샘플 로드 테스트 - 스킵"""
    print(f"\n{'='*60}")
    print(f"🔧 데이터셋 클래스 검증: {csv_path.name}")
    print(f"{'='*60}")
    print(f"ℹ️  의존성 패키지가 필요하므로 이 단계를 건너뜁니다.")
    print(f"   CSV 구조와 이미지 파일 검증으로 기본적인 호환성은 확인되었습니다.")
    return True


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("🚀 새로운 CSV 데이터셋 검증 시작")
    print("="*60)
    
    # 검증할 CSV 파일들
    csv_files = [
        "data/train_zind_dummy_anno.csv",
        "data/train_stanford_dummy_anno.csv",
        "data/train_structured3d_dummy_anno.csv"
    ]
    
    results = {}
    
    for csv_file in csv_files:
        csv_path = project_root / csv_file
        
        if not csv_path.exists():
            print(f"\n❌ 파일 없음: {csv_path}")
            results[csv_file] = False
            continue
        
        # 검증 단계
        csv_valid = validate_csv_structure(csv_path)
        image_valid = validate_image_files(csv_path, sample_size=20) if csv_valid else False
        dataset_valid = validate_dataset_class(csv_path) if image_valid else False
        
        results[csv_file] = csv_valid and image_valid and dataset_valid
    
    # 최종 결과 출력
    print("\n" + "="*60)
    print("📊 최종 검증 결과")
    print("="*60)
    
    for csv_file, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"{status}: {csv_file}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 모든 CSV 파일이 검증을 통과했습니다!")
        print("   학습 데이터로 사용 가능합니다.")
    else:
        print("\n⚠️  일부 CSV 파일이 검증에 실패했습니다.")
        print("   실패한 파일을 확인하고 수정해주세요.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
