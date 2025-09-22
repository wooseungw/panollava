#!/usr/bin/env python3
"""
Hugging Face 모델 다운로드 스크립트
네트워크 불안정 상황에서도 안정적으로 모델을 다운로드합니다.
"""

import os
import time
import json
from huggingface_hub import snapshot_download
from requests.exceptions import ConnectionError, RequestException

def load_config():
    """config.json에서 모델 정보를 로드합니다."""
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

def download_with_retry(repo_id, max_retries=5, delay=60):
    """재시도 로직이 포함된 모델 다운로드"""
    for attempt in range(max_retries):
        try:
            print(f"모델 다운로드 시도 {attempt + 1}/{max_retries}: {repo_id}")
            
            # 다운로드 실행
            local_dir = snapshot_download(
                repo_id=repo_id,
                cache_dir=None,  # 기본 캐시 디렉토리 사용
                resume_download=True,  # 중단된 다운로드 재개
                local_files_only=False,
                token=None
            )
            
            print(f"✅ 다운로드 완료: {repo_id} -> {local_dir}")
            return local_dir
            
        except (ConnectionError, RequestException) as e:
            print(f"❌ 다운로드 실패 ({attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"⏳ {delay}초 후 재시도...")
                time.sleep(delay)
            else:
                print(f"❌ 모든 재시도 실패: {repo_id}")
                raise e
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")
            raise e

def main():
    # config.json 로드
    config = load_config()
    
    # 다운로드할 모델 목록
    models_to_download = [
        config["models"]["vision_name"],
        config["models"]["language_model_name"]
    ]
    
    print("🚀 모델 다운로드를 시작합니다...")
    
    for model_name in models_to_download:
        try:
            download_with_retry(model_name)
        except Exception as e:
            print(f"❌ {model_name} 다운로드 실패: {e}")
            continue
    
    print("✅ 모든 모델 다운로드 완료!")

if __name__ == "__main__":
    main()