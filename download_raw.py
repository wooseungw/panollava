import asyncio
import aiohttp
import pandas as pd
import os
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm

class RateLimiter:
    """
    초당 요청 수(requests_per_second)를 제한하는 간단한 레이트 리미터
    """
    def __init__(self, requests_per_second: float):
        self._interval = 1.0 / requests_per_second
        self._lock = asyncio.Lock()
        self._last_call = None

    async def wait(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            if self._last_call is not None:
                elapsed = now - self._last_call
                wait_time = self._interval - elapsed
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            self._last_call = asyncio.get_event_loop().time()

async def download_image(session, url, cache_path, semaphore, rate_limiter, max_retries=3):
    """비동기 이미지 다운로드. 성공 시 로컬 경로, 실패 시 None 반환."""
    async with semaphore:
        if os.path.exists(cache_path):
            return cache_path
        
        await rate_limiter.wait()
        
        for attempt in range(max_retries):
            try:
                async with session.get(url, timeout=30) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get('Retry-After', 60))
                        print(f"[RATE_LIMIT] {url} - 대기: {retry_after}초")
                        await asyncio.sleep(retry_after)
                        continue  # 재시도
                    
                    if resp.status == 200:
                        # Content-Type 확인
                        content_type = resp.headers.get('content-type', '').lower()
                        if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                            print(f"[WARNING] {url} - 이미지가 아닌 콘텐츠: {content_type}")
                            return None
                        
                        data = await resp.read()
                        if len(data) == 0:
                            print(f"[WARNING] {url} - 빈 파일")
                            return None
                            
                        # 디렉토리 생성
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        
                        with open(cache_path, 'wb') as f:
                            f.write(data)
                        return cache_path
                    else:
                        print(f"[HTTP_ERROR] {url} - Status: {resp.status}")
                        if attempt == max_retries - 1:  # 마지막 시도
                            return None
                        await asyncio.sleep(2 ** attempt)  # 지수적 백오프
                        
            except asyncio.TimeoutError:
                print(f"[TIMEOUT] {url} - 시도 {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                print(f"[ERROR] {url} - 시도 {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
                
        return None

async def batch_download_unique(urls, cache_dir, max_concurrent, rps):
    """
    중복 URL 제거 후 한 번씩만 다운로드,
    반환값: { url: local_path or None, ... }
    """
    os.makedirs(cache_dir, exist_ok=True)
    sem = asyncio.Semaphore(max_concurrent)
    limiter = RateLimiter(rps)
    
    # SSL 및 타임아웃 설정
    connector = aiohttp.TCPConnector(
        ssl=False,  # SSL 검증 비활성화 (필요시)
        limit=100,
        limit_per_host=max_concurrent
    )
    
    timeout = aiohttp.ClientTimeout(total=60, connect=30)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    async with aiohttp.ClientSession(
        connector=connector, 
        timeout=timeout,
        headers=headers
    ) as session:
        tasks = []
        valid_urls = []
        
        for url in urls:
            if not url or pd.isna(url) or not isinstance(url, str):  # URL 유효성 검사 강화
                continue
                
            url = url.strip()  # 공백 제거
            if not url.startswith(('http://', 'https://')):  # 프로토콜 확인
                continue
                
            fname = os.path.basename(url.split('?')[0])  # 쿼리 파라미터 제거
            if not fname or '.' not in fname:
                # URL에서 해시값을 이용한 고유 파일명 생성
                fname = f"image_{abs(hash(url)) % 1000000}.jpg"  # 기본 파일명
                
            path = os.path.join(cache_dir, fname)
            tasks.append(download_image(session, url, path, sem, limiter))
            valid_urls.append(url)
            
        if not tasks:
            print("[WARNING] 다운로드할 유효한 URL이 없습니다.")
            return {url: None for url in urls}  # 모든 URL에 대해 None 반환
            
        try:
            results = await tqdm.gather(*tasks, desc="Downloading unique images")
        except Exception as e:
            print(f"[ERROR] 다운로드 중 오류 발생: {e}")
            results = [None] * len(tasks)  # 모든 결과를 None으로 설정
    
    # 원본 URLs와 결과 매핑 (유효하지 않은 URL들도 포함)
    url_to_result = {}
    valid_idx = 0
    
    for url in urls:
        if (url and not pd.isna(url) and isinstance(url, str) and 
            url.strip() and url.strip().startswith(('http://', 'https://'))):
            if valid_idx < len(results):
                url_to_result[url] = results[valid_idx]
                valid_idx += 1
            else:
                url_to_result[url] = None
        else:
            url_to_result[url] = None
    
    return url_to_result

def process_split(
    input_csv: str,
    output_csv: str,
    image_dir: str,
    url_col: str,
    max_concurrent: int,
    rps: float,
    out_dir: str,
    split: str
):
    # 1) CSV 로드
    print(f"📖 CSV 파일 로드 중: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
        print(f"   총 {len(df)} 행 발견")
    except Exception as e:
        print(f"❌ ERROR: CSV 파일 로드 실패: {e}")
        return
    
    if len(df) == 0:
        print("⚠️ WARNING: 빈 CSV 파일입니다.")
        # 빈 구조 유지하며 출력 파일 생성
        try:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"✅ 빈 CSV 파일 복사됨: {output_csv}")
        except Exception as e:
            print(f"❌ ERROR: 빈 CSV 파일 생성 실패: {e}")
        return
    
    # URL 컬럼 존재 확인
    if url_col not in df.columns:
        print(f"❌ ERROR: '{url_col}' 컬럼이 CSV에 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
        return
    
    # 유효한 URL 필터링
    valid_urls = df[url_col].dropna()
    print(f"   유효한 URL: {len(valid_urls)} 개")

    # 2) 유니크 URL 다운로드
    unique_urls = valid_urls.unique().tolist()
    # 빈 문자열이나 잘못된 URL 제거
    unique_urls = [url for url in unique_urls if url and isinstance(url, str) and url.strip()]
    print(f"   중복 제거 후: {len(unique_urls)} 개 다운로드 예정")
    
    if not unique_urls:
        print("⚠️ WARNING: 다운로드할 유효한 URL이 없습니다.")
        # 빈 CSV라도 생성하여 구조 유지
        print("📝 빈 CSV 파일을 생성합니다...")
        try:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df_empty = df.iloc[:0].copy()  # 구조만 유지하고 빈 데이터프레임
            df_empty.to_csv(output_csv, index=False)
            print(f"✅ 빈 CSV 파일 생성됨: {output_csv}")
        except Exception as e:
            print(f"❌ ERROR: 빈 CSV 파일 생성 실패: {e}")
        return
    
    url2path = asyncio.run(batch_download_unique(
        unique_urls,
        cache_dir=image_dir,
        max_concurrent=max_concurrent,
        rps=rps
    ))

    # 3) 매핑 & 필터링
    df['local_path'] = df[url_col].map(url2path)
    success_count = df['local_path'].notna().sum()
    print(f"✅ 성공적으로 다운로드: {success_count}/{len(df)} 개")
    
    # 다운로드 성공률이 너무 낮으면 경고
    success_rate = success_count / len(df) if len(df) > 0 else 0
    if success_rate < 0.1:  # 10% 미만
        print(f"⚠️ WARNING: 다운로드 성공률이 매우 낮습니다 ({success_rate:.1%})")
    
    df_filtered = df[df['local_path'].notna()].copy()
    
    if len(df_filtered) == 0:
        print("❌ ERROR: 다운로드된 이미지가 없습니다.")
        # 다운로드 실패해도 원본 구조를 유지한 빈 CSV 생성
        print("📝 원본 구조를 유지한 빈 CSV 파일을 생성합니다...")
        try:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df_empty = df.iloc[:0].copy()  # 구조만 유지
            df_empty.to_csv(output_csv, index=False)
            print(f"✅ 빈 CSV 파일 생성됨: {output_csv}")
        except Exception as e:
            print(f"❌ ERROR: 빈 CSV 파일 생성 실패: {e}")
        return

    # 4) URL 컬럼을 상대경로로 포맷
    def format_path(p: str):
        fname = os.path.basename(p)
        # data/quic360/train/images/img.jpg 와 같이 고정된 상대경로 반환
        return os.path.join(out_dir, split, "images", fname)

    df_filtered[url_col] = df_filtered['local_path'].apply(format_path)
    df_filtered.drop(columns=['local_path'], inplace=True)

    # 5) 결과 저장
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_filtered.to_csv(output_csv, index=False)
        print(f"✅ [{Path(input_csv).name}] → {output_csv} ({len(df_filtered)} rows)")
    except Exception as e:
        print(f"❌ ERROR: CSV 파일 저장 실패: {e}")
        print(f"   출력 경로: {output_csv}")
        print(f"   디렉토리 존재 여부: {os.path.exists(os.path.dirname(output_csv))}")
        
        # 대안으로 현재 디렉토리에 저장 시도
        fallback_path = f"{split}_output.csv"
        try:
            df_filtered.to_csv(fallback_path, index=False)
            print(f"✅ 대안 경로에 저장됨: {fallback_path}")
        except Exception as e2:
            print(f"❌ ERROR: 대안 저장도 실패: {e2}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="data/raw/QuIC360/*.csv → 이미지 다운로드 + CSV 갱신"
    )
    parser.add_argument("--splits", nargs="*",
                        help="처리할 split 이름(예: train test). 생략 시 raw_dir/*.csv 전부")
    parser.add_argument("--raw_dir", default="data/raw/QuIC360",
                        help="원본 CSV 디렉토리 (default: data/raw/QuIC360)")
    parser.add_argument("--out_dir", default="data/quic360",
                        help="출력 베이스 디렉토리 (default: data/quic360)")
    parser.add_argument("--url_col", default="url", help="URL 컬럼명")
    parser.add_argument("--max_concurrent", type=int, default=8, help="동시 다운로드 수")
    parser.add_argument("--requests_per_second", type=float, default=0.7, help="초당 요청 수 제한")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = args.out_dir  # e.g. "data/quic360"

    # 출력 디렉토리 미리 생성
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print(f"📁 출력 디렉토리 준비: {out_dir}")
    except Exception as e:
        print(f"❌ ERROR: 출력 디렉토리 생성 실패: {e}")
        exit(1)

    if args.splits:
        splits = args.splits
    else:
        if not raw_dir.exists():
            print(f"❌ ERROR: Raw 디렉토리가 존재하지 않습니다: {raw_dir}")
            exit(1)
        splits = [p.stem for p in raw_dir.glob("*.csv")]
        if not splits:
            print(f"❌ ERROR: {raw_dir}에 CSV 파일이 없습니다.")
            exit(1)

    print(f"🔍 처리할 splits: {splits}")

    for split in splits:
        print(f"\n🔄 [{split}] 처리 중…")
        in_csv  = raw_dir / f"{split}.csv"
        out_csv = Path(out_dir) / f"{split}.csv"
        img_dir = Path(out_dir) / split / "images"

        if not in_csv.exists():
            print(f"⚠️ 파일 없음: {in_csv}, 건너뜁니다.")
            continue

        try:
            process_split(
                input_csv=str(in_csv),
                output_csv=str(out_csv),
                image_dir=str(img_dir),
                url_col=args.url_col,
                max_concurrent=args.max_concurrent,
                rps=args.requests_per_second,
                out_dir=out_dir,
                split=split
            )
        except Exception as e:
            print(f"❌ ERROR: [{split}] 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 오류가 발생해도 다음 split 계속 처리
            continue
    
    print(f"\n✅ 모든 작업 완료!")