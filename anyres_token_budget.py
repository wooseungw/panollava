
# anyres_token_budget.py
# AnyRes 구성(전역+타일)과 ViT 패치 크기 기준 총 토큰 수 계산 및 축약 권고

import math
import argparse
from typing import Tuple

def tokens_for_square(size: int, patch: int, add_cls: bool=True) -> int:
    """정사각 size 입력을 패치크기 patch로 토큰화했을 때의 토큰 수(approx)."""
    n = math.floor(size / patch)
    tok = n * n
    return tok + (1 if add_cls else 0)

def total_tokens(
    base_size: int, tile_size: int, num_tiles: int,
    patch: int = 14, add_cls: bool=True, include_global: bool=True
) -> int:
    t_global = tokens_for_square(base_size, patch, add_cls) if include_global else 0
    t_each = tokens_for_square(tile_size, patch, add_cls)
    return t_global + num_tiles * t_each

def recommend_shrink_factor(
    base_size: int, tile_size: int, num_tiles: int,
    patch: int, T_max: int, add_cls: bool=True, include_global: bool=True
) -> Tuple[int, int]:
    """
    토큰이 T_max를 넘으면 타일 토큰을 s^2 배 축소(=해상도/patch 등가 축소)하는 정수 s>=1 선택.
    반환: (s, total_after)
    """
    current = total_tokens(base_size, tile_size, num_tiles, patch, add_cls, include_global)
    if current <= T_max:
        return (1, current)  # 축약 불필요

    # 타일 토큰만 정수 축소인자 s^2로 줄이는 보수적 정책
    # s in {1,2,3,4,...} 중 최소 s 찾기
    t_global = tokens_for_square(base_size, patch, add_cls) if include_global else 0
    t_each = tokens_for_square(tile_size, patch, add_cls)

    s = 2
    while True:
        t_each_shrunk = max(1, math.floor(t_each / (s * s)))
        total = t_global + num_tiles * t_each_shrunk
        if total <= T_max:
            return (s, total)
        s += 1

def tiles_by_sliding(hfov_deg: float, overlap: float, yaw_span: float=360.0, pitch_span: float=60.0) -> int:
    """수평×수직 슬라이딩으로 생성되는 타일 개수 근사."""
    def count(span, fov, ov):
        step = fov * (1.0 - ov)
        return max(1, math.ceil((span - fov) / step) + 1)
    nx = count(yaw_span, hfov_deg, overlap)
    ny = count(pitch_span, hfov_deg, overlap)  # vfov≈hfov 가정
    return nx * ny

def main():
    ap = argparse.ArgumentParser(description="AnyRes 토큰 예산 계산기")
    ap.add_argument("--base_size", type=int, default=336)
    ap.add_argument("--tile_size", type=int, default=672)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--yaw_span", type=float, default=360.0)
    ap.add_argument("--pitch_span", type=float, default=60.0)
    ap.add_argument("--patch", type=int, default=14, help="ViT 패치 크기(예: 14/16)")
    ap.add_argument("--T_max", type=int, default=4096, help="LLM 입력 총 토큰 상한(비전 토큰만 가정 시)")
    ap.add_argument("--no_global", action="store_true", help="전역 입력 제외")
    ap.add_argument("--no_cls", action="store_true", help="CLS 토큰 제외(근사)")
    args = ap.parse_args()

    n_tiles = tiles_by_sliding(args.hfov, args.overlap, args.yaw_span, args.pitch_span)
    total = total_tokens(
        base_size=args.base_size,
        tile_size=args.tile_size,
        num_tiles=n_tiles,
        patch=args.patch,
        add_cls=(not args.no_cls),
        include_global=(not args.no_global),
    )
    print(f"[설정] base={args.base_size}, tile={args.tile_size}, patch={args.patch}, "
          f"hfov={args.hfov}, overlap={args.overlap}, yaw_span={args.yaw_span}, pitch_span={args.pitch_span}")
    print(f"- 추정 타일 수: {n_tiles}")
    print(f"- 현재 총 토큰(근사): {total}")

    s, total_after = recommend_shrink_factor(
        base_size=args.base_size,
        tile_size=args.tile_size,
        num_tiles=n_tiles,
        patch=args.patch,
        T_max=args.T_max,
        add_cls=(not args.no_cls),
        include_global=(not args.no_global),
    )
    if s == 1:
        print(f"- 축약 불필요: total={total_after} <= T_max={args.T_max}")
    else:
        print(f"- 권장 축약 인자 s={s} (타일 토큰을 1/s^2로 축소) → 총 토큰≈{total_after} (≤ T_max={args.T_max})")

if __name__ == "__main__":
    main()
