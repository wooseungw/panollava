# anyres_e2p_nope.py
# ERP(2:1 equirectangular) → pinhole 타일(AnyRes 스타일) 전처리
# ※ 포지셔널 임베딩(PE) 계산은 포함하지 않음
# deps: py360convert, opencv-python, pillow, numpy, torch

import math
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image
import torch

# ----------------------------
# py360convert: ERP → Pinhole
# ----------------------------
try:
    from py360convert import e2p as erp_to_persp
    _HAS_PY360 = True
except Exception:
    _HAS_PY360 = False
    erp_to_persp = None


# ----------------------------
# 유틸
# ----------------------------
def deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def yaw_pitch_to_xyz(yaw_deg: float, pitch_deg: float) -> Tuple[float, float, float]:
    """yaw(수평, +CCW), pitch(수직, +up) → 단위벡터 (x,y,z)"""
    yawr, pitchr = deg2rad(yaw_deg), deg2rad(pitch_deg)
    cp = math.cos(pitchr)
    x = cp * math.cos(yawr)
    y = math.sin(pitchr)
    z = cp * math.sin(yawr)
    return (x, y, z)

def letterbox_square(img: Image.Image, size: int, fill=(0, 0, 0)) -> Image.Image:
    """긴 변을 size로 맞추고 짧은 변은 패딩하여 정사각형으로."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)

def resize_to_vit(t: torch.Tensor, vit_size: Optional[int]) -> torch.Tensor:
    """vit_size가 주어지면 정사각형으로 리사이즈, 아니면 원본 유지."""
    if vit_size is None:
        return t
    return torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(vit_size, vit_size), mode="bilinear", align_corners=False
    ).squeeze(0)


# ----------------------------
# 타일링 파라미터
# ----------------------------
def compute_vfov_from_hfov(hfov_deg: float, out_size: int) -> float:
    """정사각형 출력 기준: 픽셀 종횡비 1 → vFOV ≈ hFOV"""
    return hfov_deg

def _norm_angle_180(x: float) -> float:
    """[-180, 180)로 정규화"""
    return ((x + 180.0) % 360.0) - 180.0

def make_yaw_centers_standard(hfov_deg: float,
                              overlap: float,
                              yaw_start: float = -180.0,
                              yaw_end: float = 180.0,
                              phase_deg: float = 0.0) -> List[float]:
    """
    열린 구간 방식(기본): 중심을 yaw_start + hfov/2 + phase_deg에서 시작해 step씩.
    경계(±180°)를 타일 '중심'으로 두지 않음.
    """
    assert 0.0 <= overlap < 1.0
    step = hfov_deg * (1.0 - overlap)
    if step <= 0:
        raise ValueError("Invalid overlap leading to non-positive step.")
    centers = []
    cur = yaw_start + hfov_deg / 2.0 + phase_deg
    while cur < yaw_end - hfov_deg / 2.0 + 1e-9:
        centers.append(_norm_angle_180(cur))
        cur += step
    return centers

def make_yaw_centers_closed_loop(hfov_deg: float,
                                 overlap: float,
                                 start_deg: float = -180.0,
                                 seam_phase_deg: float = 0.0) -> List[float]:
    """
    폐곡선 방식(권장): 360°를 균일 간격으로 닫힘 분할.
    - step_raw = hfov*(1-overlap)
    - N = ceil(360/step_raw)
    - 실제 간격은 360/N로 재조정하여 원을 정확히 닫음
    - seam_phase_deg = -hfov/2 → 첫 중심이 -180°가 되도록 쉬프트 가능
    """
    assert 0.0 <= overlap < 1.0
    step_raw = hfov_deg * (1.0 - overlap)
    N = max(1, math.ceil(360.0 / step_raw))
    step = 360.0 / N

    centers = []
    cur = start_deg + seam_phase_deg
    for _ in range(N):
        centers.append(_norm_angle_180(cur))
        cur += step

    # 중복 제거(수치 오차 보호)
    uniq = []
    for c in centers:
        if all(abs(_norm_angle_180(c - u)) > 1e-6 for u in uniq):
            uniq.append(c)
    # 보기 좋게 정렬(–180 기준 시계)
    return sorted(uniq, key=lambda x: ((x + 360.0) % 360.0))

def maybe_add_seam_center(centers: List[float], seam_center: float = -180.0) -> List[float]:
    """–180°(=+180°) 중심을 강제로 추가 (이미 있으면 그대로)."""
    def ang_diff(a, b):
        return abs(_norm_angle_180(a - b))
    if all(ang_diff(c, seam_center) > 1e-6 for c in centers):
        return [_norm_angle_180(seam_center)] + centers
    return centers

def make_pitch_centers(vfov_deg: float,
                       overlap: float,
                       pitch_min: float,
                       pitch_max: float) -> List[float]:
    """수직 범위에서 vfov와 overlap 기반으로 슬라이딩."""
    assert pitch_min < pitch_max
    step = vfov_deg * (1.0 - overlap)
    if step <= 0:
        raise ValueError("Invalid overlap leading to non-positive step.")
    centers = []
    cur = pitch_min + vfov_deg / 2.0
    while cur <= pitch_max - vfov_deg / 2.0 + 1e-9:
        centers.append(cur)
        cur += step
    return centers


# ----------------------------
# ERP → pinhole 타일 생성
# ----------------------------
@dataclass
class TileMeta:
    tile_id: int
    yaw_deg: float
    pitch_deg: float
    hfov_deg: float
    vfov_deg: float
    center_xyz: Tuple[float, float, float]

@dataclass
class AnyResPack:
    global_image: torch.Tensor          # (3, G, G) (ViT 입력 크기로 선택적 리사이즈 완료)
    tiles: torch.Tensor                 # (N, 3, T, T) (ViT 입력 크기로 선택적 리사이즈 완료)
    metas: List[TileMeta]               # N
    global_meta: Dict                   # 부가정보

def erp_to_pinhole_tile(erp: np.ndarray,
                        yaw_deg: float,
                        pitch_deg: float,
                        hfov_deg: float,
                        out_size: int) -> Image.Image:
    if not _HAS_PY360:
        raise ImportError("py360convert 필요: `pip install py360convert opencv-python`")
    vfov = compute_vfov_from_hfov(hfov_deg, out_size)

    # py360convert는 BGR numpy를 기대
    if erp.dtype != np.uint8:
        erp_u8 = (np.clip(erp, 0, 1) * 255).astype(np.uint8) if erp.max() <= 1.0 else erp.astype(np.uint8)
    else:
        erp_u8 = erp
    bgr = erp_u8[:, :, ::-1]  # RGB→BGR

    persp = erp_to_persp(bgr, hfov_deg, yaw_deg, pitch_deg, out_hw=(out_size, out_size))
    rgb = persp[:, :, ::-1]
    return Image.fromarray(rgb)

def build_anyres_from_erp(
    erp_img: Image.Image,
    base_size: int = 336,               # 전역(글로벌) 정사각 크기
    tile_render_size: int = 672,        # 타일을 생성할 렌더 해상도(내부)
    vit_size: Optional[int] = None,     # 최종 ViT 입력 크기(없으면 렌더 크기 유지)
    hfov_deg: float = 90.0,
    overlap: float = 0.2,
    closed_loop_yaw: bool = False,      # 폐곡선 분할 활성화
    yaw_phase_deg: float = 0.0,         # 수평 위상 쉬프트 (–hfov/2 → –180° 중심 포함)
    include_seam_center: bool = False,  # –180° 중심 타일 강제 추가
    pitch_min: float = -45.0,
    pitch_max: float = 45.0,
    pitch_full_span: bool = False,      # –90~+90(ε) 자동 설정
    cap_eps: float = 0.5,               # ±90° 근처 수치 안정화 여유
) -> AnyResPack:

    # 1) 전역 이미지(컨텍스트)
    global_img = letterbox_square(erp_img.convert("RGB"), base_size)
    g_tensor = pil_to_tensor(global_img)
    g_tensor = resize_to_vit(g_tensor, vit_size)  # ViT 크기로 통일할 경우

    # 2) 타일 중심 계산
    if pitch_full_span:
        pitch_min = max(pitch_min, -90.0 + cap_eps)
        pitch_max = min(pitch_max,  90.0 - cap_eps)

    erp_np = np.array(erp_img.convert("RGB"))
    vfov_deg = compute_vfov_from_hfov(hfov_deg, tile_render_size)

    if closed_loop_yaw:
        yaws = make_yaw_centers_closed_loop(hfov_deg, overlap, start_deg=-180.0, seam_phase_deg=yaw_phase_deg)
    else:
        yaws = make_yaw_centers_standard(hfov_deg, overlap, yaw_start=-180.0, yaw_end=180.0, phase_deg=yaw_phase_deg)

    if include_seam_center:
        yaws = maybe_add_seam_center(yaws, seam_center=-180.0)

    pitches = make_pitch_centers(vfov_deg, overlap, pitch_min=pitch_min, pitch_max=pitch_max)

    # 3) 타일 생성
    tiles_tensors: List[torch.Tensor] = []
    metas: List[TileMeta] = []
    tid = 0
    for p in pitches:
        for y in yaws:
            tile = erp_to_pinhole_tile(erp_np, yaw_deg=y, pitch_deg=p, hfov_deg=hfov_deg, out_size=tile_render_size)
            t_tensor = pil_to_tensor(tile)
            t_tensor = resize_to_vit(t_tensor, vit_size)  # ViT 크기로 통일할 경우
            tiles_tensors.append(t_tensor)

            metas.append(TileMeta(
                tile_id=tid,
                yaw_deg=y,
                pitch_deg=p,
                hfov_deg=hfov_deg,
                vfov_deg=vfov_deg,
                center_xyz=yaw_pitch_to_xyz(y, p)
            ))
            tid += 1

    tiles = torch.stack(tiles_tensors, dim=0) if tiles_tensors else torch.empty(0, 3, tile_render_size, tile_render_size)

    gmeta = {
        "kind": "global_letterbox",
        "base_size": base_size,
        "vit_size": vit_size,
        "note": "ERP 전체 컨텍스트(정사각 패킹)"
    }
    return AnyResPack(global_image=g_tensor, tiles=tiles, metas=metas, global_meta=gmeta)
# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="ERP→pinhole AnyRes preprocessor (PE 미포함)")
    ap.add_argument("--input", type=str, required=True, help="입력 ERP 이미지 경로(2:1 equirectangular)")
    ap.add_argument("--base_size", type=int, default=336, help="전역 정사각 크기")
    ap.add_argument("--tile_size", type=int, default=672, help="타일 렌더 크기(정사각)")
    ap.add_argument("--vit_size", type=int, default=None, help="최종 ViT 입력 정사각 크기(예: 336). 생략 시 렌더 크기 유지")
    ap.add_argument("--hfov", type=float, default=90.0, help="타일 수평 FOV(deg), vFOV는 정사각 가정으로 동일 적용")
    ap.add_argument("--overlap", type=float, default=0.5, help="타일 간 겹침 비율 [0,1)")
    ap.add_argument("--closed_loop_yaw", action="store_true", help="수평 폐곡선(닫힘) 분할 사용")
    ap.add_argument("--yaw_phase_deg", type=float, default=0.0, help="수평 타일 중심 위상(deg). –hfov/2 → –180° 중심 포함")
    ap.add_argument("--include_seam_center", action="store_true", help="–180° 중심 타일 강제 추가")
    ap.add_argument("--pitch_min", type=float, default=-45.0, help="수직 최소 pitch(deg)")
    ap.add_argument("--pitch_max", type=float, default=45.0, help="수직 최대 pitch(deg)")
    ap.add_argument("--pitch_full_span", action="store_true", help="–90~+90(ε) 자동 커버")
    ap.add_argument("--cap_eps", type=float, default=0.5, help="±90° 근처 수치 안정화 여유(도)")
    ap.add_argument("--save", action="store_true", help="디버그 타일 저장")
    args = ap.parse_args()

    vit_size = args.vit_size if args.vit_size is not None else None

    erp = Image.open(args.input).convert("RGB")
    pack = build_anyres_from_erp(
        erp_img=erp,
        base_size=args.base_size,
        tile_render_size=args.tile_size,
        vit_size=vit_size,
        hfov_deg=args.hfov,
        overlap=args.overlap,
        closed_loop_yaw=args.closed_loop_yaw,
        yaw_phase_deg=args.yaw_phase_deg,
        include_seam_center=args.include_seam_center,
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
        pitch_full_span=args.pitch_full_span,
        cap_eps=args.cap_eps,
    )

    print(f"[Global] tensor: {tuple(pack.global_image.shape)}, meta={pack.global_meta}")
    print(f"[Tiles]  count={pack.tiles.shape[0]}, tensor_each={(pack.tiles.shape[1:])}")
    if pack.metas:
        print(f"       sample_meta: {asdict(pack.metas[0])}")

    # 디버그 저장
    if args.save:
        import os
        dir_name = f"vis_ex/{args.vit_size}_{args.overlap}_{args.hfov}_min{args.pitch_min}_max{args.pitch_max}"
        os.makedirs(dir_name, exist_ok=True)
        # global
        g_pil = Image.fromarray((pack.global_image.permute(1,2,0).numpy()*255).astype(np.uint8))
        g_pil.save(f"{dir_name}/global_{args.base_size if vit_size is None else args.vit_size}.jpg")
        # tiles
        for m, t in zip(pack.metas, pack.tiles):
            tp = Image.fromarray((t.permute(1,2,0).numpy()*255).astype(np.uint8))
            tp.save(f"{dir_name}/tile_{m.tile_id:04d}_yaw{m.yaw_deg:+.1f}_p{m.pitch_deg:+.1f}.jpg")

if __name__ == "__main__":
    main()
