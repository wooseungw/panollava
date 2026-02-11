#!/usr/bin/env python3
"""
AnyRes Multi-view Cost Profiler
================================

Generates FLOPs/latency profiles and efficiency-performance trade-off curves
across token/view/resolution combinations for multi-view AnyRes inference.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # type: ignore

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    plt = None
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper dataclasses and math utilities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VisionBackboneSpec:
    """Approximate compute characteristics for a ViT-like encoder."""

    embed_dim: int = 768
    num_layers: int = 12
    mlp_ratio: float = 4.0
    num_heads: int = 12
    patch_size: int = 14
    include_cls_token: bool = True


def parse_int_list(values: str) -> List[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def parse_float_list(values: str) -> List[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def tokens_for_square(
    size: int, patch: int, add_cls: bool = True, strict: bool = False
) -> int:
    """
    Estimate tokens after patch embedding for a square input.

    Args:
        size: image size (pixels) on one side.
        patch: patch size of ViT.
        add_cls: include CLS token when True.
        strict: when True, raise error if size is not divisible by patch.
    """
    if strict and size % patch != 0:
        raise ValueError(f"size={size} is not divisible by patch={patch}")

    n = math.floor(size / patch)
    tokens = n * n
    if add_cls:
        tokens += 1
    return tokens


def tile_grid_counts(
    hfov_deg: float, overlap: float, yaw_span: float, pitch_span: float
) -> Tuple[int, int]:
    """Return tile counts along yaw and pitch for sliding-window tiling."""

    def count(span: float, fov: float, ov: float) -> int:
        if span <= 0:
            return 0
        if span <= fov:
            return 1
        step = fov * (1.0 - ov)
        if step <= 0:
            return 1
        return max(1, math.ceil((span - fov) / step) + 1)

    nx = count(yaw_span, hfov_deg, overlap)
    ny = count(pitch_span, hfov_deg, overlap)
    return nx, ny


def coverage_ratio(
    nx: int,
    ny: int,
    hfov_deg: float,
    overlap: float,
    yaw_span: float,
    pitch_span: float,
) -> float:
    """Approximate unique angular coverage from overlapping tiles."""

    def axis_coverage(n_tiles: int, span: float) -> float:
        if span <= 0 or n_tiles <= 0:
            return 0.0
        if n_tiles == 1:
            covered = min(span, hfov_deg)
        else:
            step = hfov_deg * (1.0 - overlap)
            covered = hfov_deg + max(0, n_tiles - 1) * step
            covered = min(span, covered)
        return covered / span

    yaw_cover = axis_coverage(nx, yaw_span)
    pitch_cover = axis_coverage(ny, pitch_span)
    return float(np.clip(yaw_cover * pitch_cover, 0.0, 1.0))


def vit_view_flops(tokens: int, spec: VisionBackboneSpec) -> float:
    """Compute FLOPs for a single view processed by a ViT-like encoder."""
    if tokens <= 0:
        return 0.0

    d = spec.embed_dim
    L = spec.num_layers
    d_ff = int(spec.mlp_ratio * d)

    # Self-attention projections (Q, K, V, output)
    proj_flops = 4 * tokens * d * d
    # Attention matmuls (QK^T and Attn * V) across heads
    attn_flops = 2 * (tokens**2) * d
    # MLP (two linear layers)
    mlp_flops = 2 * tokens * d * d_ff

    transformer_flops = L * (proj_flops + attn_flops + mlp_flops)

    # Patch embedding (linear projection of flattened patches)
    patch_tokens = tokens - (1 if spec.include_cls_token else 0)
    patch_embed_flops = (
        2 * patch_tokens * (spec.patch_size**2) * 3 * d
    )  # conv -> matmul approx

    return transformer_flops + patch_embed_flops


def compute_latency_ms(flops: float, throughput_tflops: float, base_ms: float) -> float:
    """Convert FLOPs to latency assuming a throughput (TFLOPs) budget."""
    if throughput_tflops <= 0:
        raise ValueError("throughput_tflops must be positive.")
    execution_ms = flops / (throughput_tflops * 1e12) * 1e3
    return base_ms + execution_ms


# ---------------------------------------------------------------------------
# Core analysis pipeline
# ---------------------------------------------------------------------------
def build_profiles(
    base_sizes: Sequence[int],
    tile_sizes: Sequence[int],
    hfov_values: Sequence[float],
    overlap_values: Sequence[float],
    yaw_span: float,
    pitch_span: float,
    include_global: bool,
    vision_spec: VisionBackboneSpec,
    throughput_tflops: float,
    base_latency_ms: float,
) -> pd.DataFrame:
    rows = []
    for base_size in base_sizes:
        global_tokens = tokens_for_square(
            base_size, vision_spec.patch_size, vision_spec.include_cls_token
        )
        global_flops = vit_view_flops(global_tokens, vision_spec)

        for tile_size in tile_sizes:
            tile_tokens = tokens_for_square(
                tile_size, vision_spec.patch_size, vision_spec.include_cls_token
            )
            tile_flops = vit_view_flops(tile_tokens, vision_spec)

            for hfov_deg in hfov_values:
                for overlap in overlap_values:
                    nx, ny = tile_grid_counts(
                        hfov_deg=hfov_deg,
                        overlap=overlap,
                        yaw_span=yaw_span,
                        pitch_span=pitch_span,
                    )
                    num_tiles = nx * ny
                    total_views = num_tiles + (1 if include_global else 0)

                    total_tokens = tile_tokens * num_tiles
                    total_flops = tile_flops * num_tiles

                    if include_global:
                        total_tokens += global_tokens
                        total_flops += global_flops

                    latency_ms = compute_latency_ms(
                        flops=total_flops,
                        throughput_tflops=throughput_tflops,
                        base_ms=base_latency_ms,
                    )
                    coverage = coverage_ratio(
                        nx=nx,
                        ny=ny,
                        hfov_deg=hfov_deg,
                        overlap=overlap,
                        yaw_span=yaw_span,
                        pitch_span=pitch_span,
                    )

                    rows.append(
                        {
                            "base_size": base_size,
                            "tile_size": tile_size,
                            "hfov_deg": hfov_deg,
                            "overlap": overlap,
                            "tiles_x": nx,
                            "tiles_y": ny,
                            "num_tiles": num_tiles,
                            "total_views": total_views,
                            "global_tokens": global_tokens if include_global else 0,
                            "tile_tokens": tile_tokens,
                            "total_tokens": total_tokens,
                            "vision_flops": total_flops,
                            "vision_flops_g": total_flops / 1e9,
                            "latency_ms": latency_ms,
                            "coverage": coverage,
                            "efficiency": coverage / latency_ms if latency_ms > 0 else 0.0,
                        }
                    )

    df = pd.DataFrame(rows)
    return df.sort_values(
        ["tile_size", "hfov_deg", "overlap", "base_size"], ignore_index=True
    )


def plot_flops_latency(df: pd.DataFrame, output_path: Path) -> None:
    """Create FLOPs and latency profile scatter plots."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib not available.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cmap = plt.cm.viridis
    tile_sizes = sorted(df["tile_size"].unique())
    colors = cmap(np.linspace(0, 1, len(tile_sizes)))

    for color, tile in zip(colors, tile_sizes):
        subset = df[df["tile_size"] == tile]
        axes[0].scatter(
            subset["total_tokens"],
            subset["vision_flops_g"],
            color=color,
            alpha=0.7,
            label=f"tile {tile}px",
        )
        axes[1].scatter(
            subset["total_tokens"],
            subset["latency_ms"],
            color=color,
            alpha=0.7,
        )

    axes[0].set_xlabel("Total vision tokens")
    axes[0].set_ylabel("Vision FLOPs (G)")
    axes[0].set_title("AnyRes FLOPs profile by tokens/views")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(True, linewidth=0.3, alpha=0.5)
    axes[0].legend(title="Tile resolution")

    axes[1].set_xlabel("Total vision tokens")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("AnyRes latency profile")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_curve(
    df: pd.DataFrame,
    output_path: Path,
    base_size: int,
    tile_size: int,
) -> None:
    """Plot coverage vs latency trade-off for different views/overlap settings."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib not available.")
    trade_df = df[(df["base_size"] == base_size) & (df["tile_size"] == tile_size)]
    if trade_df.empty:
        raise ValueError(
            f"No trade-off data for base_size={base_size}, tile_size={tile_size}."
        )

    fig, ax = plt.subplots(figsize=(6.5, 5))

    hfov_values = sorted(trade_df["hfov_deg"].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(hfov_values)))

    for color, hfov in zip(colors, hfov_values):
        subset = trade_df[trade_df["hfov_deg"] == hfov].sort_values("overlap")
        ax.plot(
            subset["latency_ms"],
            subset["coverage"],
            marker="o",
            color=color,
            label=f"HFOV {hfov:.0f}°",
        )
        for _, row in subset.iterrows():
            label = f"{int(row['total_views'])}v/{row['overlap']:.2f}"
            ax.annotate(
                label,
                (row["latency_ms"], row["coverage"]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=8,
            )

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Coverage score (0-1)")
    ax.set_title(
        f"Views/overlap trade-off (base {base_size}px, tile {tile_size}px)"
    )
    ax.set_xscale("log")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(title="Horizontal FOV")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_vegalite_flops_latency(csv_path: Path, spec_path: Path) -> None:
    """Emit a Vega-Lite spec mirroring the FLOPs/latency scatter plots."""
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "AnyRes FLOPs and latency profiles.",
        "data": {"url": csv_path.name},
        "hconcat": [
            {
                "mark": {"type": "point", "filled": True, "opacity": 0.75, "size": 60},
                "encoding": {
                    "x": {
                        "field": "total_tokens",
                        "type": "quantitative",
                        "scale": {"type": "log"},
                    },
                    "y": {
                        "field": "vision_flops_g",
                        "type": "quantitative",
                        "scale": {"type": "log"},
                    },
                    "color": {
                        "field": "tile_size",
                        "type": "nominal",
                        "title": "Tile resolution",
                    },
                    "tooltip": [
                        {"field": "base_size", "type": "ordinal", "title": "Global px"},
                        {"field": "tile_size", "type": "ordinal", "title": "Tile px"},
                        {"field": "hfov_deg", "type": "quantitative", "title": "HFOV"},
                        {"field": "overlap", "type": "quantitative"},
                        {"field": "total_views", "type": "quantitative"},
                        {"field": "vision_flops_g", "type": "quantitative"},
                        {"field": "latency_ms", "type": "quantitative"},
                    ],
                },
                "title": "Vision FLOPs vs tokens",
            },
            {
                "mark": {"type": "point", "filled": True, "opacity": 0.75, "size": 60},
                "encoding": {
                    "x": {
                        "field": "total_tokens",
                        "type": "quantitative",
                        "scale": {"type": "log"},
                    },
                    "y": {
                        "field": "latency_ms",
                        "type": "quantitative",
                        "scale": {"type": "log"},
                    },
                    "color": {
                        "field": "tile_size",
                        "type": "nominal",
                        "title": "Tile resolution",
                    },
                    "tooltip": [
                        {"field": "base_size", "type": "ordinal", "title": "Global px"},
                        {"field": "tile_size", "type": "ordinal", "title": "Tile px"},
                        {"field": "hfov_deg", "type": "quantitative", "title": "HFOV"},
                        {"field": "overlap", "type": "quantitative"},
                        {"field": "total_views", "type": "quantitative"},
                        {"field": "vision_flops_g", "type": "quantitative"},
                        {"field": "latency_ms", "type": "quantitative"},
                    ],
                },
                "title": "Latency vs tokens",
            },
        ],
        "resolve": {"scale": {"color": "shared"}},
    }
    spec_path.write_text(json.dumps(spec, indent=2))


def write_vegalite_tradeoff(
    csv_path: Path,
    spec_path: Path,
    base_size: int,
    tile_size: int,
) -> None:
    """Emit Vega-Lite spec for coverage-latency trade-off."""
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "AnyRes views/overlap efficiency-performance trade-off.",
        "data": {"url": csv_path.name},
        "transform": [
            {"filter": f"datum.base_size == {base_size}"},
            {"filter": f"datum.tile_size == {tile_size}"},
        ],
        "mark": {"type": "line", "point": {"filled": True, "size": 70}},
        "encoding": {
            "x": {
                "field": "latency_ms",
                "type": "quantitative",
                "axis": {"title": "Latency (ms)"},
                "scale": {"type": "log"},
            },
            "y": {
                "field": "coverage",
                "type": "quantitative",
                "axis": {"title": "Coverage score"},
            },
            "color": {
                "field": "hfov_deg",
                "type": "nominal",
                "title": "HFOV (°)",
            },
            "detail": {"field": "hfov_deg"},
            "tooltip": [
                {"field": "hfov_deg", "type": "quantitative", "title": "HFOV"},
                {"field": "overlap", "type": "quantitative"},
                {"field": "total_views", "type": "quantitative", "title": "Views"},
                {"field": "latency_ms", "type": "quantitative"},
                {"field": "coverage", "type": "quantitative"},
            ],
        },
    }
    spec_path.write_text(json.dumps(spec, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate AnyRes FLOPs/latency profiles and trade-off charts."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/visualizations/anyres_cost",
        help="Directory for CSV/figures.",
    )
    parser.add_argument(
        "--base_sizes",
        type=str,
        default="336,384",
        help="Comma-separated global view sizes (pixels).",
    )
    parser.add_argument(
        "--tile_sizes",
        type=str,
        default="336,448,512",
        help="Comma-separated tile view sizes (pixels).",
    )
    parser.add_argument(
        "--hfovs",
        type=str,
        default="90,75,60",
        help="Comma-separated horizontal FOV degrees for tile sweep.",
    )
    parser.add_argument(
        "--overlaps",
        type=str,
        default="0.1,0.3,0.5,0.65",
        help="Comma-separated overlap ratios (0-1).",
    )
    parser.add_argument(
        "--yaw_span",
        type=float,
        default=360.0,
        help="Total yaw span in degrees.",
    )
    parser.add_argument(
        "--pitch_span",
        type=float,
        default=90.0,
        help="Total pitch span in degrees for tiles.",
    )
    parser.add_argument(
        "--include_global",
        action="store_true",
        help="Include a global panorama view in the budget.",
    )
    parser.add_argument(
        "--vision_dim",
        type=int,
        default=768,
        help="Vision encoder embedding dimension.",
    )
    parser.add_argument(
        "--vision_layers",
        type=int,
        default=12,
        help="Number of transformer layers in vision encoder.",
    )
    parser.add_argument(
        "--vision_mlp_ratio",
        type=float,
        default=4.0,
        help="MLP hidden dim ratio for vision encoder.",
    )
    parser.add_argument(
        "--vision_heads",
        type=int,
        default=12,
        help="Number of attention heads in vision encoder.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=14,
        help="ViT patch size (pixels).",
    )
    parser.add_argument(
        "--no_cls",
        action="store_true",
        help="Exclude CLS token from token count.",
    )
    parser.add_argument(
        "--throughput_tflops",
        type=float,
        default=120.0,
        help="Effective device throughput for latency estimates.",
    )
    parser.add_argument(
        "--base_latency_ms",
        type=float,
        default=20.0,
        help="Fixed latency overhead (kernel launches, data movement).",
    )
    parser.add_argument(
        "--trade_base",
        type=int,
        default=None,
        help="Base size to use for trade-off curve (defaults to first base size).",
    )
    parser.add_argument(
        "--trade_tile",
        type=int,
        default=None,
        help="Tile size to use for trade-off curve (defaults to largest tile size).",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    base_sizes = parse_int_list(args.base_sizes)
    tile_sizes = parse_int_list(args.tile_sizes)
    hfovs = parse_float_list(args.hfovs)
    overlaps = parse_float_list(args.overlaps)

    vision_spec = VisionBackboneSpec(
        embed_dim=args.vision_dim,
        num_layers=args.vision_layers,
        mlp_ratio=args.vision_mlp_ratio,
        num_heads=args.vision_heads,
        patch_size=args.patch_size,
        include_cls_token=not args.no_cls,
    )

    df = build_profiles(
        base_sizes=base_sizes,
        tile_sizes=tile_sizes,
        hfov_values=hfovs,
        overlap_values=overlaps,
        yaw_span=args.yaw_span,
        pitch_span=args.pitch_span,
        include_global=args.include_global,
        vision_spec=vision_spec,
        throughput_tflops=args.throughput_tflops,
        base_latency_ms=args.base_latency_ms,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "anyres_cost_profiles.csv"
    df.to_csv(csv_path, index=False)

    trade_base = args.trade_base if args.trade_base is not None else base_sizes[0]
    trade_tile = args.trade_tile if args.trade_tile is not None else max(tile_sizes)

    if MATPLOTLIB_AVAILABLE:
        flops_latency_path = output_dir / "anyres_flops_latency_profile.png"
        tradeoff_path = output_dir / "anyres_views_overlap_tradeoff.png"
        plot_flops_latency(df, flops_latency_path)
        plot_tradeoff_curve(
            df=df,
            output_path=tradeoff_path,
            base_size=trade_base,
            tile_size=trade_tile,
        )
        plot_outputs = [flops_latency_path.name, tradeoff_path.name]
    else:
        flops_latency_spec = output_dir / "anyres_flops_latency_profile.json"
        tradeoff_spec = output_dir / "anyres_views_overlap_tradeoff.json"
        write_vegalite_flops_latency(csv_path=csv_path, spec_path=flops_latency_spec)
        write_vegalite_tradeoff(
            csv_path=csv_path,
            spec_path=tradeoff_spec,
            base_size=trade_base,
            tile_size=trade_tile,
        )
        plot_outputs = [flops_latency_spec.name, tradeoff_spec.name]

    metadata = {
        "base_sizes": base_sizes,
        "tile_sizes": tile_sizes,
        "hfovs": hfovs,
        "overlaps": overlaps,
        "yaw_span": args.yaw_span,
        "pitch_span": args.pitch_span,
        "include_global": args.include_global,
        "vision_spec": vision_spec.__dict__,
        "throughput_tflops": args.throughput_tflops,
        "base_latency_ms": args.base_latency_ms,
        "csv_path": str(csv_path),
    }
    meta_path = output_dir / "anyres_cost_profile_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"[AnyRes Cost Profiler] Saved CSV to {csv_path}")
    print(f"[AnyRes Cost Profiler] Visual assets saved under {output_dir}: {plot_outputs}")


if __name__ == "__main__":
    main()
