#!/usr/bin/env python3
"""DINO-style feature visualization for CORA panoramic VLM.

Generates publication-quality visualizations:
  1. Self-attention maps (per-head + mean) from SigLIP2 vision encoder
  2. PCA feature maps (patch features → 3 principal components → RGB)
  3. Overlap alignment heatmaps (cosine similarity between adjacent tile strips)
  4. Feature similarity maps (reference patch → cosine sim to all patches)

Each visualization is generated per-view (9 tiles) and stitched into a
panoramic strip for direct use in papers.

Usage:
  # Attention maps from raw SigLIP2 (no checkpoint needed)
  python scripts/visualize_features.py --image path/to/pano.jpg --mode attention

  # PCA features from a trained checkpoint
  python scripts/visualize_features.py --image path/to/pano.jpg --mode pca \
      --checkpoint runs/cora_contrastive/.../finetune/last.ckpt

  # Overlap alignment comparison (before vs after VICReg)
  python scripts/visualize_features.py --image path/to/pano.jpg --mode overlap \
      --checkpoint runs/cora_contrastive/.../finetune/last.ckpt

  # All visualizations at once
  python scripts/visualize_features.py --image path/to/pano.jpg --mode all \
      --checkpoint runs/cora_contrastive/.../finetune/last.ckpt

  # Compare two checkpoints (e.g. before/after VICReg)
  python scripts/visualize_features.py --image path/to/pano.jpg --mode pca \
      --checkpoint ckpt_after.ckpt --checkpoint-baseline ckpt_before.ckpt
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ── Style constants (publication quality) ────────────────────────────

DPI = 200
FONT_SIZE = 8
CMAP_ATTENTION = "inferno"
CMAP_SIMILARITY = "RdYlBu_r"
CMAP_OVERLAP = "coolwarm"

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# =====================================================================
# Feature extraction
# =====================================================================


def load_image_and_tiles(
    image_path: str,
    config: Optional[object] = None,
) -> Tuple[Image.Image, torch.Tensor, List[dict]]:
    """Load ERP image and generate E2P tiles.

    Returns:
        (original_pil, pixel_values [1, V, 3, H, W], tile_metas)
    """
    from cora.processors.images import PanoramaImageProcessor

    img = Image.open(image_path).convert("RGB")

    if config is not None:
        img_cfg = config.image_processing
        proc = PanoramaImageProcessor(
            image_size=tuple(img_cfg.image_size) if img_cfg.image_size else None,
            crop_strategy=img_cfg.crop_strategy,
            fov_deg=img_cfg.fov_deg,
            overlap_ratio=img_cfg.overlap_ratio,
            normalize=img_cfg.normalize,
            use_vision_processor=img_cfg.use_vision_processor,
            vision_model_name=config.models.vision_name,
        )
    else:
        proc = PanoramaImageProcessor(
            crop_strategy="anyres_e2p",
            fov_deg=90.0,
            overlap_ratio=0.5,
            normalize=True,
            use_vision_processor=True,
            vision_model_name="google/siglip2-so400m-patch16-256",
        )

    pixel_values = proc(img)  # [V, 3, H, W]
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)  # [1, V, 3, H, W]

    # Generate tile metadata for labels
    metas = []
    num_views = pixel_values.shape[1]
    if num_views > 1:
        metas.append({"label": "Global", "yaw": 0, "pitch": 0})
        for i in range(1, num_views):
            yaw = (i - 1) * 45  # 8 tiles at 45° stride
            metas.append({"label": f"Tile {i} ({yaw}°)", "yaw": yaw, "pitch": 0})
    else:
        metas.append({"label": "Single view", "yaw": 0, "pitch": 0})

    return img, pixel_values, metas


def get_tile_images(pixel_values: torch.Tensor) -> List[np.ndarray]:
    """Convert pixel_values [1, V, 3, H, W] back to displayable images.

    Reverses SigLIP normalization (mean=0.5, std=0.5).
    """
    tiles = pixel_values[0]  # [V, 3, H, W]
    images = []
    for t in tiles:
        img = t.cpu().float()
        img = img * 0.5 + 0.5  # un-normalize SigLIP
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        images.append(img)
    return images


@torch.inference_mode()
def extract_attention_maps(
    vision_encoder: torch.nn.Module,
    pixel_values: torch.Tensor,
    layer_idx: int = -1,
) -> torch.Tensor:
    """Extract self-attention maps from SigLIP2 via hook-based Q·K computation.

    SigLIP2 uses F.scaled_dot_product_attention which does not return
    attention weights. We register a forward pre-hook on the target
    layer's self_attn module, capture hidden_states, and compute
    softmax(Q·K^T / sqrt(d_k)) manually.

    Args:
        vision_encoder: VisionBackbone instance.
        pixel_values: [1, V, 3, H, W]
        layer_idx: Which transformer layer (-1 = last).

    Returns:
        attention: [V, num_heads, S, S] (S = 16*16 = 256 for 256px/16patch)
    """
    encoder = vision_encoder.encoder

    # Locate the target layer's self-attention module
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
        layers = encoder.encoder.layers
    elif hasattr(encoder, "layers"):
        layers = encoder.layers
    else:
        raise AttributeError("Cannot find transformer layers in vision encoder")

    target_layer = layers[layer_idx]
    attn_module = target_layer.self_attn

    # Hook to capture hidden_states input to self-attention
    captured: dict = {}

    def pre_hook(module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        if "hidden_states" in kwargs:
            captured["hs"] = kwargs["hidden_states"].detach()
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            captured["hs"] = args[0].detach()

    handle = attn_module.register_forward_pre_hook(pre_hook, with_kwargs=True)

    # Forward pass
    V = pixel_values.shape[1]
    flat = pixel_values.view(-1, *pixel_values.shape[2:])  # [V, 3, H, W]

    call_kwargs: dict = {"pixel_values": flat}
    model_type = getattr(getattr(encoder, "config", None), "model_type", "")
    if "siglip" in model_type:
        call_kwargs["interpolate_pos_encoding"] = True
    encoder(**call_kwargs, return_dict=True)

    handle.remove()

    if "hs" not in captured:
        raise RuntimeError("Hook failed to capture hidden_states from self_attn")

    # Compute attention weights from Q and K
    hs = captured["hs"]  # [V, S, D]
    BV, S, D = hs.shape
    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim

    q = attn_module.q_proj(hs).reshape(BV, S, num_heads, head_dim).transpose(1, 2)
    k = attn_module.k_proj(hs).reshape(BV, S, num_heads, head_dim).transpose(1, 2)
    scale = head_dim ** 0.5
    attn_weights = F.softmax(
        torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1,
    )  # [V, num_heads, S, S]

    return attn_weights.cpu()


@torch.inference_mode()
def extract_patch_features(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    level: str = "vision",
) -> torch.Tensor:
    """Extract patch-level features at different pipeline stages.

    Args:
        model: PanoramaVLM instance.
        pixel_values: [1, V, 3, H, W].
        level: "vision" (raw encoder), "resampler" (after BiMamba),
               "vicreg" (after VICReg projector).

    Returns:
        features: [V, S, D]
    """
    B = pixel_values.shape[0]
    V = pixel_values.shape[1]
    flat = pixel_values.view(-1, *pixel_values.shape[2:])

    # Vision encoder output
    vision_out = model.vision_encoder(flat)
    vision_feats = vision_out["vision_features"]  # [V, S, D_vis]

    if level == "vision":
        return vision_feats.cpu()

    # Resampler output
    resampled = model.resampler_module(vision_feats)  # [V, S, D_lat]

    if level == "resampler":
        return resampled.cpu()

    # VICReg projector output
    vicreg_feats = model.vicreg_projector(resampled)  # [V, S, D_proj]
    return vicreg_feats.cpu()


# =====================================================================
# Visualization 1: Self-Attention Maps (DINO style)
# =====================================================================


def visualize_attention(
    attention: torch.Tensor,
    tile_images: List[np.ndarray],
    tile_metas: List[dict],
    output_path: Path,
    num_heads_to_show: int = 6,
) -> None:
    """DINO-style attention map visualization.

    Shows per-head attention from each patch to all others,
    averaged over query positions (mean attention per key).

    Args:
        attention: [V, num_heads, S, S]
        tile_images: list of [H, W, 3] arrays
        tile_metas: list of metadata dicts with "label" key
        output_path: where to save the figure
        num_heads_to_show: how many heads to display (default: 6)
    """
    V, num_heads, S, _ = attention.shape
    h = w = int(math.sqrt(S))

    # Select heads with most diverse attention patterns
    head_entropy = []
    for head in range(num_heads):
        mean_attn = attention[:, head].mean(dim=(0, 1))  # [S]
        entropy = -(mean_attn * (mean_attn + 1e-10).log()).sum()
        head_entropy.append((head, entropy.item()))
    head_entropy.sort(key=lambda x: -x[1])
    selected_heads = [h[0] for h in head_entropy[:num_heads_to_show]]

    n_rows = num_heads_to_show + 2  # heads + mean + original
    n_cols = V

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0, n_rows * 2.0),
        squeeze=False,
    )

    for col in range(V):
        # Row 0: original tile
        axes[0][col].imshow(tile_images[col])
        axes[0][col].set_title(tile_metas[col]["label"], fontsize=FONT_SIZE)
        axes[0][col].axis("off")

        # Row 1: mean attention over all heads
        mean_attn = attention[col].mean(dim=0).mean(dim=0)  # [S]
        attn_map = mean_attn.reshape(h, w).numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        axes[1][col].imshow(tile_images[col], alpha=0.3)
        axes[1][col].imshow(attn_map, cmap=CMAP_ATTENTION, alpha=0.7,
                            interpolation="bilinear",
                            extent=[0, tile_images[col].shape[1],
                                    tile_images[col].shape[0], 0])
        if col == 0:
            axes[1][col].set_ylabel("Mean", fontsize=FONT_SIZE)
        axes[1][col].axis("off")

        # Rows 2+: per-head attention
        for row_idx, head in enumerate(selected_heads):
            ax = axes[row_idx + 2][col]
            head_attn = attention[col, head].mean(dim=0)  # [S]
            attn_map = head_attn.reshape(h, w).numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            ax.imshow(tile_images[col], alpha=0.3)
            ax.imshow(attn_map, cmap=CMAP_ATTENTION, alpha=0.7,
                      interpolation="bilinear",
                      extent=[0, tile_images[col].shape[1],
                              tile_images[col].shape[0], 0])
            if col == 0:
                ax.set_ylabel(f"Head {head}", fontsize=FONT_SIZE)
            ax.axis("off")

    fig.suptitle("Self-Attention Maps (SigLIP2, last layer)", fontsize=FONT_SIZE + 2, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Attention maps saved to %s", output_path)


# =====================================================================
# Visualization 2: PCA Feature Maps (DINOv2 style)
# =====================================================================


def visualize_pca(
    features: torch.Tensor,
    tile_images: List[np.ndarray],
    tile_metas: List[dict],
    output_path: Path,
    level_name: str = "vision",
    n_components: int = 3,
) -> None:
    """DINOv2-style PCA feature visualization.

    Projects high-dimensional patch features to 3 principal components
    and maps them to RGB for each view.

    Args:
        features: [V, S, D]
        tile_images: original tile images for comparison
        tile_metas: metadata dicts
        output_path: save path
        level_name: label for the feature level
        n_components: PCA components (3 for RGB)
    """
    V, S, D = features.shape
    h = w = int(math.sqrt(S))

    # Stack all patches for global PCA
    all_feats = features.reshape(-1, D).numpy()  # [V*S, D]

    pca = PCA(n_components=n_components)
    pca_feats = pca.fit_transform(all_feats)  # [V*S, 3]

    # Normalize each component to [0, 1] for RGB
    for c in range(n_components):
        col = pca_feats[:, c]
        pca_feats[:, c] = (col - col.min()) / (col.max() - col.min() + 1e-8)

    pca_views = pca_feats.reshape(V, h, w, n_components)

    # Plot: original + PCA side by side
    fig, axes = plt.subplots(2, V, figsize=(V * 2.0, 4.2), squeeze=False)

    for col in range(V):
        axes[0][col].imshow(tile_images[col])
        axes[0][col].set_title(tile_metas[col]["label"], fontsize=FONT_SIZE)
        axes[0][col].axis("off")

        pca_img = pca_views[col]
        # Upsample to tile resolution for overlay
        pca_up = np.array(Image.fromarray(
            (pca_img * 255).astype(np.uint8)
        ).resize(
            (tile_images[col].shape[1], tile_images[col].shape[0]),
            Image.BILINEAR,
        )) / 255.0
        axes[1][col].imshow(pca_up)
        if col == 0:
            axes[1][col].set_ylabel("PCA → RGB", fontsize=FONT_SIZE)
        axes[1][col].axis("off")

    var_explained = pca.explained_variance_ratio_[:3].sum() * 100
    fig.suptitle(
        f"PCA Feature Map ({level_name}) — {var_explained:.1f}% variance explained",
        fontsize=FONT_SIZE + 2, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("PCA map (%s) saved to %s", level_name, output_path)


def visualize_pca_comparison(
    features_before: torch.Tensor,
    features_after: torch.Tensor,
    tile_images: List[np.ndarray],
    tile_metas: List[dict],
    output_path: Path,
    label_before: str = "Before VICReg",
    label_after: str = "After VICReg",
) -> None:
    """Side-by-side PCA comparison of features before/after training."""
    V, S, D = features_before.shape
    h = w = int(math.sqrt(S))

    # Fit PCA on combined features for consistent color mapping
    all_feats = torch.cat([
        features_before.reshape(-1, D),
        features_after.reshape(-1, D),
    ], dim=0).numpy()

    pca = PCA(n_components=3)
    pca_all = pca.fit_transform(all_feats)

    for c in range(3):
        col = pca_all[:, c]
        pca_all[:, c] = (col - col.min()) / (col.max() - col.min() + 1e-8)

    n_patches = V * S
    pca_before = pca_all[:n_patches].reshape(V, h, w, 3)
    pca_after = pca_all[n_patches:].reshape(V, h, w, 3)

    fig, axes = plt.subplots(3, V, figsize=(V * 2.0, 6.3), squeeze=False)

    for col in range(V):
        axes[0][col].imshow(tile_images[col])
        axes[0][col].set_title(tile_metas[col]["label"], fontsize=FONT_SIZE)
        axes[0][col].axis("off")

        for row, (pca_v, label) in enumerate([(pca_before, label_before),
                                               (pca_after, label_after)], start=1):
            pca_img = np.array(Image.fromarray(
                (pca_v[col] * 255).astype(np.uint8)
            ).resize(
                (tile_images[col].shape[1], tile_images[col].shape[0]),
                Image.BILINEAR,
            )) / 255.0
            axes[row][col].imshow(pca_img)
            if col == 0:
                axes[row][col].set_ylabel(label, fontsize=FONT_SIZE)
            axes[row][col].axis("off")

    fig.suptitle("PCA Feature Comparison", fontsize=FONT_SIZE + 2, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("PCA comparison saved to %s", output_path)


# =====================================================================
# Visualization 3: Overlap Alignment Heatmap
# =====================================================================


def visualize_overlap_alignment(
    features: torch.Tensor,
    tile_images: List[np.ndarray],
    tile_metas: List[dict],
    output_path: Path,
    overlap_ratio: float = 0.5,
    level_name: str = "resampler",
) -> None:
    """Cosine similarity heatmap between overlap strips of adjacent tiles.

    For each adjacent pair (tile_i, tile_{i+1}), extracts the right-k
    columns of tile_i and left-k columns of tile_{i+1}, computes
    patch-wise cosine similarity, and displays as a heatmap.

    Args:
        features: [V, S, D] (V includes global at index 0)
        tile_images: original tile images
        tile_metas: tile metadata
        output_path: save path
        overlap_ratio: fraction of overlap (0.5 = 50%)
        level_name: feature level label
    """
    V, S, D = features.shape
    h = w = int(math.sqrt(S))

    # Skip global view (index 0), work with tiles only
    has_global = V > 1 and len(tile_metas) > 0 and "Global" in tile_metas[0].get("label", "")
    start_idx = 1 if has_global else 0
    num_tiles = V - start_idx

    if num_tiles < 2:
        logger.warning("Need at least 2 tiles for overlap visualization")
        return

    k = max(1, int(w * overlap_ratio))
    n_pairs = num_tiles  # wrap-around: last tile overlaps with first

    fig, axes = plt.subplots(2, n_pairs, figsize=(n_pairs * 2.5, 5.0), squeeze=False)

    cos_sims_all = []
    im = None
    for pair_idx in range(n_pairs):
        i = start_idx + pair_idx
        j = start_idx + ((pair_idx + 1) % num_tiles)

        feat_i = features[i].reshape(h, w, D)  # [H, W, D]
        feat_j = features[j].reshape(h, w, D)

        # Right k columns of tile i, Left k columns of tile j
        strip_i = feat_i[:, -k:, :]  # [H, k, D]
        strip_j = feat_j[:, :k, :]   # [H, k, D]

        # Patch-wise cosine similarity
        strip_i_flat = F.normalize(strip_i.reshape(-1, D).float(), dim=-1)
        strip_j_flat = F.normalize(strip_j.reshape(-1, D).float(), dim=-1)
        cos_sim = (strip_i_flat * strip_j_flat).sum(dim=-1)  # [H*k]
        cos_sim_map = cos_sim.reshape(h, k).numpy()
        cos_sims_all.append(cos_sim_map.mean())

        # Row 0: tile pair images side by side
        if i < len(tile_images) and j < len(tile_images):
            canvas = np.concatenate([tile_images[i], tile_images[j]], axis=1)
            axes[0][pair_idx].imshow(canvas)
            # Draw overlap region indicator
            img_w = tile_images[i].shape[1]
            overlap_px = int(img_w * overlap_ratio)
            rect_x = img_w - overlap_px
            rect = plt.Rectangle(
                (rect_x, 0), overlap_px * 2, canvas.shape[0],
                linewidth=1.5, edgecolor="lime", facecolor="none", linestyle="--",
            )
            axes[0][pair_idx].add_patch(rect)
        axes[0][pair_idx].set_title(
            f"{tile_metas[i]['label']} ↔ {tile_metas[j]['label']}",
            fontsize=FONT_SIZE - 1,
        )
        axes[0][pair_idx].axis("off")

        # Row 1: cosine similarity heatmap
        im = axes[1][pair_idx].imshow(
            cos_sim_map, cmap=CMAP_OVERLAP, vmin=-1, vmax=1,
            aspect="auto", interpolation="nearest",
        )
        mean_sim = cos_sim_map.mean()
        axes[1][pair_idx].set_title(f"μ={mean_sim:.3f}", fontsize=FONT_SIZE)
        axes[1][pair_idx].set_xlabel("Overlap col", fontsize=FONT_SIZE - 1)
        if pair_idx == 0:
            axes[1][pair_idx].set_ylabel("Row", fontsize=FONT_SIZE - 1)

    fig.colorbar(im, ax=axes[1].tolist(), shrink=0.8, label="Cosine Similarity")
    overall_mean = np.mean(cos_sims_all)
    fig.suptitle(
        f"Overlap Alignment ({level_name}) — Mean cosine sim: {overall_mean:.3f}",
        fontsize=FONT_SIZE + 2, y=1.01,
    )
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Overlap alignment saved to %s (mean cos_sim=%.3f)", output_path, overall_mean)


# =====================================================================
# Visualization 4: Feature Similarity Map
# =====================================================================


def visualize_similarity(
    features: torch.Tensor,
    tile_images: List[np.ndarray],
    tile_metas: List[dict],
    output_path: Path,
    reference_points: Optional[List[Tuple[int, int, int]]] = None,
    level_name: str = "resampler",
) -> None:
    """Feature similarity heatmap: pick reference patches, show cosine
    similarity to all patches across all views.

    Default reference points sample from: center of global, center of
    tile 1, overlap region of tile 1, and a corner of tile 4.

    Args:
        features: [V, S, D]
        tile_images: original tile images
        tile_metas: metadata
        output_path: save path
        reference_points: list of (view_idx, row, col) tuples
        level_name: feature level label
    """
    V, S, D = features.shape
    h = w = int(math.sqrt(S))

    if reference_points is None:
        reference_points = [
            (0, h // 2, w // 2),          # center of global view
            (1, h // 2, w // 2),          # center of tile 1
            (1, h // 2, w - 2),           # right edge of tile 1 (overlap)
            (min(4, V - 1), h // 2, w // 2),  # center of tile 4
        ]
        # Filter valid points
        reference_points = [(v, r, c) for v, r, c in reference_points if v < V]

    n_refs = len(reference_points)
    fig, axes = plt.subplots(
        n_refs + 1, V, figsize=(V * 2.0, (n_refs + 1) * 2.0), squeeze=False,
    )
    im = None

    # Row 0: original tiles
    for col in range(V):
        axes[0][col].imshow(tile_images[col])
        axes[0][col].set_title(tile_metas[col]["label"], fontsize=FONT_SIZE)
        axes[0][col].axis("off")

    # Normalize features for cosine similarity
    feats_norm = F.normalize(features.float().reshape(V, h, w, D), dim=-1)

    for ref_idx, (ref_v, ref_r, ref_c) in enumerate(reference_points):
        ref_feat = feats_norm[ref_v, ref_r, ref_c]  # [D]

        for col in range(V):
            sim = (feats_norm[col] * ref_feat).sum(dim=-1)  # [h, w]
            sim_np = sim.numpy()

            # Upsample for overlay
            sim_up = np.array(Image.fromarray(
                ((sim_np + 1) / 2 * 255).astype(np.uint8)
            ).resize(
                (tile_images[col].shape[1], tile_images[col].shape[0]),
                Image.BILINEAR,
            )) / 255.0 * 2 - 1  # back to [-1, 1]

            ax = axes[ref_idx + 1][col]
            ax.imshow(tile_images[col], alpha=0.3)
            im = ax.imshow(sim_up, cmap=CMAP_SIMILARITY, vmin=-1, vmax=1,
                           alpha=0.7, interpolation="bilinear",
                           extent=[0, tile_images[col].shape[1],
                                   tile_images[col].shape[0], 0])

            # Mark reference point on its own view
            if col == ref_v:
                scale_x = tile_images[col].shape[1] / w
                scale_y = tile_images[col].shape[0] / h
                ax.plot(ref_c * scale_x + scale_x / 2,
                        ref_r * scale_y + scale_y / 2,
                        "x", color="lime", markersize=8, markeredgewidth=2)

            if col == 0:
                ax.set_ylabel(
                    f"Ref: V{ref_v}({ref_r},{ref_c})",
                    fontsize=FONT_SIZE - 1,
                )
            ax.axis("off")

    fig.colorbar(im, ax=axes[-1].tolist(), shrink=0.6, label="Cosine Similarity")
    fig.suptitle(
        f"Feature Similarity Map ({level_name})",
        fontsize=FONT_SIZE + 2, y=1.01,
    )
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Similarity map saved to %s", output_path)


# =====================================================================
# Panoramic stitch for paper figures
# =====================================================================


def stitch_panoramic_strip(
    features: torch.Tensor,
    tile_images: List[np.ndarray],
    output_path: Path,
    overlap_ratio: float = 0.5,
    level_name: str = "resampler",
) -> None:
    """Stitch PCA features into a single panoramic strip (tiles only, no global).

    Creates a horizontally stitched PCA feature panorama with overlap
    regions removed — directly usable as a paper figure.
    """
    V, S, D = features.shape
    h = w = int(math.sqrt(S))

    has_global = V > 1
    start_idx = 1 if has_global else 0
    num_tiles = V - start_idx

    if num_tiles < 2:
        return

    # PCA on tile features only
    tile_feats = features[start_idx:].reshape(-1, D).numpy()
    pca = PCA(n_components=3)
    pca_rgb = pca.fit_transform(tile_feats)
    for c in range(3):
        col_vals = pca_rgb[:, c]
        pca_rgb[:, c] = (col_vals - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-8)
    pca_views = pca_rgb.reshape(num_tiles, h, w, 3)

    # Stitch with overlap removal (same as model's stride_views)
    k = max(1, int(w * overlap_ratio))
    parts = [pca_views[0]]
    for i in range(1, num_tiles):
        parts.append(pca_views[i][:, k:, :])
    stitched = np.concatenate(parts, axis=1)  # [h, W_total, 3]

    # Similarly stitch original tile images
    tile_h, tile_w = tile_images[start_idx].shape[:2]
    k_px = int(tile_w * overlap_ratio)
    img_parts = [tile_images[start_idx]]
    for i in range(start_idx + 1, V):
        img_parts.append(tile_images[i][:, k_px:, :])
    stitched_img = np.concatenate(img_parts, axis=1)

    # Upsample PCA strip to match image height
    pca_up = np.array(Image.fromarray(
        (stitched * 255).astype(np.uint8)
    ).resize(
        (stitched_img.shape[1], stitched_img.shape[0]),
        Image.BILINEAR,
    )) / 255.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 3.5))

    axes[0].imshow(stitched_img)
    axes[0].set_title("Panoramic Tiles (stitched)", fontsize=FONT_SIZE + 1)
    axes[0].axis("off")

    axes[1].imshow(pca_up)
    var_explained = pca.explained_variance_ratio_[:3].sum() * 100
    axes[1].set_title(
        f"PCA Feature Map ({level_name}) — {var_explained:.1f}% variance",
        fontsize=FONT_SIZE + 1,
    )
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Panoramic strip saved to %s", output_path)


# =====================================================================
# Model loading
# =====================================================================


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load a CORA model from checkpoint."""
    from cora.config.manager import ConfigManager
    from cora.training.module import PanoramaTrainingModule

    ckpt_path = Path(checkpoint_path)
    config = ConfigManager.auto_detect_config(ckpt_path)
    if config is None:
        raise FileNotFoundError(f"Config not found near checkpoint: {ckpt_path}")

    module = PanoramaTrainingModule.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        config=config,
        stage="finetune",
        map_location="cpu",
    )
    model = module.model
    model.to(device)
    model.eval()
    return model, config


# =====================================================================
# CLI
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DINO-style feature visualization for CORA panoramic VLM",
    )
    parser.add_argument("--image", "-i", required=True, help="Path to ERP panorama image")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="Path to CORA checkpoint (for resampler/vicreg features)")
    parser.add_argument("--checkpoint-baseline", default=None,
                        help="Baseline checkpoint for comparison (e.g. before VICReg)")
    parser.add_argument("--mode", "-m", default="all",
                        choices=["attention", "pca", "overlap", "similarity", "stitch", "all"],
                        help="Visualization mode")
    parser.add_argument("--output-dir", "-o", default="outputs/visualizations",
                        help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Transformer layer index for attention (-1 = last)")
    parser.add_argument("--level", default="resampler",
                        choices=["vision", "resampler", "vicreg"],
                        help="Feature level for PCA/similarity/overlap")
    parser.add_argument("--num-heads", type=int, default=6,
                        help="Number of attention heads to display")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(args.image).stem

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    # Load model if checkpoint provided
    model = None
    config = None
    if args.checkpoint:
        logger.info("Loading model from %s", args.checkpoint)
        model, config = load_model(args.checkpoint, device)

    # Load image and generate tiles
    logger.info("Processing image: %s", args.image)
    original_img, pixel_values, tile_metas = load_image_and_tiles(
        args.image, config,
    )
    pixel_values = pixel_values.to(device)
    tile_images = get_tile_images(pixel_values)

    modes = [args.mode] if args.mode != "all" else [
        "attention", "pca", "overlap", "similarity", "stitch",
    ]

    for mode in modes:
        logger.info("Generating %s visualization...", mode)

        if mode == "attention":
            vision_encoder = model.vision_encoder if model else None
            if vision_encoder is None:
                # Load raw SigLIP2 for attention extraction
                from cora.model.vision_encoder import VisionBackbone
                vision_encoder = VisionBackbone(
                    "google/siglip2-so400m-patch16-256",
                    backbone_type="hf",
                ).to(device)

            attn = extract_attention_maps(vision_encoder, pixel_values, args.layer)
            visualize_attention(
                attn, tile_images, tile_metas,
                output_dir / f"{image_stem}_attention.png",
                num_heads_to_show=args.num_heads,
            )

        elif mode == "pca":
            if model is None:
                logger.warning("PCA requires --checkpoint for resampler features. Using vision-level.")
                from cora.model.vision_encoder import VisionBackbone
                ve = VisionBackbone("google/siglip2-so400m-patch16-256", backbone_type="hf").to(device)
                flat = pixel_values.view(-1, *pixel_values.shape[2:])
                vout = ve(flat)
                feats = vout["vision_features"].cpu()
                level_name = "vision (raw SigLIP2)"
            else:
                feats = extract_patch_features(model, pixel_values, args.level)
                level_name = args.level

            if args.checkpoint_baseline:
                model_bl, _ = load_model(args.checkpoint_baseline, device)
                feats_bl = extract_patch_features(model_bl, pixel_values, args.level)
                visualize_pca_comparison(
                    feats_bl, feats, tile_images, tile_metas,
                    output_dir / f"{image_stem}_pca_comparison.png",
                    label_before="Baseline",
                    label_after="Trained",
                )
            else:
                visualize_pca(
                    feats, tile_images, tile_metas,
                    output_dir / f"{image_stem}_pca_{args.level}.png",
                    level_name=level_name,
                )

        elif mode == "overlap":
            if model is None:
                logger.error("Overlap visualization requires --checkpoint")
                continue
            feats = extract_patch_features(model, pixel_values, args.level)
            overlap_ratio = config.image_processing.overlap_ratio if config else 0.5
            visualize_overlap_alignment(
                feats, tile_images, tile_metas,
                output_dir / f"{image_stem}_overlap_{args.level}.png",
                overlap_ratio=overlap_ratio,
                level_name=args.level,
            )

        elif mode == "similarity":
            if model is None:
                logger.error("Similarity map requires --checkpoint")
                continue
            feats = extract_patch_features(model, pixel_values, args.level)
            visualize_similarity(
                feats, tile_images, tile_metas,
                output_dir / f"{image_stem}_similarity_{args.level}.png",
                level_name=args.level,
            )

        elif mode == "stitch":
            if model is None:
                logger.error("Panoramic stitch requires --checkpoint")
                continue
            feats = extract_patch_features(model, pixel_values, args.level)
            overlap_ratio = config.image_processing.overlap_ratio if config else 0.5
            stitch_panoramic_strip(
                feats, tile_images,
                output_dir / f"{image_stem}_stitch_{args.level}.png",
                overlap_ratio=overlap_ratio,
                level_name=args.level,
            )

    logger.info("All visualizations saved to %s", output_dir)


if __name__ == "__main__":
    main()
