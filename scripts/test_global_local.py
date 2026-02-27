#!/usr/bin/env python3
"""Test: global-local loss + VICReg tile-only fix."""

import sys
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("test_gl")

DEVICE = "cuda:0"
CONFIG_PATH = "configs/cora/default.yaml"


def main() -> int:
    from cora.config.manager import ConfigManager
    from cora.training.losses import GlobalLocalLoss, VICRegLoss

    # ── Test 1: GlobalLocalLoss standalone ──────────────────────
    logger.info("=" * 60)
    logger.info("TEST 1: GlobalLocalLoss standalone")
    logger.info("=" * 60)

    B, T, S, D = 2, 8, 64, 1024
    g = torch.randn(B, S, D, device=DEVICE)
    t = torch.randn(B * T, S, D, device=DEVICE)

    for loss_type in ("cosine", "mse"):
        gl = GlobalLocalLoss(loss_type=loss_type).to(DEVICE)
        val = gl(g, t, batch_size=B, num_tiles=T)
        logger.info("  %s loss = %.6f (grad_fn=%s)", loss_type, val.item(), val.grad_fn is not None)

    # ── Test 2: VICReg now receives tiles only (no global) ──────
    logger.info("=" * 60)
    logger.info("TEST 2: VICReg overlap with tiles only (8 views, not 9)")
    logger.info("=" * 60)

    vicreg = VICRegLoss(overlap_ratio=0.5, vicreg_mode="pairwise", debug=True).to(DEVICE)
    tile_feats = torch.randn(B * T, S, D, device=DEVICE)
    vicreg_loss = vicreg(tile_feats, B, T)
    logger.info("  VICReg loss = %.6f (grad_fn=%s)", vicreg_loss.item(), vicreg_loss.grad_fn is not None)

    # ── Test 3: Full model forward (vision stage) ───────────────
    logger.info("=" * 60)
    logger.info("TEST 3: Full model forward — vision stage")
    logger.info("=" * 60)

    config = ConfigManager.load(CONFIG_PATH)
    from cora.training.module import PanoramaTrainingModule

    module = PanoramaTrainingModule(config=config, stage="vision", vision_trainable_blocks=0)
    module = module.to(DEVICE)
    module.eval()

    from cora.data.dataset import PanoramaDataset, custom_collate_fn
    from cora.processors.images import PanoramaImageProcessor
    from cora.processors.processor import PanoramaProcessor
    from transformers import AutoTokenizer

    img_cfg = config.image_processing
    image_size = img_cfg.image_size or [256, 256]
    img_proc = PanoramaImageProcessor(
        image_size=image_size, crop_strategy=img_cfg.crop_strategy,
        fov_deg=img_cfg.fov_deg, overlap_ratio=img_cfg.overlap_ratio,
        normalize=img_cfg.normalize, use_vision_processor=img_cfg.use_vision_processor,
        vision_model_name=config.models.vision_name, anyres_max_patches=img_cfg.anyres_max_patches,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.models.language_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = PanoramaProcessor(img_proc, tokenizer)

    ds = PanoramaDataset(csv_path="train.csv", processor=processor, mode="train", max_text_length=32)
    batch = custom_collate_fn([ds[0], ds[1 % len(ds)]])

    device_batch = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = module.model(stage="vision", **device_batch)

    logger.info("  Output keys: %s", list(outputs.keys()))
    logger.info("  vicreg_features: %s", outputs["vicreg_features"].shape)
    logger.info("  global_features: %s", outputs["global_features"].shape if outputs.get("global_features") is not None else None)
    logger.info("  batch_size=%d, num_views=%d", outputs["batch_size"], outputs["num_views"])

    expected_tiles = img_proc.num_views - 1
    assert outputs["num_views"] == expected_tiles, (
        f"num_views should be {expected_tiles} (tiles only), got {outputs['num_views']}"
    )
    assert outputs["global_features"] is not None, "global_features should not be None"
    logger.info("  ✓ Global view separated, tiles only in vicreg_features")

    # ── Test 4: training_step end-to-end ────────────────────────
    logger.info("=" * 60)
    logger.info("TEST 4: training_step (gl_loss disabled by default)")
    logger.info("=" * 60)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = module.training_step(device_batch, batch_idx=0)
    logger.info("  Loss = %.6f", loss.item())
    logger.info("  ✓ training_step passed (global_local_loss_weight=0)")

    # ── Test 5: training_step with gl_loss enabled ──────────────
    logger.info("=" * 60)
    logger.info("TEST 5: training_step (gl_loss enabled, weight=1.0)")
    logger.info("=" * 60)

    module.gl_loss_weight = 1.0
    module.global_local_loss = GlobalLocalLoss(loss_type="cosine").to(DEVICE)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss_gl = module.training_step(device_batch, batch_idx=0)
    logger.info("  Loss = %.6f", loss_gl.item())
    logger.info("  ✓ training_step passed with global-local loss")

    del module
    torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
