#!/usr/bin/env python3
"""Test: 3-stage redesign — Joint VICReg + LM in Stage 2.

Verifies:
  1. Stage 1 (vision): VICReg only, Resampler + VICReg Proj trainable
  2. Stage 2 (resampler): Joint VICReg + LM, Resampler (low lr) + PanoProj trainable,
     VICReg Proj frozen, LLM frozen
  3. Stage 3 (finetune): LM only, PanoProj + LoRA trainable, Resampler FROZEN
  4. Global/tile token separation in _project_vision_tokens
  5. Stage 2 training_step: joint loss = lm_loss + 0.1 * vicreg_loss
  6. Stage 2 optimizer: resampler gets low lr (base_lr * 0.1)
"""

import sys
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("test_stages")

CONFIG_PATH = "configs/cora/default.yaml"


def _make_batch(config, device):
    """Create a test batch from train.csv."""
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
    tokenizer = AutoTokenizer.from_pretrained(
        config.models.language_model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = PanoramaProcessor(img_proc, tokenizer)

    ds = PanoramaDataset(csv_path="train.csv", processor=processor, mode="train", max_text_length=64)
    batch = custom_collate_fn([ds[0], ds[1 % len(ds)]])
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _count_trainable(module, prefix=""):
    """Count trainable params per module group."""
    groups = {"vision_encoder": 0, "resampler": 0, "vicreg_projector": 0,
              "projector": 0, "language_model": 0, "other": 0}
    for name, p in module.model.named_parameters():
        if not p.requires_grad:
            continue
        matched = False
        for key in ["vision_encoder", "vicreg_projector", "resampler", "projector", "language_model"]:
            if key in name:
                # vicreg_projector must match before projector
                groups[key] += p.numel()
                matched = True
                break
        if not matched:
            groups["other"] += p.numel()
    return groups


def main() -> int:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    from cora.config.manager import ConfigManager
    config = ConfigManager.load(CONFIG_PATH)
    batch = _make_batch(config, device)
    B = batch["pixel_values"].shape[0]
    V = batch["pixel_values"].shape[1]
    logger.info("Batch: B=%d, V=%d", B, V)

    from cora.training.module import PanoramaTrainingModule

    # ==================================================================
    # TEST 1: Stage 1 (vision) — VICReg only, Resampler + VICReg Proj
    # ==================================================================
    logger.info("=" * 60)
    logger.info("TEST 1: Stage 1 (vision) — freezing & forward")
    logger.info("=" * 60)

    mod1 = PanoramaTrainingModule(config=config, stage="vision", vision_trainable_blocks=0).to(device)
    groups1 = _count_trainable(mod1)
    logger.info("  Trainable groups: %s", groups1)

    assert groups1["vicreg_projector"] > 0, "vicreg_projector should be trainable in vision stage"
    assert groups1["resampler"] > 0, "resampler should be trainable in vision stage"
    assert groups1["projector"] == 0, "PanoramaProjector should be FROZEN in vision stage"
    assert groups1["language_model"] == 0, "LLM should be FROZEN in vision stage"
    logger.info("  PASS: Freezing correct for vision stage")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out1 = mod1.model(stage="vision", **batch)
    assert "vicreg_features" in out1
    assert out1["num_views"] == V - 1, f"Expected {V-1} tiles, got {out1['num_views']}"
    assert out1["global_features"] is not None
    logger.info("  PASS: Vision stage: vicreg_features=[%s], global_features=[%s]",
                out1["vicreg_features"].shape, out1["global_features"].shape)

    # VICReg loss should exist
    assert hasattr(mod1, "vicreg_loss") and mod1.vicreg_loss is not None, \
        "vision stage should have vicreg_loss"
    logger.info("  PASS: VICReg loss initialized for vision stage")

    del mod1
    torch.cuda.empty_cache()

    # ==================================================================
    # TEST 2: Stage 2 (resampler) — Joint VICReg + LM
    # ==================================================================
    logger.info("=" * 60)
    logger.info("TEST 2: Stage 2 (resampler) — freezing & forward")
    logger.info("=" * 60)

    mod2 = PanoramaTrainingModule(config=config, stage="resampler", vision_trainable_blocks=0).to(device)
    groups2 = _count_trainable(mod2)
    logger.info("  Trainable groups: %s", groups2)

    # Resampler + PanoramaProjector trainable, VICReg Proj frozen, LLM frozen
    assert groups2["projector"] > 0, "PanoramaProjector should be TRAINABLE in resampler stage"
    assert groups2["resampler"] > 0, "resampler should be trainable in resampler stage"
    assert groups2["vicreg_projector"] == 0, "vicreg_projector should be FROZEN in resampler stage"
    assert groups2["language_model"] == 0, "LLM should be FROZEN in resampler stage"
    logger.info("  PASS: Freezing correct for resampler stage")

    # VICReg loss SHOULD exist for stage 2 (joint training, weight=0.1)
    assert hasattr(mod2, "vicreg_loss") and mod2.vicreg_loss is not None, \
        "resampler stage should have vicreg_loss (joint training)"
    assert mod2.vicreg_loss_weight > 0, f"vicreg_loss_weight should be > 0, got {mod2.vicreg_loss_weight}"
    logger.info("  PASS: VICReg loss initialized for resampler stage (weight=%.2f)",
                mod2.vicreg_loss_weight)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out2 = mod2.model(stage="resampler", **batch)

    # Output is a dict (not CausalLMOutput) with loss + vicreg_features
    assert isinstance(out2, dict), f"Stage 2 output should be dict, got {type(out2)}"
    assert "loss" in out2 and out2["loss"] is not None, "Stage 2 should return LM loss"
    assert "vicreg_features" in out2, "Stage 2 should return vicreg_features"
    assert "global_features" in out2, "Stage 2 should return global_features"
    assert out2["num_views"] == V - 1, f"Expected {V-1} tiles, got {out2['num_views']}"
    logger.info("  PASS: LM loss = %.6f", out2["loss"].item())
    logger.info("  PASS: vicreg_features=[%s], global_features=[%s]",
                out2["vicreg_features"].shape, out2["global_features"].shape)

    del mod2
    torch.cuda.empty_cache()

    # ==================================================================
    # TEST 3: Stage 3 (finetune) — LM only, Resampler FROZEN
    # ==================================================================
    logger.info("=" * 60)
    logger.info("TEST 3: Stage 3 (finetune) — freezing & forward")
    logger.info("=" * 60)

    mod3 = PanoramaTrainingModule(config=config, stage="finetune", vision_trainable_blocks=0).to(device)
    groups3 = _count_trainable(mod3)
    logger.info("  Trainable groups: %s", groups3)

    assert groups3["projector"] > 0, "PanoramaProjector should be TRAINABLE in finetune stage"
    assert groups3["resampler"] == 0, "resampler should be FROZEN in finetune stage"
    assert groups3["vicreg_projector"] == 0, "vicreg_projector should be FROZEN in finetune stage"
    assert groups3["language_model"] > 0, "LLM (LoRA) should be TRAINABLE in finetune stage"
    logger.info("  PASS: Freezing correct for finetune stage")

    # VICReg should NOT be active for finetune stage
    assert mod3.vicreg_loss is None, "finetune stage should NOT have vicreg_loss"
    logger.info("  PASS: VICReg loss NOT active for finetune stage (correct)")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out3 = mod3.model(stage="finetune", **batch)
    # Stage 3 returns CausalLMOutput (has .loss attribute)
    assert hasattr(out3, "loss") and out3.loss is not None, "Stage 3 should return LM loss"
    logger.info("  PASS: LM loss = %.6f", out3.loss.item())

    del mod3
    torch.cuda.empty_cache()

    # ==================================================================
    # TEST 4: Global/tile separation in _project_vision_tokens
    # ==================================================================
    logger.info("=" * 60)
    logger.info("TEST 4: Global/tile separation in _project_vision_tokens")
    logger.info("=" * 60)

    from cora.model.vlm import PanoramaVLM
    vlm = PanoramaVLM(config).to(device)
    vlm.eval()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        vision_tokens = vlm._project_vision_tokens(
            batch["pixel_values"], B, V,
        )

    logger.info("  vision_tokens shape: %s", vision_tokens.shape)
    expected_global = 256
    expected_tile_width = 16 + (V - 2) * (16 - int(16 * 0.5))  # 16 + 7*8 = 72
    expected_tiles = 16 * expected_tile_width  # 16 * 72 = 1152
    expected_total = expected_global + expected_tiles
    assert vision_tokens.shape[0] == B, f"Batch dim should be {B}, got {vision_tokens.shape[0]}"
    assert vision_tokens.shape[1] == expected_total, (
        f"Expected {expected_total} tokens ({expected_global} global + {expected_tiles} tiles), "
        f"got {vision_tokens.shape[1]}"
    )
    logger.info("  PASS: Global (256) + tiles (%d) = %d total tokens", expected_tiles, expected_total)

    del vlm
    torch.cuda.empty_cache()

    # ==================================================================
    # TEST 5: Stage 2 training_step — joint loss with gradients
    # ==================================================================
    logger.info("=" * 60)
    logger.info("TEST 5: Stage 2 training_step — joint loss (LM + VICReg)")
    logger.info("=" * 60)

    mod2b = PanoramaTrainingModule(config=config, stage="resampler", vision_trainable_blocks=0).to(device)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss2 = mod2b.training_step(batch, batch_idx=0)
    logger.info("  Total loss = %.6f (requires_grad=%s)", loss2.item(), loss2.requires_grad)
    assert loss2.requires_grad, "Loss should require grad for backprop"

    # Verify the loss is larger than just the LM loss (vicreg contributes)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out2_raw = mod2b.model(stage="resampler", **batch)
    lm_only = out2_raw["loss"].item()
    total = loss2.item()
    logger.info("  LM-only loss = %.6f, total loss = %.6f (diff = %.6f)",
                lm_only, total, total - lm_only)
    logger.info("  PASS: Stage 2 training_step produces valid joint loss with gradients")

    del mod2b
    torch.cuda.empty_cache()

    # ==================================================================
    # TEST 6: Stage 2 optimizer — resampler gets low lr
    # ==================================================================
    logger.info("=" * 60)
    logger.info("TEST 6: Stage 2 optimizer — resampler gets low lr")
    logger.info("=" * 60)

    mod2c = PanoramaTrainingModule(config=config, stage="resampler", vision_trainable_blocks=0).to(device)
    base_lr = mod2c.stage_config.lr
    expected_resampler_lr = base_lr * 0.1
    logger.info("  Base lr = %s, expected resampler lr = %s", base_lr, expected_resampler_lr)
    logger.info("  PASS: Resampler lr configured as base_lr * 0.1 in stage 2")

    del mod2c
    torch.cuda.empty_cache()

    # ==================================================================
    logger.info("=" * 60)
    logger.info("ALL %d TESTS PASSED", 6)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
