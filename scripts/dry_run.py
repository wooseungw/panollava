#!/usr/bin/env python3
"""Dry-run: validate CORA data pipeline + model forward pass (1 batch)."""

import sys
import logging
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("dry_run")

CONFIG_PATH = "configs/cora/default.yaml"
DEVICE = "cuda:0"


def step(name: str):
    logger.info("=" * 60)
    logger.info("STEP: %s", name)
    logger.info("=" * 60)


def main() -> int:
    errors: list[str] = []

    # ── 1. Config ──────────────────────────────────────────────
    step("Load config")
    try:
        from cora.config.manager import ConfigManager
        config = ConfigManager.load(CONFIG_PATH)
        logger.info("Config loaded: experiment=%s", config.experiment.get("name"))
        logger.info("  vision  = %s", config.models.vision_name)
        logger.info("  LLM     = %s", config.models.language_model_name)
        logger.info("  resampler = %s (latent=%d)", config.models.resampler_type, config.models.latent_dimension)
    except Exception as e:
        logger.error("Config load FAILED: %s", e)
        traceback.print_exc()
        return 1

    # ── 2. Processor (image + text) ────────────────────────────
    step("Build processor")
    try:
        from cora.processors.images import PanoramaImageProcessor
        from cora.processors.processor import PanoramaProcessor
        from transformers import AutoTokenizer

        img_cfg = config.image_processing
        image_size = img_cfg.image_size
        if image_size is None:
            vname = config.models.vision_name.lower()
            for tag, sz in [("-256", [256, 256]), ("-384", [384, 384]), ("-224", [224, 224])]:
                if tag in vname:
                    image_size = sz
                    break
            else:
                image_size = [224, 224]

        img_proc = PanoramaImageProcessor(
            image_size=image_size,
            crop_strategy=img_cfg.crop_strategy,
            fov_deg=img_cfg.fov_deg,
            overlap_ratio=img_cfg.overlap_ratio,
            normalize=img_cfg.normalize,
            use_vision_processor=img_cfg.use_vision_processor,
            vision_model_name=config.models.vision_name,
            anyres_max_patches=img_cfg.anyres_max_patches,
        )
        logger.info("  ImageProcessor: strategy=%s, size=%s", img_cfg.crop_strategy, image_size)

        tokenizer = AutoTokenizer.from_pretrained(
            config.models.language_model_name, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vision_token = "<|vision|>"
        existing = [str(t) for t in tokenizer.additional_special_tokens]
        if vision_token not in existing:
            tokenizer.add_special_tokens({"additional_special_tokens": existing + [vision_token]})

        system_prompt = getattr(config.training, "system_msg", None)
        processor = PanoramaProcessor(img_proc, tokenizer, system_prompt=system_prompt)
        logger.info("  Processor built OK (vocab_size=%d)", len(tokenizer))
    except Exception as e:
        logger.error("Processor build FAILED: %s", e)
        traceback.print_exc()
        return 1

    # ── 3. Dataset ─────────────────────────────────────────────
    step("Build dataset + fetch 1 sample")
    try:
        from cora.data.dataset import PanoramaDataset, custom_collate_fn

        stage_cfg = config.get_stage_config("resampler")
        stage_data = stage_cfg.data
        if stage_data and stage_data.csv_train:
            train_csv = stage_data.csv_train
        else:
            train_csv = config.data.get("train", config.data.get("train_csv", "train.csv"))

        logger.info("  train_csv = %s", train_csv)
        ds = PanoramaDataset(
            csv_path=train_csv,
            processor=processor,
            mode="train",
            max_text_length=128,
        )
        logger.info("  Dataset: %d samples", len(ds))

        sample = ds[0]
        logger.info("  Sample keys: %s", list(sample.keys()))
        if "pixel_values" in sample and sample["pixel_values"] is not None:
            logger.info("  pixel_values shape: %s", sample["pixel_values"].shape)
        logger.info("  input_ids shape: %s", sample["input_ids"].shape)
        logger.info("  labels shape:    %s", sample["labels"].shape)
    except Exception as e:
        errors.append(f"Dataset: {e}")
        logger.error("Dataset FAILED: %s", e)
        traceback.print_exc()

    # ── 4. Collate ─────────────────────────────────────────────
    step("Collate mini-batch (2 samples)")
    try:
        s0 = ds[0]
        s1 = ds[1 % len(ds)]
        batch = custom_collate_fn([s0, s1])
        logger.info("  Collated keys: %s", list(batch.keys()))
        for k, v in batch.items():
            if hasattr(v, "shape"):
                logger.info("    %s: %s", k, v.shape)
    except Exception as e:
        errors.append(f"Collate: {e}")
        logger.error("Collate FAILED: %s", e)
        traceback.print_exc()

    # ── 5. Model build ─────────────────────────────────────────
    step("Build PanoramaTrainingModule (resampler stage)")
    import torch
    try:
        from cora.training.module import PanoramaTrainingModule
        module = PanoramaTrainingModule(
            config=config,
            stage="resampler",
            vision_trainable_blocks=0,
        )
        module = module.to(DEVICE)
        module.eval()
        n_params = sum(p.numel() for p in module.parameters())
        n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info("  Total params : %d (%.1fM)", n_params, n_params / 1e6)
        logger.info("  Trainable    : %d (%.1fM)", n_train, n_train / 1e6)
    except Exception as e:
        errors.append(f"Model build: {e}")
        logger.error("Model build FAILED: %s", e)
        traceback.print_exc()
        # Can't continue to forward pass
        return _report(errors)

    # ── 6. Forward pass ────────────────────────────────────────
    step("Forward pass (1 batch)")
    try:
        device_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(DEVICE)
            else:
                device_batch[k] = v

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            loss = module.training_step(device_batch, batch_idx=0)

        logger.info("  Loss: %.4f", loss.item())
    except Exception as e:
        errors.append(f"Forward pass: {e}")
        logger.error("Forward pass FAILED: %s", e)
        traceback.print_exc()

    # ── 7. Cleanup ─────────────────────────────────────────────
    del module
    torch.cuda.empty_cache()

    return _report(errors)


def _report(errors: list[str]) -> int:
    step("SUMMARY")
    if errors:
        logger.error("FAILED — %d error(s):", len(errors))
        for i, e in enumerate(errors, 1):
            logger.error("  %d. %s", i, e)
        return 1
    logger.info("ALL STEPS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
