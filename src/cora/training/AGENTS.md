# src/cora/training/ — 3-Stage Training Pipeline

## OVERVIEW

PyTorch Lightning-based training with 3-stage progressive curriculum. `CORATrainer` orchestrates stages; `PanoramaTrainingModule` is the Lightning module.

## FILES

| File | Key Classes | Role |
|------|------------|------|
| `trainer.py` | `CORATrainer` | Stage orchestrator: config→datamodule→module→pl.Trainer→fit |
| `module.py` | `PanoramaTrainingModule` | Lightning module: forward, loss, freeze/unfreeze per stage |
| `losses.py` | `VICRegLoss`, `PanoContrastiveLoss`, `DenseCLLoss` | Self-supervised losses for vision stage |
| `callbacks.py` | `CUDACacheCleanupCallback`, `MetadataCallback` | OOM prevention, config snapshot on save |

## 3-STAGE PIPELINE

```
CORATrainer.run()
├── Stage 1: "vision"     — VICReg/Contrastive/DenseCL loss
│   Trainable: resampler + VICReg projector (+ optional top-N vision blocks)
│   Frozen: LLM (entirely), most of vision encoder
│
├── Stage 2: "resampler"  — LM loss + weighted VICReg regularization
│   Trainable: resampler + panorama projector
│   Frozen: LLM, vision encoder, VICReg projector
│
└── Stage 3: "finetune"   — LM loss only
    Trainable: resampler + projector + LLM (via LoRA)
    Frozen: vision encoder
```

## STAGE STATE PERSISTENCE

```
runs/{experiment}/{YYYYMMDD_NNN}/stage_state.json
{
  "completed_stages": ["vision", "resampler"],
  "current_stage": "finetune",
  "vision": {"checkpoint": "...", "best_metric": 0.42},
  "resampler": {"checkpoint": "...", "best_metric": 1.23}
}
```

Resume: `python scripts/train.py --config ... --resume auto`

## LOSS FUNCTIONS

| Loss | Config key | Stage | Description |
|------|-----------|-------|-------------|
| `VICRegLoss` | `vision_loss_type: "vicreg"` | vision/resampler | Variance + Invariance + Covariance on overlapping view pairs |
| `PanoContrastiveLoss` | `vision_loss_type: "contrastive"` | vision/resampler | Symmetric InfoNCE with dropout-based two views |
| `DenseCLLoss` | `vision_loss_type: "densecl"` | vision/resampler | Single-view overlap InfoNCE (simplest) |
| LM loss | automatic | resampler/finetune | `model(..., labels=labels).loss` (autoregressive CE) |

## ANTI-PATTERNS (TRAINING CODE)

- **Step return**: ALWAYS `torch.tensor(0.0, device=self.device, requires_grad=True)` for fallback. Never `torch.zeros([])`.
- **Gradient checkpointing**: Enable for finetune stage to prevent OOM with LoRA activation storage.
- **`on_validation_epoch_start`**: Must call `torch.cuda.empty_cache()` + `gc.collect()` — validation + checkpoint save spikes memory.
- **`CUDACacheCleanupCallback`**: Register for LoRA runs >7000 steps. Config: `training.cache_cleanup_interval`.
- **Autobatch**: `fraction=0.85` leaves ~3.6GB headroom on 24GB GPU — fragile with LoRA. Prefer explicit `batch_size`.
- **`num_workers`**: Reduce to 4 if system RAM < 32GB. Fork-based workers accumulate CoW pages.
- **Exception handling in callbacks**: Use `except Exception` + `# noqa: BLE001` — training must never crash from a callback.
