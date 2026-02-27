# src/cora/ — CORA Package

Core Python package. All imports start with `from cora.<module> import ...`.

## MODULE MAP

| Module | Key Classes | Role |
|--------|------------|------|
| `model/vlm.py` | `PanoramaVLM` | Main orchestrator — vision + resampler + projector + LLM |
| `model/vision_encoder.py` | `VisionBackbone` | Wraps SigLIP/CLIP/DINOv2 via HF AutoModel |
| `model/language_model.py` | `LanguageModel` | Wraps Qwen/Llama/Gemma + LoRA setup |
| `model/resampler/` | `ResamplerModule` | Factory for 7 resampler types |
| `model/projectors.py` | `PanoramaProjector`, `VICRegProjector` | Vision→LLM dim projection |
| `model/language_fusion.py` | `LanguageFusion` | Prepends vision tokens to text sequence |
| `model/positional.py` | `PanoramaPositionalEncoding` | Yaw-continuous sinusoidal PE for panorama tiles |
| `training/trainer.py` | `CORATrainer` | 3-stage pipeline orchestrator |
| `training/module.py` | `PanoramaTrainingModule` | Lightning module with stage-aware freezing |
| `training/losses.py` | `VICRegLoss`, `PanoContrastiveLoss`, `DenseCLLoss` | Self-supervised + contrastive losses |
| `training/callbacks.py` | `CUDACacheCleanupCallback`, `MetadataCallback` | OOM prevention, config snapshot |
| `processors/images.py` | `PanoramaImageProcessor` | 7 crop strategies for 360° images |
| `processors/text.py` | `UniversalTextFormatter` | Chat template formatting + Qwen3 fix |
| `processors/processor.py` | `PanoramaProcessor` | Unified image+text preprocessing |
| `data/dataset.py` | `PanoramaDataset` | CSV-based dataset with retry logic |
| `data/datamodule.py` | `PanoramaDataModule` | Lightning DataModule wrapper |
| `config/schema.py` | `CORAConfig`, `ModelConfig`, `StageConfig` | Pydantic schemas |
| `config/manager.py` | `ConfigManager` | YAML loading + CLI override |
| `evaluation/metrics.py` | `CORAEvaluator` | BLEU/CIDEr/METEOR/ROUGE-L/SPICE |
| `inference/generator.py` | `PanoramaGenerator` | Checkpoint→text generation |
| `baseline/finetune.py` | `BaselineTrainer` | HF Trainer LoRA path for commercial VLMs |
| `baseline/models.py` | Model wrappers | InternVL monkey-patch, Gemma3, BLIP2 support |

## CODING RULES (THIS PACKAGE)

- **All functions must have type hints** — mypy strict mode is enforced
- **New modules**: Add `logger = logging.getLogger(__name__)` at top
- **Config access**: Always through `CORAConfig` Pydantic model, never raw dict
- **Optional deps**: `try/except ImportError` → set to `None`, check at runtime
- **Error fallback in Lightning steps**: Return `torch.tensor(0.0, device=self.device, requires_grad=True)` — NEVER `torch.zeros([])`
- **Pydantic models**: Always include `model_config = {"extra": "allow"}`

## ADDING NEW COMPONENTS

| What | Where | How |
|------|-------|-----|
| New resampler | `model/resampler/` | Create file → register in `resamplers.py` `create_resampler()` |
| New crop strategy | `processors/images.py` | Add to `VALID_STRATEGIES` + implement in `__call__` |
| New loss function | `training/losses.py` | Add class → wire in `training/module.py` `_compute_vision_loss()` |
| New config field | `config/schema.py` | Add to appropriate Pydantic model + update YAML configs |
| New baseline model | `baseline/models.py` | Add wrapper + register in `MODEL_REGISTRY` |
| New eval metric | `evaluation/metrics.py` | Add to `CORAEvaluator.evaluate()` |

## DATA FLOW (TRAINING)

```
YAML config → ConfigManager.load()
    → CORAConfig (validated)
    → CORATrainer.run()
        → for stage in ["vision", "resampler", "finetune"]:
            PanoramaDataModule(stage_config)
            PanoramaTrainingModule(model, stage)
            pl.Trainer.fit()
            → save stage_state.json
```

## DATA FLOW (INFERENCE)

```
PanoramaGenerator.from_checkpoint(ckpt_path)
    → PanoramaVLM (loaded)
    → PanoramaImageProcessor(crop_strategy)
    → processor(image) → pixel_values [B, V, C, H, W]
    → model.generate(pixel_values, input_ids)
    → tokenizer.decode() → text
```
