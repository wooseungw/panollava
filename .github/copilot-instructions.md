# PanoLLaVA AI Agent Instructions

## Project Overview
**PanoLLaVA** is a panoramic Vision-Language Model (VLM) that combines vision encoders (SigLIP, CLIP, DINOv2) with large language models (Qwen, Llama, Gemma) to understand 360° panoramic images. It uses a 3-stage progressive training approach with VICReg self-supervised learning.

## Architecture & Data Flow

### Core Pipeline (3 Stages)
1. **Vision Stage**: VICReg overlap loss trains Resampler to learn spatial consistency across panoramic views
   - Vision Encoder (frozen) → Resampler (trainable) → VICReg Projector (trainable)
   - Loss: `25×invariance + 25×variance + 1×covariance` on overlapping patches
   
2. **Resampler Stage**: Align vision tokens with language model space
   - Vision features → Resampler → Language Model (autoregressive loss on text)
   
3. **Finetune Stage**: End-to-end fine-tuning with optional LoRA

### Module Organization (RECENT REFACTORING)
```
src/panovlm/
├── processors/        # Image/text processing (NEW: moved from data/)
│   ├── image.py              # PanoramaImageProcessor (multi-view strategies)
│   ├── pano_llava_processor.py
│   ├── universal_text_formatter.py
│   ├── vision.py
│   ├── anyres_e2p.py         # ERP→pinhole tile generation
│   └── anyres_integration.py # VICReg loss for AnyRes tiles
├── models/
│   ├── model.py       # PanoramaVLM (main model class)
│   ├── vision/        # VisionBackbone, ResamplerModule
│   └── language_fusion.py
├── losses/            # VicRegLoss, vicreg_overlap
├── config/            # Config, ModelConfig, StageConfig
├── dataset.py     # BaseChatPanoDataset, VLMDataModule
└
```

**CRITICAL**: After recent refactoring, imports changed:
- ✅ `from panovlm.processors import PanoramaImageProcessor, UniversalTextFormatter`
- ✅ `from panovlm.data import VLMDataModule, ChatPanoDataset`
- ❌ OLD: `from panovlm.data.image import ...` (removed)

### Image Processing Strategies
`PanoramaImageProcessor` supports multiple crop strategies (set in config):
- `e2p`: Equirectangular-to-Perspective projection (90° FOV, central crop)
- `sliding_window`: Horizontal sliding with overlap
- `cubemap`: 4-face cube projection (F/R/B/L)
- `anyres_e2p`: **AnyRes-style ERP tiling** with yaw/pitch grid (NEW)
- `anyres`, `anyres_max`: Grid-based patches
- `resize`: Simple resize (baseline)

**AnyRes ERP** (`crop_strategy: anyres_e2p`):
- Generates global view + tiles covering yaw (360°) × pitch (configurable range)
- Supports closed-loop yaw division for seamless 360° coverage
- See `processors/anyres_e2p.py::build_anyres_from_erp()` for implementation

## Configuration System

### YAML-First Approach (configs/default.yaml)
All training parameters are in YAML. CLI args are minimal (just `--config`, `--stage`, `--resume`).

**Key Sections**:
```yaml
experiment:
  name: "ADDDATA_SQ3_1_latent768_PE"

models:
  vision_name: "google/siglip-base-patch16-224"
  language_model_name: "Qwen/Qwen3-0.6B"
  resampler_type: "mlp"  # or "qformer", "perceiver"

image_processing:
  crop_strategy: "anyres_e2p"  # KEY: determines view generation
  fov_deg: 90.0
  overlap_ratio: 0.5
  use_vision_processor: true   # Use HF AutoProcessor if true

training:
  stages: ["vision", "resampler", "finetune"]  # Sequential execution
  stage_configs:
    vision:
      epochs: 5
      lr: 5e-4
      vicreg_loss_weight: 1.0
      vicreg_similarity_weight: 25.0
      vision_trainable_blocks: 2  # 0=freeze all, -1=unfreeze all
```

**Stage State Persistence**: Training saves `{prefix}_stage_state.json` to track completed stages. Resume with `--resume auto`.

## Development Workflows

### Training Workflow
```bash
# 1. Validate new CSV datasets before training
python scripts/validate_new_datasets.py --csv data/quic360/train.csv

# 2. Run multi-stage training
python scripts/train.py --config configs/default.yaml

# 3. Resume from last checkpoint
python scripts/train.py --config configs/default.yaml --resume auto

# 4. Run specific stage only
python scripts/train.py --config configs/default.yaml --stage resampler
```

### Dataset Format (CSV)
Columns: `url`, `query`, `annotation`
- `url`: Image URL or local path
- `query`: User question
- `annotation`: Ground truth answer

**Validation**: `scripts/validate_new_datasets.py` checks:
- CSV structure, image accessibility, dataset class compatibility
- Reports: sample count, validation pass rate, errors

### Testing & Quality
```bash
make test              # Run all tests
make test-cov          # With coverage report
make lint              # Ruff + flake8 + mypy
make format            # Black + isort
```

## Project-Specific Patterns

### 1. VICReg Overlap Loss (Vision Stage)
**Location**: `src/panovlm/losses/vicreg_overlap.py::compute_vicreg_overlap_loss()`

Operates on **adjacent view overlaps**:
```python
# Input: [B, V, H', W', D] (spatial grid of features)
# Extract overlapping regions between view[v] and view[v+1]
k = int(W' * overlap_ratio)
curr = features[:, v, :, -k:, :]  # Right k columns of view v
next = features[:, v+1, :, :k, :] # Left k columns of view v+1

# VICReg components
inv_loss = mse(curr, next)  # Invariance
var_loss = relu(γ - std(features))  # Variance (prevent collapse)
cov_loss = off_diagonal(cov)^2  # Covariance (decorrelation)
```

**Config tuning** (`docs/CONFIG_GUIDE.md`):
- Start with `vicreg_local_weight: 0.2` (low), increase gradually
- `vicreg_similarity_weight: 25.0` controls invariance strength
- Monitor metrics: `vicreg_local_inv`, `vicreg_local_var`, `vicreg_local_cov`

### 2. Text Formatting (Chat Templates)
**`UniversalTextFormatter`** (processors/universal_text_formatter.py):
- **Always** uses model's `chat_template` (via `tokenizer.apply_chat_template()`)
- For multi-turn: computes label masking by tokenizing prefix vs full text
- Fallback templates exist but are rarely used (HF models have templates)

```python
formatter = UniversalTextFormatter(tokenizer, system_message=...)
formatted = formatter.format_chat(question, answer)  # Returns formatted string
# Tokenization happens in dataset/processor
```

### 3. Progressive Stage Training
**State tracking** (`runs/{prefix}_stage_state.json`):
```json
{
  "completed_stages": ["vision"],
  "current_stage": "resampler",
  "vision": {"checkpoint": "runs/.../vision_final.ckpt", ...}
}
```

**Resume logic** (train.py):
- `--resume auto`: Loads last checkpoint for current stage
- `--resume runs/xxx.ckpt`: Explicit checkpoint path
- Stage execution: Sequential (vision → resampler → finetune)

### 4. LoRA Support (Optional)
**Enabled via PEFT**:
```yaml
training:
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_target_modules: ["q_proj", "v_proj"]
```

Applied to language model only during finetune stage.

## Common Pitfalls & Solutions

### Import Errors After Refactoring
**Problem**: `ModuleNotFoundError: No module named 'panovlm.data.image'`
**Solution**: Use `from panovlm.processors.image import PanoramaImageProcessor`

### VICReg Loss Not Decreasing
**Check**:
1. `vision_trainable_blocks` in config (should be > 0 for vision stage)
2. Learning rate (try `5e-4` instead of `1e-5`)
3. `vicreg_similarity_weight` (reduce if loss magnitude too high)

### Circular Import Issues
**Pattern**: Processors depend on nothing except external libs. Data imports from processors. Models import from both.
```
processors/ (leaf)
    ↑
data/ ← models/
```

### AnyRes ERP Dependency Missing
If using `crop_strategy: anyres_e2p`, ensure `py360convert` is installed:
```bash
pip install py360convert opencv-python
```

## Key Files to Reference

- **Training entry**: `scripts/train.py` (L1-1491, config-driven)
- **Model architecture**: `src/panovlm/models/model.py::PanoramaVLM`
- **Image processing**: `src/panovlm/processors/image.py::PanoramaImageProcessor`
- **VICReg loss**: `src/panovlm/losses/vicreg_overlap.py`
- **Config schema**: `src/panovlm/config/config_manager.py`
- **Architecture flow**: `docs/flow.md`
- **Config tuning guide**: `docs/CONFIG_GUIDE.md`

## External Dependencies

- **HuggingFace**: `transformers`, `peft` (for models/tokenizers/LoRA)
- **PyTorch Lightning**: Training loop, callbacks, logging
- **py360convert**: ERP↔Pinhole projection (for anyres_e2p strategy)
- **wandb**: Experiment tracking (optional, set `wandb_project` in config)

## Debugging Tips

1. **Enable debug logging**: `training.stage_configs.vision.debug_vicreg_loss: true`
2. **Check dataset loading**: `scripts/validate_new_datasets.py --csv your_file.csv`
3. **Visualize AnyRes tiles**: `scripts/anyres_e2p.py --input pano.jpg --save`
4. **Monitor stage state**: `cat runs/{prefix}_stage_state.json`
5. **Inspect checkpoint**: `torch.load(ckpt_path)['state_dict'].keys()`

## Questions for User Feedback

1. Are there specific vision encoder configs (like DINOv2 vs SigLIP) that need more documentation?
2. Should I add more details about the resampler types (MLP vs Qformer vs Perceiver)?
3. Do you want examples of custom loss integration or evaluation metrics?
4. Should I document the Docker workflow (Dockerfile exists but not in README)?
5. Are there any deprecated patterns I should explicitly warn about?
