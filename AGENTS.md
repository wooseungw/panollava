Always use the OpenAI developer documentation MCP server if you need to work with the OpenAI API, ChatGPT Apps SDK, Codex, or related docs without me having to explicitly ask.

# CORA / PanoLLaVA — Project Knowledge Base

**Generated:** 2026-02-19 **Commit:** ab73e9e **Branch:** main

## OVERVIEW

Panoramic 360° VLM research project. Combines vision encoders (SigLIP/CLIP/DINOv2) + LLMs (Qwen/Llama/Gemma) via learnable resamplers. 3-stage progressive training with VICReg self-supervised loss. Package name: `cora` v2.0.0.

## STRUCTURE

```
panollava/
├── src/cora/              # Active package (pip install -e .)
│   ├── model/             # VLM architecture (vlm.py orchestrator)
│   │   └── resampler/     # Pluggable: MLP, BiMamba, Perceiver, C-Abstractor, QFormer, SpatialPool, MaskedDrop
│   ├── training/          # 3-stage trainer, Lightning module, losses, callbacks
│   ├── processors/        # Panorama image processing (7 crop strategies) + text formatting
│   ├── data/              # CSV dataset + Lightning DataModule
│   ├── config/            # Pydantic schemas (CORAConfig) + YAML manager
│   ├── evaluation/        # 5 metrics: BLEU, CIDEr, METEOR, ROUGE-L, SPICE
│   ├── inference/         # PanoramaGenerator
│   └── baseline/          # Separate LoRA finetuning path for off-the-shelf VLMs
├── scripts/               # Thin CLI entry points (train.py, eval.py, etc.)
├── configs/               # YAML experiment configs
├── data/                  # CSV datasets (quic360)
├── runs/                  # Training outputs, checkpoints
├── legacy/                # Archived prior codebase — DO NOT import from
└── .sisyphus/             # AI agent plans/notepads
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Train CORA model | `scripts/train.py` → `CORATrainer` | `--config configs/default.yaml` |
| Train baseline VLM | `scripts/baseline_finetune.py` | LoRA on Qwen2-VL, InternVL, Gemma3, BLIP2 |
| Evaluate model | `scripts/eval.py` | Inference + 5 metrics, supports `--csv-input` |
| LLM-as-Judge eval | `scripts/llm_judge_eval.py` | Needs `OPENAI_API_KEY` in `.env` |
| Add new resampler | `src/cora/model/resampler/` | Add file + register in `resamplers.py` |
| Add crop strategy | `src/cora/processors/images.py` | Extend `VALID_STRATEGIES` |
| Modify config schema | `src/cora/config/schema.py` | Pydantic BaseModel subclasses |
| Smoke-test pipeline | `scripts/dry_run.py` | Config→model→forward pass (1 batch) |
| Aggregate results | `scripts/collect_results.py` | Markdown/LaTeX/CSV comparison tables |

## CONVENTIONS

- **Line length**: 120 (Black + isort + ruff all set to 120)
- **Type hints**: MANDATORY — mypy strict mode (`disallow_untyped_defs=true`)
- **Imports**: `isort` with `profile="black"`, `known_first_party=["cora"]`
- **Config**: YAML-first. All hyperparams in `configs/`. Scripts are thin wrappers.
- **Pydantic**: All config schemas use `BaseModel` with `model_config = {"extra": "allow"}`
- **Logging**: `logger = logging.getLogger(__name__)` per module
- **Optional deps**: Use `try/except ImportError` → set to `None`. Mamba, py360convert, LlavaOnevision are optional.
- **Conda env**: `conda activate pano`
- **Install**: `pip install -e .` (editable mode from project root)

## ANTI-PATTERNS (THIS PROJECT)

### CRITICAL — Will break training

- **`torch.zeros([])`** in Lightning step returns → Use `torch.tensor(0.0, device=self.device, requires_grad=True)`. Empty tensor (numel=0) crashes Lightning aggregation/DDP reduce.
- **`weights_only=True`** in `torch.load()` → Fails with PyTorch 2.6+ on our checkpoints. Always use `weights_only=False`.
- **`max_length`/`truncation` in HF processor calls** → Breaks multimodal models (InternVL). Truncate `input_ids` manually after processing.

### CRITICAL — Will produce wrong output

- **Qwen3 chat_template** includes `<think>` blocks even with `enable_thinking=False`. `UniversalTextFormatter` auto-overrides this for SFT. Don't bypass.
- **Qwen3 inference** requires `<think>\n\n</think>` prefix injection. `PanoramaGenerator` handles this. Don't call `model.generate()` directly for Qwen3.
- **Qwen3 default sampling**: `do_sample=True, temperature=0.6`. Override for deterministic eval.

### GOTCHAS — Silent failures

- **`resampler_type: "bimamba"` without `mamba_ssm`** → `Mamba = None` silently. Runtime crash at model init, not import time. Fix: `pip install mamba_ssm` + `source fix_mamba_cuda.sh`.
- **`crop_strategy: "anyres_e2p"` without `py360convert`** → Same pattern. Fix: `pip install py360convert opencv-python`.
- **CUDA memory fragmentation** in LoRA runs >7000 steps. Ensure `CUDACacheCleanupCallback` is registered (`training.cache_cleanup_interval` in YAML).
- **OOM at epoch boundaries** — validation metrics + checkpoint save spike memory. Use `torch.cuda.empty_cache()` in `on_validation_epoch_start`. Reduce `num_workers` if system RAM-bound.

### NAMING — Post-refactoring confusion

- Package is `cora`, repo is `panollava`, README says `panovlm`. **Import as `cora`**.
- `src/cora/` is active. `legacy/` is archived. **Never import from `legacy/`**.
- `.github/copilot-instructions.md` references stale `panovlm` paths — ignore that file, use this one.

## IMPORT PATTERNS

```python
# Model
from cora.model.vlm import PanoramaVLM
from cora.model.vision_encoder import VisionBackbone
from cora.model.language_model import LanguageModel
from cora.model.resampler import ResamplerModule

# Training
from cora.training.trainer import CORATrainer
from cora.training.module import PanoramaTrainingModule
from cora.training.losses import VICRegLoss, PanoContrastiveLoss, DenseCLLoss

# Data
from cora.data.dataset import PanoramaDataset
from cora.data.datamodule import PanoramaDataModule

# Processors
from cora.processors.images import PanoramaImageProcessor
from cora.processors.text import UniversalTextFormatter
from cora.processors.processor import PanoramaProcessor

# Config
from cora.config.schema import CORAConfig, ModelConfig, StageConfig, BaselineConfig
from cora.config.manager import ConfigManager

# Evaluation
from cora.evaluation.metrics import CORAEvaluator

# WRONG — these are dead paths:
# from panovlm.* import ...      ← ModuleNotFoundError
# from cora.data.image import ... ← removed in refactoring
```

## DEPENDENCY GRAPH (no circular imports)

```
processors/ (leaf — no internal cora imports)
    ↑
data/ ← model/
    ↑       ↑
training/ ──┘
    ↑
inference/
```

## COMMANDS

```bash
# Setup
conda activate pano
pip install -e ".[dev]"
source fix_mamba_cuda.sh          # if using BiMamba resampler

# Train
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --stage resampler --resume auto
python scripts/baseline_finetune.py --config <yaml>

# Evaluate
python scripts/eval.py --checkpoint <ckpt> --test-csv <csv>
python scripts/eval.py --csv-input predictions.csv
bash install_eval_metrics.sh      # first-time: installs pycocoevalcap, NLTK, etc.

# Validate pipeline (no GPU training)
python scripts/dry_run.py

# Quality (no Makefile exists — run tools directly)
ruff check src/
black --check --line-length 120 src/
isort --check-only src/
mypy src/cora/
pytest tests/
```

## PREVIOUS SESSION LEARNINGS

- **OOM at epoch end**: Signal 9 kills at 99% of epoch = validation + checkpoint memory spike. Fix: reduce batch_size, add gradient checkpointing, `torch.cuda.empty_cache()` before validation.
- **Autobatch `fraction=0.85`** leaves only 3.6GB headroom on 24GB 3090. Combined with LoRA activation storage, this is fragile.
- **Repo restructure completed**: `src/panovlm/` + `CORA/cora/` merged into `src/cora/`. Old code in `legacy/`.
- **Shell scripts hardcode paths**: `/home/wsw/miniconda3/envs/pano/bin/python` — not portable.
- **No CI/CD**: No `.github/workflows/`, no Makefile, no pre-commit config.
- **Stage state**: `runs/{experiment}/stage_state.json` tracks completed stages. `--resume auto` reads this.

## NOTES

- Dataset CSV format: columns `url`, `query`, `annotation` (fallback: `image`, `instruction`, `response`)
- Dataset retry: `_MAX_RETRIES=10` on corrupted images — silently skips to next sample
- Eval resume: `predictions_partial.csv` saved every 500 samples — can be interrupted/resumed
- `precision: "16-mixed"` default (bf16 auto-detected; fallback to fp16)
- `scripts/inference.py` is a **stub** — actual inference via `scripts/eval.py` or `PanoramaVLM.generate()`
- `pyproject.toml` CLI entries `cora-train`/`cora-eval` may be broken — use `scripts/` directly
- `attn_implementation: "sdpa"` is default, not `"flash_attention_2"` — Flash Attn 2 is optional