# src/cora/model/ — VLM Architecture

`PanoramaVLM` orchestrates all components. Forward pass varies by training stage.

## ARCHITECTURE

```
PanoramaVLM (vlm.py)
├── VisionBackbone (vision_encoder.py)    — Frozen/partially-trainable encoder
├── ResamplerModule (resampler/)          — Compresses vision tokens (pluggable)
├── VICRegProjector (projectors.py)       — Stage 1 only: self-supervised head
├── PanoramaProjector (projectors.py)     — Vision dim → LLM dim + positional encoding
├── LanguageFusion (language_fusion.py)    — Prepends vision tokens to text embeddings
└── LanguageModel (language_model.py)     — HF AutoModelForCausalLM + optional LoRA
```

## FORWARD PASS BY STAGE

| Stage | Path | Trainable | Loss |
|-------|------|-----------|------|
| `vision` | encoder→resampler→VICReg projector | resampler + VICReg proj | VICReg/Contrastive/DenseCL |
| `resampler` | encoder→resampler→projector→fusion→LLM + VICReg branch | resampler + projector | LM + weighted VICReg |
| `finetune` | encoder→resampler→projector→fusion→LLM | resampler + projector + LLM (LoRA) | LM only |

## TENSOR SHAPES

```
pixel_values:    [B, V, C, H, W]   # V=num_views (1 global + N tiles)
vision_features: [B*V, S, D_vis]   # S=spatial tokens from encoder
resampled:       [B*V, S', D_lat]  # S'=compressed sequence length
vision_tokens:   [B, T, D_lm]      # T=total vision tokens for LLM
```

## GLOBAL + TILE SEPARATION

When `num_views > 1`, view index 0 = global (full panorama downscaled), views 1..N = E2P tiles.
- **Tiles**: Full projector pipeline (positional encoding + linear + stitching)
- **Global**: Linear projection only (no PE, no stitching)
- Concatenated: `[global_tokens, tile_tokens]` along sequence dim

## RESAMPLERS (model/resampler/)

| Type | File | Key Params | Notes |
|------|------|------------|-------|
| `mlp` | `resamplers.py` | `hidden_dim`, `depth`, `pool_type` | Simplest, avg/max pool first |
| `bimamba` | `bimamba.py` | `d_state=64`, `d_conv=4`, `expand=2.0` | SSM-based. Requires `mamba_ssm` |
| `qformer` | `resamplers.py` | `num_query_tokens`, `num_hidden_layers` | Cross-attention queries |
| `perceiver` | `perceiver.py` | `num_latents`, `heads`, `depth` | Flamingo-style |
| `c_abstractor` | `c_abstractor.py` | `kernel_size=7`, `se_reduction=4` | Conv + squeeze-excite |
| `spatial_pool` | `resamplers.py` | `pool_size`, `depth` | Adaptive spatial pooling |
| `masked_drop` | `resamplers.py` | `mask_ratio`, `depth` | Random token masking |

Factory: `ResamplerModule(config.models, vision_hidden_size)` reads `resampler_type` from config.

## ANTI-PATTERNS (MODEL CODE)

- **Never hardcode model names** — always from `config.models.vision_name` / `config.models.language_model_name`
- **dtype management**: Use `self.dtype_cache` dict to avoid repeated dtype queries. Cast via `.to(target.dtype)`.
- **LoRA setup**: Called in `__init__` if `config.lora.use_lora`. Applied to `language_model_wrapper` only.
- **`attn_implementation`**: Default is `"sdpa"`, not `"flash_attention_2"`. FA2 is optional.
