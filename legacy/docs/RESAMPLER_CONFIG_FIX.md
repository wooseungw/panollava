# Resampler Configuration Dimension Mismatch Fix

## Problem Summary

**Issue**: When loading a trained checkpoint for evaluation, the model fails with dimension mismatch errors like:
```
size mismatch for resampler_module.resampler.input_proj.weight: 
copying a param with shape torch.Size([1024, 1152]) from checkpoint, 
the shape in current model is torch.Size([1536, 1152])
```

**Root Cause**: 
1. `resampler_hidden_dim` and BiMamba `expand` parameters were not saved in checkpoint metadata
2. During evaluation, the model was created with default values (e.g., `hidden_dim=1536`, `expand=2.0`) instead of the trained values (e.g., `hidden_dim=1024`, `expand=1.75`)

## Solution (3-Layer Defense)

### Layer 1: Save Resampler Config in Metadata ✅

**File**: `scripts/train.py`

**Changes**:
```python
checkpoint_metadata = {
    "model_config": {
        # ... existing fields ...
        # ✨ NEW: Save resampler configuration
        "resampler_config": getattr(lit_model.model_config, 'resampler_config', None),
        "resampler_hidden_dim": getattr(lit_model.model_config, 'resampler_hidden_dim', None),
    },
}
```

**Purpose**: Store the exact resampler configuration used during training in `checkpoint_metadata.json`.

---

### Layer 2: Auto-Infer from Checkpoint Weights ✅

**File**: `src/panovlm/models/model.py::from_checkpoint()`

**Changes**:

#### 2.1 Infer `resampler_hidden_dim` from weight shapes
```python
# Auto-infer resampler_hidden_dim from checkpoint weights
for k in ['resampler_module.resampler.input_proj.weight', ...]:
    if k in model_state_dict:
        w = model_state_dict[k]
        # torch.Size([hidden_dim, in_features])
        resampler_hidden_dim = int(w.shape[0])
        print(f"🔍 체크포인트에서 resampler_hidden_dim 자동 추론: {resampler_hidden_dim}")
```

#### 2.2 Infer BiMamba `expand` parameter
```python
# BiMamba의 in_proj.weight는 torch.Size([expand * hidden_dim * 2, hidden_dim])
for k in ['resampler_module.resampler.blocks.0.forward_block.in_proj.weight', ...]:
    if k in model_state_dict:
        w = model_state_dict[k]
        expanded_dim = int(w.shape[0] // 2)
        bimamba_expand = expanded_dim / resampler_hidden_dim
        print(f"🔍 체크포인트에서 BiMamba expand 자동 추론: {bimamba_expand}")
```

#### 2.3 Load from metadata (highest priority)
```python
# Try loading from checkpoint_metadata.json first
metadata_path = checkpoint_path.parent / "checkpoint_metadata.json"
if metadata_path.exists():
    metadata = json.load(open(metadata_path))
    model_cfg = metadata.get('model_config', {})
    
    # Metadata overrides weight inference
    if model_cfg.get('resampler_hidden_dim'):
        resampler_hidden_dim = model_cfg['resampler_hidden_dim']
    if model_cfg.get('resampler_config', {}).get('expand'):
        bimamba_expand = model_cfg['resampler_config']['expand']
```

**Purpose**: Even without metadata, infer correct dimensions from checkpoint weights as fallback.

---

### Layer 3: Proper Config Propagation ✅

**File**: `src/panovlm/models/vision/resampler.py::ResamplerModule.__init__()`

**Changes**:
```python
# Check top-level config attributes FIRST (before cfg_dict)
resampler_hidden_dim = getattr(config, 'resampler_hidden_dim', None)
if resampler_hidden_dim is not None:
    print(f"🔧 [ResamplerModule] config.resampler_hidden_dim={resampler_hidden_dim} 발견, 적용합니다")
    preset_kwargs['hidden_dim'] = resampler_hidden_dim
```

**Purpose**: Ensure `config.resampler_hidden_dim` takes precedence over default presets.

---

## Priority Order

When loading a checkpoint, the system now uses this priority order:

1. **Metadata file** (`checkpoint_metadata.json`) - highest priority
2. **Checkpoint weight inference** - fallback if metadata missing
3. **Config defaults** - only if both above fail

## Testing

### Before Fix
```bash
python scripts/eval.py --checkpoint-dir runs/model/finetune/anyres-e2p_bimamba
# ❌ RuntimeError: size mismatch for resampler weights
```

### After Fix
```bash
python scripts/eval.py --checkpoint-dir runs/model/finetune/anyres-e2p_bimamba
# ✅ 모델 로드 성공!
# 🔍 체크포인트에서 resampler_hidden_dim 자동 추론: 1024
# 🔍 체크포인트에서 BiMamba expand 자동 추론: 1.75
# 🔧 [ResamplerModule] config.resampler_hidden_dim=1024 발견, 적용합니다
```

## Future-Proofing

### For New Training Runs
- **No action needed** - metadata is automatically saved with correct resampler config

### For Old Checkpoints (without metadata)
- **Auto-inference works** - dimensions are correctly inferred from weight shapes
- **Recommended**: Re-save with metadata by running:
  ```bash
  python scripts/add_metadata_to_checkpoint.py --checkpoint runs/old_model/final.ckpt
  ```

## Related Files

- `scripts/train.py` - Metadata creation (Line 1407-1417)
- `src/panovlm/models/model.py` - Auto-inference logic (Line 1461-1540)
- `src/panovlm/models/vision/resampler.py` - Config application (Line 92-99)
- `src/panovlm/config/schema.py` - ModelConfig schema (Line 197: `resampler_hidden_dim`)

## Technical Details

### BiMamba Dimension Calculations

```python
# Given checkpoint weights:
input_proj.weight: [hidden_dim, vision_features]      # e.g., [1024, 1152]
in_proj.weight: [expand * hidden_dim * 2, hidden_dim] # e.g., [3584, 1024]

# Inference:
hidden_dim = input_proj.weight.shape[0]  # 1024
expanded_dim = in_proj.weight.shape[0] // 2  # 3584 // 2 = 1792
expand = expanded_dim / hidden_dim  # 1792 / 1024 = 1.75
```

### Default Values (`src/panovlm/config/schema.py`)

```python
RESAMPLER_CONFIGS = {
    "bimamba": {
        "hidden_dim": 1024,  # ✅ Correct default
        "expand": 1.75,      # ✅ Correct default
        "d_state": 64,
        "d_conv": 4,
    }
}
```

## Summary

✅ **Training**: Automatically saves resampler config in metadata  
✅ **Evaluation**: Loads from metadata → weight inference → defaults (in that order)  
✅ **Legacy Support**: Old checkpoints work via auto-inference  
✅ **No User Action Required**: All automatic  

**Result**: No more dimension mismatch errors! 🎉
