# PanoRoPE-1D Quick Reference

## TL;DR

| Question | Answer |
|----------|--------|
| **Do InternVL3.5 & Gemma3 use 1D RoPE?** | ✅ YES, both use standard 1D RoPE |
| **Is it Qwen2.5-VL's 3D M-RoPE?** | ❌ NO, that's only for Qwen2.5-VL |
| **Where are position_ids created?** | In language model's forward (Qwen2/Gemma3TextModel) |
| **Where to inject modified position_ids?** | **Before language_model.forward()** (Hook Point 1) |
| **Position ID shape?** | `(batch_size, seq_len)` — 1D array |
| **Position ID indexing?** | 0-indexed (0, 1, 2, ...) for both models |
| **How do vision tokens get position_ids?** | Via `masked_scatter()` — inherit from `<image>` placeholder |
| **Does InternVL3.5 support dynamic resolution?** | ✅ YES, via `pixel_shuffle()` |
| **Does Gemma3 support dynamic resolution?** | ❌ NO, fixed resolution |
| **Can I modify position_ids without touching transformers?** | ✅ YES, subclass and override forward() |

---

## Position ID Flow Diagram

```
InternVLForConditionalGeneration.forward()
    ↓
InternVLModel.forward()
    ├─ Vision processing (pixel_values → image_features)
    ├─ Merge vision into embeddings (masked_scatter)
    └─ Call language_model.forward(position_ids=position_ids)
        ↓
    Qwen2Model.forward()
        ├─ If position_ids is None: position_ids = cache_position.unsqueeze(0)
        └─ Call rotary_emb.forward(hidden_states, position_ids)
            ↓
        Qwen2RotaryEmbedding.forward()
            ├─ Compute: freqs = inv_freq @ position_ids  (outer product)
            └─ Return: cos, sin (for RoPE application)
```

---

## Hook Point 1: Recommended Implementation

```python
from transformers import InternVLForConditionalGeneration
import torch

class PanoRoPEInternVL(InternVLForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ):
        # ✅ HOOK POINT: Modify position_ids here
        if position_ids is not None:
            position_ids = self.apply_pano_rope_1d(position_ids, pixel_values)
        
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
    
    def apply_pano_rope_1d(self, position_ids, pixel_values):
        """Your PanoRoPE-1D logic here."""
        # position_ids: (batch_size, seq_len)
        # pixel_values: (batch_size, 3, H, W) or None
        return modified_position_ids
```

---

## Key Class Locations

| Model | Class | File | Line |
|-------|-------|------|------|
| **InternVL3.5** | `InternVLModel.forward()` | `modeling_internvl.py` | 625 |
| **InternVL3.5** | `InternVLForConditionalGeneration.forward()` | `modeling_internvl.py` | 786 |
| **Gemma3** | `Gemma3Model.forward()` | `modeling_gemma3.py` | 882 |
| **Gemma3** | `Gemma3ForConditionalGeneration.forward()` | `modeling_gemma3.py` | 1023 |
| **Qwen2** (InternVL's LM) | `Qwen2Model.forward()` | `modeling_qwen2.py` | 363 |
| **Qwen2** (InternVL's LM) | `Qwen2RotaryEmbedding.forward()` | `modeling_qwen2.py` | 102 |
| **Gemma3** (LM) | `Gemma3TextModel.forward()` | `modeling_gemma3.py` | 527 |
| **Gemma3** (LM) | `Gemma3RotaryEmbedding.forward()` | `modeling_gemma3.py` | 216 |

---

## Position ID Modification Examples

### Example 1: Assign 2D spatial positions to vision patches

```python
def apply_pano_rope_1d(self, position_ids, pixel_values):
    """
    Assign 2D spatial positions to vision patches.
    For a 32×32 patch grid, assign positions based on (row, col).
    """
    batch_size, seq_len = position_ids.shape
    modified_ids = position_ids.clone()
    
    # Find vision token positions
    image_token_id = self.config.image_token_id
    image_mask = (position_ids == image_token_id)  # ❌ WRONG: position_ids are ints, not token IDs
    
    # Actually, you need to track image positions separately:
    # This is a simplified example
    
    return modified_ids
```

### Example 2: Use relative positions within panoramic regions

```python
def apply_pano_rope_1d(self, position_ids, pixel_values):
    """
    Use relative positions within each panoramic region.
    All patches in the same panoramic image get relative positions 0-N.
    """
    batch_size, seq_len = position_ids.shape
    modified_ids = position_ids.clone()
    
    # Your logic to identify panoramic regions and assign relative positions
    # ...
    
    return modified_ids
```

---

## Testing Your Implementation

```python
# 1. Load model
model = PanoRoPEInternVL.from_pretrained("OpenGVLab/InternVL3-1B-hf")

# 2. Add debug hook to verify position_ids reach RoPE
original_rope = model.model.language_model.rotary_emb.forward

def debug_rope(x, position_ids):
    print(f"✅ RoPE received position_ids: {position_ids}")
    print(f"   Shape: {position_ids.shape}")
    print(f"   Min: {position_ids.min()}, Max: {position_ids.max()}")
    return original_rope(x, position_ids)

model.model.language_model.rotary_emb.forward = debug_rope

# 3. Run forward pass
output = model(
    input_ids=torch.tensor([[1, 2, 3]]),
    pixel_values=torch.randn(1, 3, 448, 448),
)
```

---

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Position IDs have gaps (e.g., [0, 1, 5, 6]) | Ensure continuous sequence: [0, 1, 2, 3] |
| Wrong dtype (float32 instead of int64) | Use `position_ids.long()` or `dtype=torch.long` |
| Wrong device (CPU when model on GPU) | Use `position_ids.to(device)` |
| Modifying position_ids but not cache_position | Update both if using KV cache |
| Assuming all vision tokens have same position | They inherit from `<image>` placeholder positions |

---

## InternVL2.5 vs InternVL3.5

| Feature | 2.5 | 3.5 |
|---------|-----|-----|
| Dynamic Resolution | ❌ | ✅ |
| In HF transformers | ❌ | ✅ |
| Vision Encoder | InternViT-6B (custom) | InternViT (HF) |
| Language Model | Qwen2-7B | Qwen2 (configurable) |
| Position Encoding | 1D RoPE | 1D RoPE |

---

## Gemma3 Unique Features

- **token_type_ids**: Enables bidirectional attention within image blocks
- **layer_type variants**: Different RoPE parameters for "full_attention" vs "sliding_attention"
- **attention_scaling**: Per-layer-type scaling factor for RoPE

---

## References

- **Full Analysis**: `docs/PANOROPЕ_POSITION_IDS.md`
- **InternVL3.5 HF Hub**: https://huggingface.co/OpenGVLab/InternVL3-1B-hf
- **Gemma3 HF Hub**: https://huggingface.co/google/gemma-3-4b-it

