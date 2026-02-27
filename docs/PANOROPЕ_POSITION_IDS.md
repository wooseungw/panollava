# PanoRoPE-1D Position ID Handling: InternVL3.5 vs Gemma3

## EXECUTIVE SUMMARY

Both InternVL3.5 and Gemma3 use **standard 1D RoPE** (not Qwen2.5-VL's 3D M-RoPE). Position IDs flow through the forward pass identically in both models, but they handle vision token integration differently. The key hook point for injecting modified position IDs is **before the language model's forward pass**, where you can intercept and modify `position_ids` without touching transformers source.

---

## 1. POSITION_IDS CREATION & FLOW

### InternVL3.5 Position ID Flow

**File**: `/tmp/transformers/src/transformers/models/internvl/modeling_internvl.py`

```python
# Line 625-672: InternVLModel.forward()
def forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,  # ← INPUT
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    vision_feature_layer: int | list[int] | None = None,
    vision_feature_select_strategy: str | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | InternVLModelOutputWithPast:
    # ... vision processing ...
    
    # Line 657-664: PASSES position_ids DIRECTLY to language_model
    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,  # ← PASSED THROUGH
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )
```

**Key Point**: InternVL3.5 uses `AutoModel.from_config(config.text_config)` to load the language model (line 539). This is typically **Qwen2** or **Qwen2.5** (default), but can be any HF model.

### Gemma3 Position ID Flow

**File**: `/tmp/transformers/src/transformers/models/gemma3/modeling_gemma3.py`

```python
# Line 882-985: Gemma3Model.forward()
def forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,  # ← INPUT
    past_key_values: Cache | None = None,
    token_type_ids: torch.LongTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    **lm_kwargs: Unpack[TransformersKwargs],
) -> tuple | Gemma3ModelOutputWithPast:
    # ... vision processing ...
    
    # Line 968-977: PASSES position_ids to language_model
    outputs = self.language_model(
        attention_mask=causal_mask_mapping,  # ← DICT (custom masking)
        position_ids=position_ids,  # ← PASSED THROUGH
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        return_dict=True,
        cache_position=cache_position,
        **lm_kwargs,
    )
```

---

## 2. WHERE POSITION_IDS ARE CREATED (IF NOT PROVIDED)

### Qwen2 (InternVL3.5's default language model)

**File**: `/tmp/transformers/src/transformers/models/qwen2/modeling_qwen2.py`

```python
# Line 363-415: Qwen2Model.forward()
def forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    
    # Line 385-386: CREATION POINT
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)  # Shape: (1, seq_len)
    
    # Line 408: PASSED TO RoPE
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

### Gemma3's Language Model (Gemma3TextModel)

**File**: `/tmp/transformers/src/transformers/models/gemma3/modeling_gemma3.py`

```python
# Line 527-601: Gemma3TextModel.forward()
def forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    
    # Line 553-554: CREATION POINT
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)  # Shape: (1, seq_len)
    
    # Line 583: PASSED TO RoPE (per layer_type)
    position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)
```

---

## 3. ROPE IMPLEMENTATION: 1D STANDARD

### Qwen2 RoPE (1D)

**File**: `/tmp/transformers/src/transformers/models/qwen2/modeling_qwen2.py`

```python
# Line 102-113: Qwen2RotaryEmbedding.forward()
def forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
        position_ids.shape[0], -1, 1
    ).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    
    # STANDARD 1D RoPE: outer product of inv_freq and position_ids
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    # Shape: (batch_size, seq_len, head_dim)
    
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin
```

**Input shapes**:
- `x`: `(batch_size, seq_len, hidden_size)`
- `position_ids`: `(batch_size, seq_len)` — **1D positions**
- `inv_freq`: `(head_dim // 2,)`

**Output shapes**:
- `cos`, `sin`: `(batch_size, seq_len, head_dim)`

### Gemma3 RoPE (1D with layer_type variants)

**File**: `/tmp/transformers/src/transformers/models/gemma3/modeling_gemma3.py`

```python
# Line 216-230: Gemma3RotaryEmbedding.forward()
def forward(self, x, position_ids, layer_type=None):
    inv_freq = getattr(self, f"{layer_type}_inv_freq")
    attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
    
    inv_freq_expanded = inv_freq[None, :, None].float().expand(
        position_ids.shape[0], -1, 1
    ).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    
    # STANDARD 1D RoPE: identical to Qwen2
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    # Shape: (batch_size, seq_len, head_dim)
    
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

**Key Difference**: Gemma3 supports **multiple layer types** (e.g., "full_attention" vs "sliding_attention") with different RoPE parameters per layer type.

---

## 4. HOOK POINTS FOR INJECTING MODIFIED POSITION_IDS

### Hook Point 1: Before Language Model Forward (RECOMMENDED)

**Location**: In your wrapper around `InternVLForConditionalGeneration` or `Gemma3ForConditionalGeneration`

```python
class PanoRoPEInternVL(InternVLForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ):
        # HOOK POINT 1: Modify position_ids BEFORE calling parent forward
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
        """
        Modify position_ids for panoramic vision tokens.
        position_ids shape: (batch_size, seq_len)
        """
        # Your PanoRoPE-1D logic here
        return modified_position_ids
```

**Advantages**:
- ✅ No transformers source modification
- ✅ Works for both InternVL3.5 and Gemma3
- ✅ Intercepts before RoPE computation
- ✅ Can access `pixel_values` for context

### Hook Point 2: In Language Model's Forward (ADVANCED)

**For InternVL3.5** (Qwen2-based):

```python
# Monkey-patch Qwen2Model.forward
original_qwen2_forward = Qwen2Model.forward

def patched_qwen2_forward(self, input_ids=None, position_ids=None, **kwargs):
    # HOOK POINT 2: Modify position_ids inside language model
    if position_ids is not None:
        position_ids = apply_pano_rope_1d(position_ids)
    
    return original_qwen2_forward(
        self, input_ids=input_ids, position_ids=position_ids, **kwargs
    )

Qwen2Model.forward = patched_qwen2_forward
```

**For Gemma3**:

```python
# Monkey-patch Gemma3TextModel.forward
original_gemma3_forward = Gemma3TextModel.forward

def patched_gemma3_forward(self, input_ids=None, position_ids=None, **kwargs):
    # HOOK POINT 2: Modify position_ids inside language model
    if position_ids is not None:
        position_ids = apply_pano_rope_1d(position_ids)
    
    return original_gemma3_forward(
        self, input_ids=input_ids, position_ids=position_ids, **kwargs
    )

Gemma3TextModel.forward = patched_gemma3_forward
```

### Hook Point 3: In RoPE Forward (MOST INVASIVE)

**For Qwen2** (InternVL3.5):

```python
# Monkey-patch Qwen2RotaryEmbedding.forward
original_rope_forward = Qwen2RotaryEmbedding.forward

def patched_rope_forward(self, x, position_ids):
    # HOOK POINT 3: Modify position_ids at RoPE level
    position_ids = apply_pano_rope_1d(position_ids)
    return original_rope_forward(self, x, position_ids)

Qwen2RotaryEmbedding.forward = patched_rope_forward
```

**For Gemma3**:

```python
# Monkey-patch Gemma3RotaryEmbedding.forward
original_rope_forward = Gemma3RotaryEmbedding.forward

def patched_rope_forward(self, x, position_ids, layer_type=None):
    # HOOK POINT 3: Modify position_ids at RoPE level
    position_ids = apply_pano_rope_1d(position_ids, layer_type)
    return original_rope_forward(self, x, position_ids, layer_type)

Gemma3RotaryEmbedding.forward = patched_rope_forward
```

---

## 5. INTERNVL2.5 vs INTERNVL3.5 ARCHITECTURE DIFFERENCES

### InternVL2.5 (Legacy)

- **Vision Encoder**: InternViT-6B (custom, not in HF transformers)
- **Language Model**: Qwen2-7B or Llama2-7B
- **Position Encoding**: Standard 1D RoPE (Qwen2/Llama)
- **Vision-Language Connector**: Simple linear projection
- **Dynamic Resolution**: NOT supported (fixed 448×448)

### InternVL3.5 (Current, in HF transformers)

- **Vision Encoder**: InternViT (HF-compatible, in `modeling_internvl.py`)
- **Language Model**: Qwen2 (default) or any HF model via `AutoModel.from_config()`
- **Position Encoding**: Standard 1D RoPE (inherited from language model)
- **Vision-Language Connector**: Masked scatter (line 655 in `modeling_internvl.py`)
- **Dynamic Resolution**: **SUPPORTED via pixel_shuffle** (line 674-707)

**Key Architectural Change**:

```python
# InternVL3.5: Pixel Shuffle for Dynamic Resolution
def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):
    """
    Downsamples vision features by rearranging spatial dimensions.
    Input:  (batch_size, width, height, channels)
    Output: (batch_size, height*scale_factor, width*scale_factor, channels/(scale_factor^2))
    """
    batch_size, width, height, channels = vision_features.size()
    
    # Reshape to allow downsampling
    vision_features = vision_features.view(
        batch_size, width, int(height * scale_factor), int(channels / scale_factor)
    )
    # Permute and reshape...
    return vision_features
```

**Impact on Position IDs**:
- InternVL2.5: Fixed number of vision tokens (e.g., 32×32 = 1024 patches)
- InternVL3.5: **Variable number of vision tokens** depending on `scale_factor`
  - This affects position ID assignment for vision tokens!

---

## 6. DYNAMIC RESOLUTION & POSITION ID ASSIGNMENT

### How InternVL3.5 Handles Variable Vision Tokens

**File**: `/tmp/transformers/src/transformers/models/internvl/modeling_internvl.py`

```python
# Line 644-655: Vision feature extraction and merging
if pixel_values is not None:
    image_features = self.get_image_features(
        pixel_values=pixel_values,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy=vision_feature_select_strategy,
        return_dict=True,
    ).pooler_output  # Shape: (batch_size, num_patches, hidden_size)
    
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    
    # MASKED SCATTER: Replaces <image> tokens with actual image features
    special_image_mask = self.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_features
    )
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
```

**Position ID Assignment**:
1. **Text tokens**: Get sequential position IDs (0, 1, 2, ...)
2. **Image tokens**: Replaced by `masked_scatter()` — **position IDs remain unchanged**
3. **Result**: Vision tokens inherit position IDs from the `<image>` placeholder tokens

**Problem for PanoRoPE-1D**:
- If you have multiple `<image>` tokens at different positions, they'll get different position IDs
- For panoramic vision, you may want **all panoramic patches to share spatial position information**

### How Gemma3 Handles Vision Tokens

**File**: `/tmp/transformers/src/transformers/models/gemma3/modeling_gemma3.py`

```python
# Line 946-952: Vision feature extraction and merging
if pixel_values is not None:
    image_features = self.get_image_features(pixel_values, return_dict=True).pooler_output
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    
    special_image_mask = self.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_features
    )
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
```

**Identical to InternVL3.5**: Same masked scatter approach.

**Additional Gemma3 Feature: token_type_ids**

```python
# Line 770: token_type_ids for bidirectional attention on images
token_type_ids: torch.LongTensor | None = None,

# Line 753-757: Bidirectional attention within image blocks
is_image_block = (token_type_ids_at_q_idx == 1) & (token_type_ids_at_kv_idx == 1)
same_image_block = image_group_ids_at_q_idx == image_group_ids_at_kv_idx
return is_image_block & same_image_block
```

**Key Difference**: Gemma3 uses `token_type_ids` to enable **bidirectional attention within image blocks**, while InternVL3.5 uses standard causal attention.

---

## 7. POSITION_IDS SHAPE & SEMANTICS

### Standard 1D RoPE Position IDs

**Shape**: `(batch_size, seq_len)`

**Example**:
```python
# Single sample, 10 tokens
position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

# Batch of 2, 10 tokens each
position_ids = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
])
```

### With KV Cache (Incremental Decoding)

```python
# During generation, only new token's position_id is needed
# cache_position tracks the absolute position in the sequence
cache_position = torch.tensor([10])  # Next token is at position 10
position_ids = cache_position.unsqueeze(0)  # Shape: (1, 1)
```

### For Vision Tokens (InternVL3.5 & Gemma3)

```python
# If you have: [text_0, text_1, <image>, text_2, text_3]
# And <image> is replaced by 256 vision patches:
# [text_0, text_1, patch_0, patch_1, ..., patch_255, text_2, text_3]

# Position IDs would be:
position_ids = torch.tensor([[0, 1, 2, 3, ..., 257, 258, 259]])
#                                    ↑ vision patches get positions 2-257
```

---

## 8. PALIGEMMA COMPARISON: 1-INDEXED POSITIONS

**File**: `/tmp/transformers/src/transformers/models/paligemma/modeling_paligemma.py`

```python
# Line 377-378: PaliGemma uses 1-indexed positions!
if position_ids is None:
    position_ids = cache_position.unsqueeze(0) + 1  # ← +1 for 1-indexing

# Line 579-580: Adjusted in prepare_inputs_for_generation
if model_inputs.get("position_ids") is not None:
    model_inputs["position_ids"] += 1
```

**Key Difference**:
- **Qwen2, Gemma3**: 0-indexed positions (0, 1, 2, ...)
- **PaliGemma**: 1-indexed positions (1, 2, 3, ...)

**For PanoRoPE-1D**: Ensure you match the indexing convention of your base model!

---

## 9. IMPLEMENTATION CHECKLIST FOR PANOROPЕ-1D

### Step 1: Determine Position ID Modification Strategy

```python
def compute_pano_position_ids(
    position_ids: torch.Tensor,  # (batch_size, seq_len)
    pixel_values: torch.Tensor | None,  # (batch_size, 3, H, W)
    image_token_positions: list[int],  # Indices where <image> tokens are
) -> torch.Tensor:
    """
    Modify position_ids for panoramic vision tokens.
    
    Options:
    1. Assign 2D spatial positions to vision patches
    2. Use relative positions within each panoramic region
    3. Interpolate positions for dynamic resolution
    """
    # Your PanoRoPE-1D logic
    return modified_position_ids
```

### Step 2: Identify Vision Token Positions

```python
def get_image_token_positions(input_ids, image_token_id):
    """Find positions of <image> tokens in input_ids."""
    return torch.where(input_ids == image_token_id)[1]  # Returns column indices
```

### Step 3: Hook into Forward Pass

**For InternVL3.5**:

```python
class PanoRoPEInternVL(InternVLForConditionalGeneration):
    def forward(self, input_ids=None, pixel_values=None, position_ids=None, **kwargs):
        # Compute image token positions
        image_token_positions = get_image_token_positions(
            input_ids, self.config.image_token_id
        )
        
        # Modify position_ids
        if position_ids is not None and image_token_positions is not None:
            position_ids = compute_pano_position_ids(
                position_ids, pixel_values, image_token_positions
            )
        
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            position_ids=position_ids,
            **kwargs,
        )
```

**For Gemma3**:

```python
class PanoRoPEGemma3(Gemma3ForConditionalGeneration):
    def forward(self, input_ids=None, pixel_values=None, position_ids=None, **kwargs):
        # Compute image token positions
        image_token_positions = get_image_token_positions(
            input_ids, self.config.image_token_id
        )
        
        # Modify position_ids
        if position_ids is not None and image_token_positions is not None:
            position_ids = compute_pano_position_ids(
                position_ids, pixel_values, image_token_positions
            )
        
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            position_ids=position_ids,
            **kwargs,
        )
```

### Step 4: Test Position ID Flow

```python
# Verify position_ids reach RoPE correctly
model = PanoRoPEInternVL.from_pretrained("OpenGVLab/InternVL3-1B-hf")

# Add debug hook to RoPE
original_rope = model.model.language_model.rotary_emb.forward

def debug_rope(x, position_ids):
    print(f"RoPE input position_ids shape: {position_ids.shape}")
    print(f"RoPE input position_ids: {position_ids}")
    return original_rope(x, position_ids)

model.model.language_model.rotary_emb.forward = debug_rope

# Run forward pass
output = model(input_ids=..., pixel_values=..., position_ids=...)
```

---

## 10. KEY DIFFERENCES SUMMARY TABLE

| Aspect | InternVL3.5 | Gemma3 |
|--------|-------------|--------|
| **RoPE Type** | 1D (Qwen2) | 1D (with layer_type variants) |
| **Position ID Shape** | `(batch_size, seq_len)` | `(batch_size, seq_len)` |
| **Position ID Indexing** | 0-indexed | 0-indexed |
| **Vision Token Handling** | `masked_scatter()` | `masked_scatter()` |
| **Dynamic Resolution** | ✅ `pixel_shuffle()` | ❌ Fixed resolution |
| **Bidirectional Attention** | ❌ Causal only | ✅ Bidirectional for images |
| **token_type_ids** | ❌ Not used | ✅ Used for image masking |
| **Language Model** | Qwen2 (default) | Gemma3TextModel (custom) |
| **Hook Point** | Before `language_model.forward()` | Before `language_model.forward()` |

---

## 11. CRITICAL IMPLEMENTATION NOTES

### ⚠️ Position ID Continuity

When modifying position_ids for vision tokens, ensure:
1. **No gaps**: Position IDs should be continuous (0, 1, 2, ..., seq_len-1)
2. **Correct dtype**: Must be `torch.LongTensor`
3. **Correct device**: Must match input device (GPU/CPU)

```python
# ✅ CORRECT
position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

# ❌ WRONG
position_ids = torch.arange(seq_len, dtype=torch.float32)  # Wrong dtype
position_ids = torch.arange(seq_len)  # Wrong device (CPU if model on GPU)
```

### ⚠️ Cache Position Handling

During generation with KV cache:

```python
# cache_position is used to compute position_ids if not provided
cache_position = torch.arange(
    past_seen_tokens, past_seen_tokens + seq_len, device=device
)

# If you modify position_ids, ensure cache_position is also updated
if position_ids is not None:
    # Your modification
    position_ids = apply_pano_rope(position_ids)
    # Update cache_position to match
    cache_position = position_ids[0]  # Extract from modified position_ids
```

### ⚠️ Batch Processing

Position IDs must be consistent across batch:

```python
# ✅ CORRECT: Same position sequence for all batch items
position_ids = torch.tensor([
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
])

# ❌ WRONG: Different positions per batch item (unless intentional)
position_ids = torch.tensor([
    [0, 1, 2, 3, 4],
    [10, 11, 12, 13, 14],
])
```

---

## 12. REFERENCES

### InternVL3.5 Source
- **Modeling**: `/tmp/transformers/src/transformers/models/internvl/modeling_internvl.py`
- **Config**: `/tmp/transformers/src/transformers/models/internvl/configuration_internvl.py`
- **HF Hub**: https://huggingface.co/OpenGVLab/InternVL3-1B-hf

### Gemma3 Source
- **Modeling**: `/tmp/transformers/src/transformers/models/gemma3/modeling_gemma3.py`
- **Config**: `/tmp/transformers/src/transformers/models/gemma3/configuration_gemma3.py`
- **HF Hub**: https://huggingface.co/google/gemma-3-4b-it

### Qwen2 RoPE (InternVL3.5 default)
- **Modeling**: `/tmp/transformers/src/transformers/models/qwen2/modeling_qwen2.py`
- **RoPE Forward**: Line 102-113

### PaliGemma (Reference for 1-indexed positions)
- **Modeling**: `/tmp/transformers/src/transformers/models/paligemma/modeling_paligemma.py`
- **Position ID Adjustment**: Line 377-378, 579-580

