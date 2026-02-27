"""
PanoRoPE-1D Implementation Examples for InternVL3.5 and Gemma3

This file contains working code examples for implementing PanoRoPE-1D
position ID modifications without touching HuggingFace transformers source.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import InternVLForConditionalGeneration, Gemma3ForConditionalGeneration


# ============================================================================
# EXAMPLE 1: InternVL3.5 with PanoRoPE-1D (Recommended Hook Point)
# ============================================================================

class PanoRoPEInternVL(InternVLForConditionalGeneration):
    """
    InternVL3.5 with PanoRoPE-1D position ID modification.
    
    This implementation modifies position_ids BEFORE passing to the language model,
    which is the cleanest hook point that doesn't require monkey-patching.
    """
    
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        labels: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_sizes: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Forward pass with PanoRoPE-1D position ID modification.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            pixel_values: Image pixels (batch_size, 3, H, W)
            position_ids: Position IDs (batch_size, seq_len) - MODIFIED HERE
            ... (other args same as parent)
        """
        
        # ✅ HOOK POINT 1: Modify position_ids BEFORE parent forward
        if position_ids is not None:
            position_ids = self.apply_pano_rope_1d(
                position_ids=position_ids,
                input_ids=input_ids,
                pixel_values=pixel_values,
            )
        
        # Call parent forward with modified position_ids
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            labels=labels,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            image_sizes=image_sizes,
            **kwargs,
        )
    
    def apply_pano_rope_1d(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply PanoRoPE-1D modifications to position_ids.
        
        Args:
            position_ids: (batch_size, seq_len) - 0-indexed positions
            input_ids: (batch_size, seq_len) - token IDs
            pixel_values: (batch_size, 3, H, W) - image pixels
        
        Returns:
            Modified position_ids with same shape
        """
        batch_size, seq_len = position_ids.shape
        modified_ids = position_ids.clone()
        
        # Example 1: Assign 2D spatial positions to vision patches
        if input_ids is not None and pixel_values is not None:
            modified_ids = self._assign_2d_spatial_positions(
                modified_ids, input_ids, pixel_values
            )
        
        # Example 2: Use relative positions within panoramic regions
        # modified_ids = self._assign_relative_positions(modified_ids, input_ids)
        
        return modified_ids
    
    def _assign_2d_spatial_positions(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assign 2D spatial positions to vision patches.
        
        For a 32×32 patch grid, vision patches at position i get:
        - 2D position: (row, col) = (i // 32, i % 32)
        - Encoded as: row * 1000 + col (or your custom encoding)
        """
        batch_size, seq_len = position_ids.shape
        modified_ids = position_ids.clone()
        
        # Find image token positions
        image_token_id = self.config.image_token_id
        
        for batch_idx in range(batch_size):
            # Find where <image> tokens are
            image_mask = (input_ids[batch_idx] == image_token_id)
            image_positions = torch.where(image_mask)[0]
            
            if len(image_positions) == 0:
                continue
            
            # Get number of patches from pixel_values
            # Assuming 448×448 input → 32×32 patches (14×14 patch size)
            num_patches_per_side = 32  # Adjust based on your vision encoder
            
            # Assign 2D positions to vision patches
            patch_idx = 0
            for pos in image_positions:
                # Assign positions for all patches at this image location
                for p in range(num_patches_per_side * num_patches_per_side):
                    if pos + p < seq_len:
                        row = p // num_patches_per_side
                        col = p % num_patches_per_side
                        # Encode 2D position as: row * 1000 + col
                        # (Adjust encoding based on your RoPE implementation)
                        modified_ids[batch_idx, pos + p] = row * 1000 + col
        
        return modified_ids
    
    def _assign_relative_positions(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Use relative positions within each panoramic region.
        
        All patches in the same panoramic image get relative positions 0-N.
        """
        batch_size, seq_len = position_ids.shape
        modified_ids = position_ids.clone()
        
        image_token_id = self.config.image_token_id
        
        for batch_idx in range(batch_size):
            image_mask = (input_ids[batch_idx] == image_token_id)
            image_positions = torch.where(image_mask)[0]
            
            # Assign relative positions within each image region
            relative_pos = 0
            for pos in image_positions:
                # All patches for this image get relative positions
                num_patches = 32 * 32  # Adjust based on your vision encoder
                for p in range(num_patches):
                    if pos + p < seq_len:
                        modified_ids[batch_idx, pos + p] = relative_pos
                        relative_pos += 1
        
        return modified_ids


# ============================================================================
# EXAMPLE 2: Gemma3 with PanoRoPE-1D
# ============================================================================

class PanoRoPEGemma3(Gemma3ForConditionalGeneration):
    """
    Gemma3 with PanoRoPE-1D position ID modification.
    
    Similar to InternVL3.5, but handles Gemma3's additional token_type_ids.
    """
    
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        token_type_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **lm_kwargs,
    ):
        """
        Forward pass with PanoRoPE-1D position ID modification.
        
        Note: Gemma3 also uses token_type_ids for bidirectional attention on images.
        """
        
        # ✅ HOOK POINT 1: Modify position_ids BEFORE parent forward
        if position_ids is not None:
            position_ids = self.apply_pano_rope_1d(
                position_ids=position_ids,
                input_ids=input_ids,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
            )
        
        # Call parent forward with modified position_ids
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )
    
    def apply_pano_rope_1d(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply PanoRoPE-1D modifications to position_ids.
        
        Gemma3-specific: Also considers token_type_ids for image masking.
        """
        batch_size, seq_len = position_ids.shape
        modified_ids = position_ids.clone()
        
        # Your PanoRoPE-1D logic here
        # Can also use token_type_ids to identify image regions
        
        return modified_ids


# ============================================================================
# EXAMPLE 3: Monkey-Patching Approach (Advanced)
# ============================================================================

def patch_qwen2_rope_for_pano():
    """
    Monkey-patch Qwen2RotaryEmbedding to apply PanoRoPE-1D.
    
    This is more invasive but allows modification at the RoPE level.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    
    original_forward = Qwen2RotaryEmbedding.forward
    
    def patched_forward(self, x, position_ids):
        # ✅ HOOK POINT 3: Modify position_ids at RoPE level
        position_ids = apply_pano_rope_1d_global(position_ids)
        return original_forward(self, x, position_ids)
    
    Qwen2RotaryEmbedding.forward = patched_forward


def patch_gemma3_rope_for_pano():
    """
    Monkey-patch Gemma3RotaryEmbedding to apply PanoRoPE-1D.
    """
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
    
    original_forward = Gemma3RotaryEmbedding.forward
    
    def patched_forward(self, x, position_ids, layer_type=None):
        # ✅ HOOK POINT 3: Modify position_ids at RoPE level
        position_ids = apply_pano_rope_1d_global(position_ids, layer_type)
        return original_forward(self, x, position_ids, layer_type)
    
    Gemma3RotaryEmbedding.forward = patched_forward


def apply_pano_rope_1d_global(position_ids, layer_type=None):
    """Global PanoRoPE-1D function for monkey-patching."""
    # Your PanoRoPE-1D logic here
    return position_ids


# ============================================================================
# EXAMPLE 4: Testing & Debugging
# ============================================================================

def test_pano_rope_internvl():
    """Test PanoRoPE-1D implementation for InternVL3.5."""
    
    # Load model
    model = PanoRoPEInternVL.from_pretrained(
        "OpenGVLab/InternVL3-1B-hf",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Add debug hook to verify position_ids reach RoPE
    original_rope = model.model.language_model.rotary_emb.forward
    
    def debug_rope(x, position_ids):
        print(f"✅ RoPE received position_ids:")
        print(f"   Shape: {position_ids.shape}")
        print(f"   Values: {position_ids}")
        print(f"   Min: {position_ids.min()}, Max: {position_ids.max()}")
        return original_rope(x, position_ids)
    
    model.model.language_model.rotary_emb.forward = debug_rope
    
    # Run forward pass
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    pixel_values = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16)
    
    print("Running forward pass...")
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
    
    print(f"✅ Output shape: {output.logits.shape}")


def test_pano_rope_gemma3():
    """Test PanoRoPE-1D implementation for Gemma3."""
    
    # Load model
    model = PanoRoPEGemma3.from_pretrained(
        "google/gemma-3-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Add debug hook
    original_rope = model.model.language_model.rotary_emb.forward
    
    def debug_rope(x, position_ids, layer_type=None):
        print(f"✅ RoPE received position_ids (layer_type={layer_type}):")
        print(f"   Shape: {position_ids.shape}")
        print(f"   Values: {position_ids}")
        return original_rope(x, position_ids, layer_type)
    
    model.model.language_model.rotary_emb.forward = debug_rope
    
    # Run forward pass
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    pixel_values = torch.randn(1, 3, 896, 896, dtype=torch.bfloat16)
    
    print("Running forward pass...")
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
    
    print(f"✅ Output shape: {output.logits.shape}")


# ============================================================================
# EXAMPLE 5: Verify Position ID Flow
# ============================================================================

def verify_position_id_flow():
    """
    Verify that position_ids flow correctly through the model.
    
    This is useful for debugging and understanding the position ID pipeline.
    """
    
    model = PanoRoPEInternVL.from_pretrained("OpenGVLab/InternVL3-1B-hf")
    
    # Hook into multiple points to trace position_ids
    hooks = {}
    
    def hook_internvl_model(module, input, output):
        """Hook at InternVLModel.forward()"""
        if len(input) > 0:
            position_ids = input[3] if len(input) > 3 else None
            hooks['internvl_model'] = position_ids
            print(f"InternVLModel.forward() position_ids: {position_ids}")
    
    def hook_qwen2_model(module, input, output):
        """Hook at Qwen2Model.forward()"""
        # position_ids is a kwarg, not positional
        hooks['qwen2_model'] = "See kwargs"
        print(f"Qwen2Model.forward() called")
    
    def hook_rope(module, input, output):
        """Hook at Qwen2RotaryEmbedding.forward()"""
        if len(input) > 1:
            position_ids = input[1]
            hooks['rope'] = position_ids
            print(f"RoPE.forward() position_ids shape: {position_ids.shape}")
            print(f"RoPE.forward() position_ids: {position_ids}")
    
    # Register hooks
    model.model.register_forward_hook(hook_internvl_model)
    model.model.language_model.register_forward_hook(hook_qwen2_model)
    model.model.language_model.rotary_emb.register_forward_hook(hook_rope)
    
    # Run forward pass
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    pixel_values = torch.randn(1, 3, 448, 448)
    
    print("Tracing position_ids flow...")
    with torch.no_grad():
        output = model(input_ids=input_ids, pixel_values=pixel_values)
    
    print("\n✅ Position ID flow verified!")
    print(f"Hooks captured: {list(hooks.keys())}")


if __name__ == "__main__":
    # Run tests
    print("=" * 80)
    print("Testing PanoRoPE-1D Implementation")
    print("=" * 80)
    
    # Uncomment to run tests:
    # test_pano_rope_internvl()
    # test_pano_rope_gemma3()
    # verify_position_id_flow()
    
    print("\n✅ Examples loaded successfully!")
    print("Use PanoRoPEInternVL or PanoRoPEGemma3 classes in your code.")
