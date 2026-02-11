"""Text-language fusion utilities shared across the project."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..processors.text import UniversalTextFormatter


class LanguageFusion:
    """Utility helper for stitching vision embeddings with language tokens."""

    def __init__(
        self,
        language_model: nn.Module,
        tokenizer,
        vision_token_id: int,
        ignore_index: int,
        max_text_length: int,
        system_prompt: Optional[str] = None,
        default_generation_prompt: Optional[str] = None,
        formatter: Optional[UniversalTextFormatter] = None,
        fallback_insert_position: str = "prefix",
    ) -> None:
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.vision_token_id = vision_token_id
        self.ignore_index = ignore_index
        self.max_text_length = max_text_length
        self.default_generation_prompt = (
            default_generation_prompt or "Describe the panoramic scene."
        )

        if fallback_insert_position not in {"prefix", "suffix"}:
            fallback_insert_position = "prefix"
        self.fallback_insert_position = fallback_insert_position

        self.text_formatter = formatter or UniversalTextFormatter(
            tokenizer,
            system_msg=system_prompt,
        )

    # ------------------------------------------------------------------
    # Configuration updates
    # ------------------------------------------------------------------
    def update_vision_token_id(self, vision_token_id: int) -> None:
        self.vision_token_id = vision_token_id

    def update_max_text_length(self, max_text_length: int) -> None:
        self.max_text_length = max_text_length

    # ------------------------------------------------------------------
    # Public helpers used by PanoramaVLM
    # ------------------------------------------------------------------
    def prepare_generation_inputs(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids is None:
            input_ids, attention_mask = self.create_default_prompt(batch_size, device)
        else:
            input_ids = self._ensure_2d(input_ids).to(device)
            if attention_mask is None:
                attention_mask = (input_ids != self._pad_token_id()).long()
            attention_mask = self._ensure_2d(attention_mask).to(device)
            input_ids, attention_mask = self.adjust_batch_size(input_ids, attention_mask, batch_size)

        input_ids = self._trim_tensor(input_ids)
        attention_mask = self._trim_tensor(attention_mask)
        return input_ids, attention_mask

    def create_default_prompt(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized = self.text_formatter.tokenize_for_generation(
            user_msg=self.default_generation_prompt,
            max_length=self.max_text_length,
        )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        input_ids = self._ensure_2d(input_ids)
        attention_mask = self._ensure_2d(attention_mask)
        input_ids, attention_mask = self.adjust_batch_size(input_ids, attention_mask, batch_size)
        return input_ids, attention_mask

    def adjust_batch_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        current = input_ids.size(0)
        if current == target_batch_size:
            return input_ids, attention_mask
        if current == 1 and target_batch_size > 1:
            input_ids = input_ids.repeat(target_batch_size, 1)
            attention_mask = attention_mask.repeat(target_batch_size, 1)
            return input_ids, attention_mask
        if current > target_batch_size:
            return input_ids[:target_batch_size], attention_mask[:target_batch_size]

        repeat_factor = math.ceil(target_batch_size / current)
        input_ids = input_ids.repeat(repeat_factor, 1)[:target_batch_size]
        attention_mask = attention_mask.repeat(repeat_factor, 1)[:target_batch_size]
        return input_ids, attention_mask

    def prepare_text_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if input_ids is None:
            raise ValueError("input_ids must be provided for text preparation")

        input_ids = self._ensure_2d(input_ids)
        if attention_mask is None:
            attention_mask = (input_ids != self._pad_token_id()).long()
        attention_mask = self._ensure_2d(attention_mask)
        if labels is not None:
            labels = self._ensure_2d(labels)

        input_ids = self._trim_tensor(input_ids)
        attention_mask = self._trim_tensor(attention_mask)
        if labels is not None:
            labels = self._trim_tensor(labels)

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        result: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
        }
        if labels is not None:
            result["labels"] = labels
        return result

    def fuse(
        self,
        vision_tokens: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if text_inputs is None:
            text_inputs = self.prepare_text_inputs(input_ids, attention_mask, labels)

        embeds = text_inputs["inputs_embeds"]
        attn = text_inputs["attention_mask"]
        ids = text_inputs["input_ids"]
        lbls = text_inputs.get("labels") if text_inputs else labels

        batch_size = embeds.size(0)
        fused_embeds = []
        fused_attn = []
        fused_labels = [] if lbls is not None else None
        num_vision_tokens = vision_tokens.size(1)

        for idx in range(batch_size):
            text_embed = embeds[idx]
            text_attn = attn[idx]
            text_ids = ids[idx]
            text_labels = lbls[idx] if lbls is not None else None

            vision_mask = text_ids == self.vision_token_id
            has_vision_token = torch.any(vision_mask)

            if has_vision_token:
                first_position = int(torch.nonzero(vision_mask, as_tuple=False)[0].item())
                filtered_embed = text_embed[~vision_mask]
                filtered_attn = text_attn[~vision_mask]
                filtered_labels = text_labels[~vision_mask] if text_labels is not None else None
                tokens_before = int((~vision_mask[:first_position]).sum().item())
                insert_pos = tokens_before
            else:
                filtered_embed = text_embed
                filtered_attn = text_attn
                filtered_labels = text_labels
                insert_pos = 0 if self.fallback_insert_position == "prefix" else filtered_embed.size(0)

            prefix_embed = filtered_embed[:insert_pos]
            suffix_embed = filtered_embed[insert_pos:]
            combined_embed = torch.cat([prefix_embed, vision_tokens[idx], suffix_embed], dim=0)

            prefix_attn = filtered_attn[:insert_pos]
            suffix_attn = filtered_attn[insert_pos:]
            vision_attn = torch.ones(num_vision_tokens, dtype=filtered_attn.dtype, device=vision_tokens.device)
            combined_attn = torch.cat([prefix_attn, vision_attn, suffix_attn], dim=0)

            fused_embeds.append(combined_embed)
            fused_attn.append(combined_attn)

            if fused_labels is not None and filtered_labels is not None:
                prefix_labels = filtered_labels[:insert_pos]
                suffix_labels = filtered_labels[insert_pos:]
                vision_labels = torch.full(
                    (num_vision_tokens,),
                    self.ignore_index,
                    dtype=filtered_labels.dtype,
                    device=vision_tokens.device,
                )
                combined_labels = torch.cat([prefix_labels, vision_labels, suffix_labels], dim=0)
                fused_labels.append(combined_labels)

        padded_embeds, padded_attn, padded_labels = self._pad_fused_sequences(
            fused_embeds, fused_attn, fused_labels, vision_tokens.size(-1)
        )

        output: Dict[str, torch.Tensor] = {
            "inputs_embeds": padded_embeds,
            "attention_mask": padded_attn,
        }
        if padded_labels is not None:
            output["labels"] = padded_labels
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pad_fused_sequences(
        self,
        fused_embeds,
        fused_attn,
        fused_labels,
        embed_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        target_len = max(seq.size(0) for seq in fused_embeds)

        padded_embeds = []
        padded_attn = []
        padded_labels = [] if fused_labels is not None else None

        for idx, embed_seq in enumerate(fused_embeds):
            pad_amount = target_len - embed_seq.size(0)
            if pad_amount > 0:
                pad_embed = torch.zeros((pad_amount, embed_dim), dtype=embed_seq.dtype, device=embed_seq.device)
                pad_attn = torch.zeros(
                    pad_amount,
                    dtype=fused_attn[idx].dtype,
                    device=fused_attn[idx].device,
                )
                embed_seq = torch.cat([embed_seq, pad_embed], dim=0)
                fused_attn[idx] = torch.cat([fused_attn[idx], pad_attn], dim=0)
                if padded_labels is not None and fused_labels is not None:
                    pad_labels = torch.full(
                        (pad_amount,),
                        self.ignore_index,
                        dtype=fused_labels[idx].dtype,
                        device=fused_labels[idx].device,
                    )
                    fused_labels[idx] = torch.cat([fused_labels[idx], pad_labels], dim=0)
            padded_embeds.append(embed_seq)
            padded_attn.append(fused_attn[idx])
            if padded_labels is not None and fused_labels is not None:
                padded_labels.append(fused_labels[idx])

        stacked_embeds = torch.stack(padded_embeds, dim=0)
        stacked_attn = torch.stack(padded_attn, dim=0)
        stacked_labels = torch.stack(padded_labels, dim=0) if padded_labels is not None else None
        return stacked_embeds, stacked_attn, stacked_labels

    def _ensure_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    def _trim_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.max_text_length is None:
            return tensor
        if tensor.size(1) <= self.max_text_length:
            return tensor
        return tensor[:, : self.max_text_length]

    def _pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
