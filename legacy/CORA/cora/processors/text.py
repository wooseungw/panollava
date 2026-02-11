"""
Universal Text Formatter (Chat-template-first)
──────────────────────────────────────────────
- Goal: Use 'chat_template' provided by the model first for various LLMs.
- Assumption: Text-only (vision tokens handling is delegated to fusion module or pre-inserted).
- Features:
  1) Apply apply_chat_template if available (HF recommended).
  2) Multi-turn training label masking: Uses prefix length difference for accurate masking.
  3) Minimal fallback templates if no template is found.
"""

from typing import Optional, Dict, Any, List
import re
import torch
import warnings

_DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant specialized in analyzing panoramic images."
)

class UniversalTextFormatter:
    """Chat-template-first LLM text formatter."""

    FALLBACK_FORMATS: Dict[str, Dict[str, str]] = {
        "llama": {
            "system_template": "[SYSTEM]\n{system}\n",
            "user_template": "[USER]\n{user}\n",
            "assistant_template": "[ASSISTANT]\n{assistant}\n",
            "assistant_start": "[ASSISTANT]\n",
        },
        "qwen": {
            "system_template": "<|im_start|>system\n{system}<|im_end|>\n",
            "user_template": "<|im_start|>user\n{user}<|im_end|>\n",
            "assistant_template": "<|im_start|>assistant\n{assistant}<|im_end|>\n",
            "assistant_start": "<|im_start|>assistant\n",
        },
        "default": {
            "system_template": "{system}\n\n",
            "user_template": "User: {user}\n",
            "assistant_template": "Assistant: {assistant}\n",
            "assistant_start": "Assistant: ",
        },
    }

    def __init__(
        self,
        tokenizer,
        system_msg: Optional[str] = None,
        *,
        vision_placeholder: Optional[str] = "<|vision|>",
    ) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided to UniversalTextFormatter")

        self.tok = tokenizer
        self.system_msg = system_msg if system_msg is not None else _DEFAULT_SYSTEM_MESSAGE
        self.vision_placeholder = vision_placeholder

        name = getattr(tokenizer, "name_or_path", str(tokenizer.__class__.__name__)).lower()
        self.model_family = self._detect_model_family(name)
        self.is_instruct = self._detect_is_instruct(name)
        self._fallback_cfg = self.FALLBACK_FORMATS.get(
            self.model_family, self.FALLBACK_FORMATS["default"]
        )
        self._assistant_start = self._fallback_cfg.get("assistant_start", "")
        self._has_chat_template = self._can_use_chat_template(self.tok)

    @staticmethod
    def _detect_model_family(name: str) -> str:
        if "llama" in name:
            return "llama"
        if "qwen" in name:
            return "qwen"
        return "default"

    @staticmethod
    def _detect_is_instruct(name: str) -> bool:
        instruct_keywords = ("instruct", "chat", "dialog", "conversation", "it")
        return any(keyword in name for keyword in instruct_keywords)

    @staticmethod
    def _can_use_chat_template(tokenizer) -> bool:
        try:
            template = getattr(tokenizer, "chat_template", None)
            return hasattr(tokenizer, "apply_chat_template") and template not in (None, "", False)
        except Exception:
            return False

    @staticmethod
    def _to_1d_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.dim() == 0: return value.view(1).clone()
            if value.dim() == 1: return value.clone()
            return value[0].clone()
        if isinstance(value, list):
            return torch.tensor(value, dtype=torch.long)
        raise TypeError(f"Unsupported type for tensor conversion: {type(value)}")

    def _format_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        if not messages:
            return ""

        if self._has_chat_template:
            try:
                template_kwargs = {
                    "add_generation_prompt": add_generation_prompt,
                    "tokenize": False,
                }
                return self.tok.apply_chat_template(messages, **template_kwargs)
            except Exception as e:
                warnings.warn(
                    f"[UniversalTextFormatter] apply_chat_template failed: {e}. "
                    f"Falling back to manual template for {self.model_family}.",
                    UserWarning
                )

        # Fallback
        cfg = self._fallback_cfg
        text_parts: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            if role == "system":
                if content:
                    text_parts.append(cfg.get("system_template", "{system}\n").format(system=content))
            elif role == "assistant":
                text_parts.append(
                    cfg.get("assistant_template", "{assistant}").format(assistant=content)
                )
            else:
                text_parts.append(cfg.get("user_template", "{user}\n").format(user=content))

        if add_generation_prompt:
            text_parts.append(cfg.get("assistant_start", ""))
        return "".join(text_parts)

    def format_conversation(
        self,
        user_msg: str,
        assistant_msg: Optional[str] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        user_text = "" if user_msg is None else str(user_msg)
        placeholder = self.vision_placeholder
        if placeholder and placeholder not in user_text:
            user_text = f"{placeholder}\n{user_text}" if user_text else placeholder

        msgs: List[Dict[str, str]] = []
        if self.system_msg:
            msgs.append({"role": "system", "content": self.system_msg})
        msgs.append({"role": "user", "content": user_text})
        if assistant_msg is not None:
            msgs.append({"role": "assistant", "content": str(assistant_msg)})
        return self._format_messages(msgs, add_generation_prompt=add_generation_prompt)

    def tokenize_messages_for_training(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 4096,
    ) -> Dict[str, Any]:
        """Tokenize messages and create labels (masking non-assistant turns)."""
        
        # 1. Full encoding
        if self._has_chat_template:
            full_enc = self.tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )
            if isinstance(full_enc, torch.Tensor):
                input_ids = self._to_1d_tensor(full_enc)
            else:
                input_ids = self._to_1d_tensor(full_enc["input_ids"])
        else:
            formatted_text = self._format_messages(messages, add_generation_prompt=False)
            enc = self.tok(
                formatted_text,
                max_length=max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = self._to_1d_tensor(enc["input_ids"])

        # Truncate
        if max_length is not None and input_ids.numel() > max_length:
            input_ids = input_ids[:max_length]
        
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)

        # 2. Masking loop
        prev_len = 0
        for idx, message in enumerate(messages):
            # Encode up to this message
            if self._has_chat_template:
                prefix_enc = self.tok.apply_chat_template(
                    messages[: idx + 1],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                )
                if isinstance(prefix_enc, torch.Tensor):
                    prefix_ids = self._to_1d_tensor(prefix_enc)
                else:
                    prefix_ids = self._to_1d_tensor(prefix_enc["input_ids"])
            else:
                prefix_txt = self._format_messages(messages[: idx + 1], add_generation_prompt=False)
                prefix_enc = self.tok(prefix_txt, padding=False, truncation=True, return_tensors="pt")
                prefix_ids = self._to_1d_tensor(prefix_enc["input_ids"])

            cur_len = min(prefix_ids.numel(), input_ids.numel())
            seg_start = min(prev_len, input_ids.numel())
            seg_end = cur_len
            
            # If assistant, unmask labels
            if message.get("role") == "assistant" and seg_start < seg_end:
                labels[seg_start:seg_end] = input_ids[seg_start:seg_end]
                
            prev_len = cur_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
