# coding: utf-8
"""
Universal Text Formatter (Chat-template-first)
──────────────────────────────────────────────
- 목표: 다양한 LLM에 대해, '모델이 제공하는 chat_template'를 최우선으로 사용.
- 전제: 텍스트 전용(비전 토큰·툴 호출 불필요).
- 특징:
  1) chat_template 있으면 항상 apply_chat_template 우선 적용(HF 권장)
  2) 멀티턴 학습 라벨 마스킹: tokenize=True로 prefix 길이 차를 이용해 정확히 계산
  3) 템플릿이 없을 때만 최소한의 폴백 템플릿 사용(라이트 버전)
"""

from typing import Optional, Dict, Any, List
import re
import torch


_DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant specialized in analyzing panoramic images."
)


class UniversalTextFormatter:
    """chat-template-first LLM 텍스트 포맷터(텍스트 전용)"""

    # 가능한 한 단순하게 유지 (모델 카드가 제공하는 chat_template가 정답)
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

    # ---------------------------
    # 헬퍼
    # ---------------------------
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
            if value.dim() == 0:
                return value.view(1).clone()
            if value.dim() == 1:
                return value.clone()
            return value[0].clone()
        if isinstance(value, list):
            return torch.tensor(value, dtype=torch.long)
        raise TypeError(f"Unsupported type for tensor conversion: {type(value)}")

    # ---------------------------
    # 메시지 → 문자열 (템플릿 우선)
    # ---------------------------
    def _format_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        if not messages:
            return ""

        if self._has_chat_template:
            try:
                # 선택적 파라미터를 딕셔너리로 관리 (모델별 호환성)
                template_kwargs = {
                    "add_generation_prompt": add_generation_prompt,
                    "tokenize": False,
                    "enable_thinking": False,  # DeepSeek-R1 등에서 thinking 모드 비활성화
                }
                return self.tok.apply_chat_template(messages, **template_kwargs)
            except Exception as e:
                # apply_chat_template 실패 시 명시적 경고
                import warnings
                warnings.warn(
                    f"[UniversalTextFormatter] apply_chat_template failed: {e}. "
                    f"Falling back to manual template for {self.model_family}. "
                    f"This may cause formatting inconsistencies.",
                    UserWarning
                )
        else:
            # chat_template이 없는 경우 한 번만 경고
            if not hasattr(self, '_warned_no_template'):
                import warnings
                warnings.warn(
                    f"[UniversalTextFormatter] No chat_template found for tokenizer. "
                    f"Using fallback template for {self.model_family}. "
                    f"Consider using a model with official chat_template support.",
                    UserWarning
                )
                self._warned_no_template = True

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

    # ---------------------------
    # 단일턴 헬퍼
    # ---------------------------
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

    # ---------------------------
    # 생성용 토큰화
    # ---------------------------
    def tokenize_for_generation(
        self,
        user_msg: str,
        max_length: int = 2048,
    ) -> Dict[str, Any]:
        prompt_text = self.format_conversation(
            user_msg=user_msg,
            add_generation_prompt=True,
        )

        enc = self.tok(
            prompt_text,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = self._to_1d_tensor(enc["input_ids"])
        attention_mask_value = enc.get("attention_mask")
        if attention_mask_value is not None:
            attention_mask = self._to_1d_tensor(attention_mask_value).to(torch.long)
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "formatted_text": prompt_text,
            "eos_token_id": getattr(self.tok, "eos_token_id", None),
        }

    # ---------------------------
    # 학습용(멀티턴) 토큰화: assistant만 라벨 on
    # ---------------------------
    def tokenize_messages_for_training(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 4096,
    ) -> Dict[str, Any]:
        formatted_text = self._format_messages(messages, add_generation_prompt=False)

        if self._has_chat_template:
            full_enc = self.tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )

            # HF 4.41+ returns a tensor directly; older versions return BatchEncoding
            if isinstance(full_enc, torch.Tensor):
                input_ids = self._to_1d_tensor(full_enc)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                input_ids = self._to_1d_tensor(full_enc["input_ids"])
                attention_mask_value = full_enc.get("attention_mask")
                if attention_mask_value is not None:
                    attention_mask = self._to_1d_tensor(attention_mask_value).to(torch.long)
                else:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            enc = self.tok(
                formatted_text,
                max_length=max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = self._to_1d_tensor(enc["input_ids"])
            attention_mask_value = enc.get("attention_mask")
            if attention_mask_value is not None:
                attention_mask = self._to_1d_tensor(attention_mask_value).to(torch.long)
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if max_length is not None and input_ids.numel() > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        labels = torch.full_like(input_ids, -100)

        prev_len = 0
        for idx, message in enumerate(messages):
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
                prefix_txt = self._format_messages(
                    messages[: idx + 1], add_generation_prompt=False
                )
                prefix_enc = self.tok(
                    prefix_txt,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prefix_ids = self._to_1d_tensor(prefix_enc["input_ids"])

            cur_len = min(prefix_ids.numel(), input_ids.numel())
            seg_start = min(prev_len, input_ids.numel())
            seg_end = cur_len
            if message.get("role") == "assistant" and seg_start < seg_end:
                labels[seg_start:seg_end] = input_ids[seg_start:seg_end]
            prev_len = cur_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "formatted_text": formatted_text,
        }

    # ---------------------------
    # 학습용(단일턴) 헬퍼
    # ---------------------------
    def tokenize_for_training(
        self,
        user_msg: str,
        assistant_msg: str,
        max_length: int = 4096,
    ) -> Dict[str, Any]:
        msgs: List[Dict[str, str]] = []
        if self.system_msg:
            msgs.append({"role": "system", "content": self.system_msg})
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": assistant_msg})
        return self.tokenize_messages_for_training(msgs, max_length=max_length)

    # ---------------------------
    # 생성 설정(요약)
    # ---------------------------
    def get_generation_config(self) -> Dict[str, Any]:
        return {
            "eos_token_id": getattr(self.tok, "eos_token_id", None),
            "pad_token_id": getattr(self.tok, "pad_token_id", None),
            "stop_strings": [],
        }

    # ---------------------------
    # 생성 결과 후처리
    # ---------------------------
    def extract_assistant_response(self, generated_text: str) -> str:
        response = (generated_text or "").strip()

        if self._assistant_start and response.startswith(self._assistant_start):
            response = response[len(self._assistant_start) :]

        eos_tokens = [getattr(self.tok, "eos_token", None), getattr(self.tok, "bos_token", None)]
        for token in eos_tokens:
            if token and token in response:
                response = response.split(token)[0]

        response = re.sub(r"!{3,}", "", response)
        response = re.sub(r"\n!\s*$", "", response)
        response = re.sub(r"\n!\s*\n", "\n", response)
        response = response.strip()

        if not response or not any(ch.isalnum() for ch in response):
            response = "I can see a panoramic view."
        return response


# ---------------------------
# 편의 함수
# ---------------------------
def create_formatter(tokenizer, system_msg: Optional[str] = None, **kwargs: Any) -> UniversalTextFormatter:
    if isinstance(tokenizer, str):
        raise TypeError("create_formatter expects a tokenizer instance, not a model name string.")
    return UniversalTextFormatter(tokenizer, system_msg, **kwargs)


def format_simple_instruction(
    user_msg: str,
    assistant_msg: Optional[str] = None,
    *,
    tokenizer,
    system_msg: Optional[str] = None,
) -> str:
    formatter = UniversalTextFormatter(tokenizer, system_msg)
    return formatter.format_conversation(user_msg, assistant_msg)
