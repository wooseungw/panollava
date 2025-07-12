# conversation_builder.py
# ---------------------------------------------------------------
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, BatchEncoding

class ConversationPromptBuilder:
    """
    - 대화 내역(역할·메시지)을 리스트 형태로 유지
    - 토크나이저 chat_template(*) 존재 시 적용
      * Transformers ≥4.40
    - history truncate / role 변경 등 유틸리티 제공
    """

    def __init__(
        self,
        tokenizer,                          # 이미 로드된 AutoTokenizer
        system_prompt: str | None = None,   # 첫 system 메시지
        add_generation_prompt: bool = True, # <assistant> 토큰 삽입
    ):
        self.tokenizer = tokenizer
        self.config    = AutoConfig.from_pretrained(tokenizer.name_or_path)
        self.has_template = hasattr(tokenizer, "apply_chat_template") and \
                            getattr(tokenizer, "chat_template", None)

        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.push("system", system_prompt)

        self.add_generation_prompt = add_generation_prompt

    # ----------------------------------------------------------- #
    # 메시지 조작
    # ----------------------------------------------------------- #
    def push(self, role: str, content: str) -> None:
        assert role in {"system", "user", "assistant"}, "role must be system/user/assistant"
        self.messages.append({"role": role, "content": content})

    def pop(self) -> Dict[str, str]:
        return self.messages.pop()

    def reset(self, keep_system: bool = True) -> None:
        if keep_system and self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def truncate(self, max_tokens: int) -> None:
        """
        전체 token 길이가 max_tokens 를 넘지 않도록 오래된 user/assistant 쌍부터 삭제
        (system 메시지는 유지)
        """
        while True:
            enc = self.to_tokenized(max_length=max_tokens, truncation=False)
            if enc["input_ids"].shape[-1] <= max_tokens:
                break
            # system 다음 인덱스부터 2개(user+assistant) 씩 제거
            if len(self.messages) > 1:
                del self.messages[1:3]
            else:
                break

    # ----------------------------------------------------------- #
    # 포맷 변환
    # ----------------------------------------------------------- #
    def to_formatted_str(self) -> str:
        if self.has_template:
            return self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=self.add_generation_prompt,
            )
        # 템플릿이 없는 경우: 간단한 role prefix 포맷
        prompt = ""
        for m in self.messages:
            prompt += f"{m['role'].capitalize()}: {m['content']}\n"
        if self.add_generation_prompt:
            prompt += "Assistant:"
        return prompt

    def to_tokenized(
        self,
        max_length: int | None = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        formatted = self.to_formatted_str()
        max_len   = max_length or getattr(self.config, "model_max_length", 2048)
        return self.tokenizer(
            formatted,
            max_length=max_len,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    # 편의 함수
    __call__ = to_tokenized

# -----------------------------------------------------------------
# 사용 예시 (train / inference 스크립트에서)
# -----------------------------------------------------------------
if __name__ == "__main__":
    model_name = "qwen/Qwen3-0.6B"
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=None)
    builder    = ConversationPromptBuilder(
        tokenizer,
        system_prompt="당신은 친절한 파노라마 분석 비서입니다.",
    )

    # 대화 추가
    builder.push("user", "이 방은 어떤 스타일인가요?")
    enc = builder.to_tokenized()

    #--- 모델 불러오기 & generate -----------------------------------
    from transformers import AutoModelForCausalLM
    import torch, os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids   = enc["input_ids"].to(device),
            attention_mask = enc["attention_mask"].to(device),
            max_new_tokens=128,
            temperature=0.7,
        )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n--- Assistant ---")
    print(answer.split("Assistant:")[-1].lstrip())