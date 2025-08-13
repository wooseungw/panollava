# coding: utf-8
"""
Universal Text Formatter for Various LLM Models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

다양한 LLM 모델(Instruct/Base)에 대응하는 범용 텍스트 포맷터.
ConversationPromptBuilder 대신 사용하여 모델별 최적화된 프롬프트 생성.

지원 모델:
- Qwen/Qwen2.5-*-Instruct (ChatML 형식)
- Qwen/Qwen2.5-* (Base 모델)
- meta-llama/Llama-* (Instruct/Base)
- microsoft/DialoGPT-* 
- 기타 Base 모델들

특징:
1. 모델명 기반 자동 포맷 감지
2. Instruct vs Base 모델 구분
3. 표준 VLM 패턴 적용
4. 간단하고 안정적인 라벨 처리
"""

from typing import Optional, Dict, Any, List, Tuple
import re
import torch
from transformers import AutoTokenizer


class UniversalTextFormatter:
    """다양한 LLM 모델에 대응하는 범용 텍스트 포맷터"""
    
    # 모델별 포맷 정의
    MODEL_FORMATS = {
        # Qwen 시리즈
        "qwen2.5": {
            "instruct": {
                "system_template": "<|im_start|>system\n{system_msg}<|im_end|>\n",
                "user_template": "<|im_start|>user\n{user_msg}<|im_end|>\n",
                "assistant_template": "<|im_start|>assistant\n{assistant_msg}<|im_end|>\n",
                "assistant_start": "<|im_start|>assistant\n",
                "end_token": "<|im_end|>",
                "generation_stop": ["<|im_end|>", "<|endoftext|>"]
            },
            "base": {
                "system_template": "System: {system_msg}\n\n",
                "user_template": "User: {user_msg}\n\n",
                "assistant_template": "Assistant: {assistant_msg}\n\n",
                "assistant_start": "Assistant: ",
                "end_token": "\n\n",
                "generation_stop": ["\n\nUser:", "\n\nSystem:", "<|endoftext|>"]
            }
        },
        
        # Llama 시리즈
        "llama": {
            "instruct": {
                "system_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>",
                "user_template": "<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>",
                "assistant_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>",
                "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "end_token": "<|eot_id|>",
                "generation_stop": ["<|eot_id|>", "<|end_of_text|>"]
            },
            "base": {
                "system_template": "{system_msg}\n\n",
                "user_template": "### Human: {user_msg}\n\n",
                "assistant_template": "### Assistant: {assistant_msg}\n\n",
                "assistant_start": "### Assistant: ",
                "end_token": "\n\n",
                "generation_stop": ["\n\n### Human:", "\n\n### System:", "</s>"]
            }
        },
        
        # 기본 포맷 (모든 Base 모델용)
        "default": {
            "base": {
                "system_template": "{system_msg}\n\n",
                "user_template": "User: {user_msg}\n\n",
                "assistant_template": "Assistant: {assistant_msg}\n\n",
                "assistant_start": "Assistant: ",
                "end_token": "\n\n",
                "generation_stop": ["\n\nUser:", "\n\nSystem:", "<|endoftext|>", "</s>"]
            }
        }
    }
    
    def __init__(self, tokenizer_name_or_path: str, system_msg: Optional[str] = None):
        """
        Args:
            tokenizer_name_or_path: 토크나이저 이름 또는 경로
            system_msg: 기본 시스템 메시지
        """
        self.tokenizer_name = tokenizer_name_or_path.lower()
        self.system_msg = system_msg or "You are a helpful AI assistant specialized in analyzing panoramic images."
        
        # 모델 타입 감지
        self.model_family, self.is_instruct = self._detect_model_type()
        self.format_config = self._get_format_config()
        
        print(f"[TextFormatter] Detected: {self.model_family} ({'Instruct' if self.is_instruct else 'Base'})")
        print(f"[TextFormatter] Assistant start: '{self.format_config['assistant_start']}'")

    # === 신규: chat_template 사용 가능 여부 ===
    def _can_use_chat_template(self, tokenizer) -> bool:
        try:
            return (
                tokenizer is not None
                and hasattr(tokenizer, "apply_chat_template")
                and getattr(tokenizer, "chat_template", None) not in (None, "", False)
            )
        except Exception:
            return False

    # === 신규: messages -> 문자열 포맷팅(CT 우선, 폴백 지원) ===
    def _format_messages(self, messages: List[Dict[str, str]], tokenizer=None,
                         add_generation_prompt: bool = False) -> str:
        """
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        tokenizer가 chat_template를 지원하면 그 경로로, 아니면 하드코딩 템플릿으로.
        """
        if self._can_use_chat_template(tokenizer) and self.is_instruct:
            # HF 표준 시그니처: tokenize=False → 순수 문자열 반환
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False
            )

        # ---- 폴백: 기존 하드코딩 포맷 사용 ----
        txt = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system" and "system_template" in self.format_config:
                txt += self.format_config["system_template"].format(system_msg=content)
            elif role == "user":
                txt += self.format_config["user_template"].format(user_msg=content)
            elif role == "assistant":
                txt += self.format_config["assistant_template"].format(assistant_msg=content)
            else:
                # 알 수 없는 role → user로 다운캐스팅
                txt += self.format_config["user_template"].format(user_msg=content)

        if add_generation_prompt:
            txt += self.format_config["assistant_start"]
        return txt
    
    def _detect_model_type(self) -> Tuple[str, bool]:
        """모델명으로부터 모델 계열과 Instruct 여부 감지"""
        name = self.tokenizer_name
        
        # Instruct 모델 감지
        is_instruct = any(keyword in name for keyword in [
            "instruct", "chat", "dialog", "conversation", "it"
        ])
        
        # 모델 계열 감지
        if "qwen" in name:
            return "qwen2.5", is_instruct
        elif "llama" in name:
            return "llama", is_instruct
        else:
            # 알 수 없는 모델은 Base로 간주
            return "default", False
    
    def _get_format_config(self) -> Dict[str, str]:
        """현재 모델에 맞는 포맷 설정 반환"""
        model_config = self.MODEL_FORMATS.get(self.model_family, self.MODEL_FORMATS["default"])
        
        if self.is_instruct and "instruct" in model_config:
            return model_config["instruct"]
        else:
            return model_config["base"]
    
    # === 호환성 유지: 단일 턴 대화 빌더 (기존 호출부가 깨지지 않도록 유지) ===
    def format_conversation(self, user_msg: str, assistant_msg: Optional[str] = None,
                            add_generation_prompt: bool = False, tokenizer=None) -> str:
        """
        대화 형식으로 텍스트 포맷팅 - chat_template 우선 사용
        
        Args:
            user_msg: 사용자 메시지
            assistant_msg: 어시스턴트 응답 (None이면 생성용)
            add_generation_prompt: 생성용 프롬프트 추가 여부
            tokenizer: HuggingFace 토크나이저 (chat_template 사용 시 필요)
        
        Returns:
            포맷팅된 텍스트
        """
        msgs = []
        if self.system_msg:
            msgs.append({"role": "system", "content": self.system_msg})
        msgs.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            msgs.append({"role": "assistant", "content": assistant_msg})
        return self._format_messages(msgs, tokenizer=tokenizer,
                                     add_generation_prompt=add_generation_prompt)
    
    # === 학습용(단일 턴) ===
    def tokenize_for_training(self, user_msg: str, assistant_msg: str,
                              tokenizer, max_length: int = 512) -> Dict[str, Any]:
        """
        학습용 토큰화 - 표준 VLM 방식
        
        Args:
            user_msg: 사용자 메시지
            assistant_msg: 어시스턴트 응답
            tokenizer: HuggingFace 토크나이저
            max_length: 최대 시퀀스 길이
        
        Returns:
            input_ids, attention_mask, labels를 포함한 딕셔너리
        """
        full_text = self.format_conversation(user_msg, assistant_msg, tokenizer=tokenizer)

        enc = tokenizer(
            full_text,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc.get("attention_mask",
                       (input_ids != tokenizer.pad_token_id).long()).squeeze(0)

        labels = input_ids.clone()

        # --- 기존 라벨 마스킹 유지: assistant_start 토큰 시퀀스 탐색 ---
        assistant_start_text = self.format_config["assistant_start"]
        try:
            assistant_tokens = tokenizer(
                assistant_start_text, add_special_tokens=False
            )["input_ids"]
            pos = self._find_subsequence(input_ids.tolist(), assistant_tokens)
            if pos is not None:
                labels[:pos + len(assistant_tokens)] = -100
            else:
                # chat_template와 문자열 상이로 탐색 실패 시: 시스템/유저 길이로 대체 탐색
                # (더 안전한 보정) system+user만 포함한 텍스트 길이를 기준으로 마스킹
                prev_txt = self._format_messages(
                    ([{"role": "system", "content": self.system_msg}] if self.system_msg else []) + 
                    [{"role": "user", "content": user_msg}],
                    tokenizer=tokenizer, add_generation_prompt=False
                )
                prev_tok = tokenizer(
                    prev_txt,
                    max_length=max_length,
                    padding=False,
                    truncation=True,
                    return_tensors=None
                )["input_ids"]
                cut = min(len(prev_tok), input_ids.size(0))
                labels[:cut] = -100
        except Exception:
            # 실패 시 전체 학습(디버그 편의)
            pass

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "formatted_text": full_text,
        }

    # === (옵션) 멀티턴 학습용: messages 기반 ===
    def tokenize_messages_for_training(self, messages: List[Dict[str, str]],
                                       tokenizer, max_length: int = 2048) -> Dict[str, Any]:
        """
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        - system/user는 -100, assistant만 라벨 유지
        - chat_template 우선 적용
        """
        full_txt = self._format_messages(messages, tokenizer=tokenizer,
                                         add_generation_prompt=False)
        enc = tokenizer(
            full_txt,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc.get("attention_mask",
                       (input_ids != tokenizer.pad_token_id).long()).squeeze(0)
        labels = input_ids.clone()
        labels[:] = -100  # 기본 -100

        # 메시지 경계 기반으로 assistant 구간만 라벨 on
        # 아이디어: (i) 직전까지의 대화 → 토큰화 길이(prev_len),
        #          (ii) 해당 메시지까지의 대화 → 토큰화 길이(cur_len),
        #          두 길이 차이 구간을 해당 role의 라벨로 설정
        prev_len = 0
        for i in range(len(messages)):
            cur_txt = self._format_messages(messages[:i+1], tokenizer=tokenizer,
                                            add_generation_prompt=False)
            cur_tok = tokenizer(
                cur_txt,
                max_length=max_length, padding=False, truncation=True, return_tensors=None
            )["input_ids"]
            cur_len = min(len(cur_tok), input_ids.size(0))

            # 이전 길이(prev_len)가 잘려서 현재 시퀀스보다 클 수는 없도록 보정
            seg_start = min(prev_len, input_ids.size(0))
            seg_end = cur_len

            role = messages[i]["role"]
            if role == "assistant" and seg_start < seg_end:
                labels[seg_start:seg_end] = input_ids[seg_start:seg_end]

            prev_len = cur_len

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "formatted_text": full_txt,
        }
    
    # === 생성용 ===
    def tokenize_for_generation(self, user_msg: str, tokenizer,
                                max_length: int = 512) -> Dict[str, Any]:
        """
        생성용 토큰화
        
        Args:
            user_msg: 사용자 메시지
            tokenizer: HuggingFace 토크나이저
            max_length: 최대 시퀀스 길이
        
        Returns:
            input_ids, attention_mask를 포함한 딕셔너리
        """
        prompt_text = self.format_conversation(user_msg, tokenizer=tokenizer,
                                               add_generation_prompt=True)
        enc = tokenizer(
            prompt_text,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc.get("attention_mask",
                       (input_ids != tokenizer.pad_token_id).long()).squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "formatted_text": prompt_text,
            "generation_stop_strings": self.format_config["generation_stop"],
        }
    
    def extract_assistant_response(self, generated_text: str) -> str:
        """
        생성된 텍스트에서 Assistant 응답 부분만 추출
        
        Args:
            generated_text: 모델이 생성한 전체 텍스트
        
        Returns:
            정리된 Assistant 응답
        """
        # Assistant 시작 부분 이후 텍스트 추출
        assistant_start = self.format_config["assistant_start"]
        
        if assistant_start in generated_text:
            response = generated_text.split(assistant_start, 1)[-1]
        else:
            # Assistant 패턴이 없으면 전체 텍스트 사용
            response = generated_text
        
        # 종료 토큰들 제거
        for stop_token in self.format_config["generation_stop"]:
            if stop_token in response:
                response = response.split(stop_token)[0]
        
        # 기본 정리
        response = response.strip()
        
        # 연속된 느낌표 제거 (3개 이상의 느낌표는 불필요한 것으로 간주)
        import re
        response = re.sub(r'!{3,}', '', response)
        response = re.sub(r'\n!\s*$', '', response)  # 줄 끝 단독 느낌표
        response = re.sub(r'\n!\s*\n', '\n', response)  # 줄 중간 단독 느낌표
        response = response.strip()
        
        # 빈 응답 처리
        if not response or not any(c.isalnum() for c in response):
            response = "I can see a panoramic view."
        
        return response
    
    # === 기존 유틸 유지 ===
    def _find_subsequence(self, sequence: List[int], subsequence: List[int]) -> Optional[int]:
        """시퀀스에서 부분 시퀀스의 시작 위치 찾기"""
        if not subsequence:
            return None
        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i + len(subsequence)] == subsequence:
                return i
        return None

    def get_generation_config(self) -> Dict[str, Any]:
        """생성용 설정 반환"""
        return {
            "stop_strings": self.format_config["generation_stop"],
            "assistant_start_token": self.format_config["assistant_start"],
            "end_token": self.format_config["end_token"],
        }


# 편의 함수들
def create_formatter(tokenizer_name: str, system_msg: Optional[str] = None) -> UniversalTextFormatter:
    """편의용 포맷터 생성 함수"""
    return UniversalTextFormatter(tokenizer_name, system_msg)


def format_simple_instruction(user_msg: str, assistant_msg: Optional[str] = None, 
                            model_name: str = "default") -> str:
    """간단한 Instruction 포맷팅 (ConversationPromptBuilder 대체)"""
    formatter = UniversalTextFormatter(model_name)
    return formatter.format_conversation(user_msg, assistant_msg)