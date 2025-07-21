from typing import List, Dict
from transformers import BatchEncoding

class ConversationPromptBuilder:
    """system/user/assistant 대화 & chat_template 대응"""

    def __init__(self, tokenizer, system_msg:str|None=None, add_gen:bool=False):
        self.tok = tokenizer; self.add_gen=add_gen
        self.has_tpl = hasattr(tokenizer,"apply_chat_template") and getattr(tokenizer,"chat_template",None)
        self.msgs:List[Dict[str,str]]=[]
        if system_msg: self.push("system",system_msg)

    def push(self, role:str, content:str):
        assert role in {"system","user","assistant"}
        self.msgs.append({"role":role,"content":content})

    def build_for_training(self, user_query: str, assistant_response: str):
        """학습용: user query와 assistant response를 모두 추가"""
        self.push("user", user_query)
        self.push("assistant", assistant_response)
        return self

    def build_for_evaluation(self, user_query: str):
        """평가용: user query만 추가, assistant response는 생성을 위해 비워둠"""
        self.push("user", user_query)
        # 평가 시에는 assistant 응답을 추가하지 않음
        return self

    def formatted(self):
        if self.has_tpl:
            return self.tok.apply_chat_template(self.msgs, tokenize=False, add_generation_prompt=self.add_gen)
        txt="".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in self.msgs)
        return txt + ("Assistant:" if self.add_gen else "")

    def tokenized(self,max_len:int|None=None) -> BatchEncoding:
        return self.tok(self.formatted(),max_length=max_len or self.tok.model_max_length,
                        padding="max_length",truncation=True,return_tensors="pt")
