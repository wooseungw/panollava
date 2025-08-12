from typing import List, Dict
from transformers import BatchEncoding

class ConversationPromptBuilder:
    """system/user/assistant 대화 & chat_template 대응"""

    def __init__(self, tokenizer, system_msg:str|None=None, add_gen:bool=False):
        self.tok = tokenizer; self.add_gen=add_gen
        self.has_tpl = hasattr(tokenizer,"apply_chat_template") and getattr(tokenizer,"chat_template",None)
        self.msgs:List[Dict[str,str]]=[]
        # print(f"[DEBUG Builder] Received system_msg: '{system_msg}'")  # 디버깅용
        if system_msg: 
            # print(f"[DEBUG Builder] Adding system message to msgs")  # 디버깅용
            self.push("system",system_msg)

    def push(self, role:str, content:str):
        assert role in {"system","user","assistant"}
        self.msgs.append({"role":role,"content":content})

    def formatted(self):
        # VLM 모델에 더 적합한 간단한 형태로 강제 (chat_template 사용 안 함)
        txt="".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in self.msgs)
        result = txt + ("Assistant:" if self.add_gen else "")
        # print(f"[DEBUG] Messages: {self.msgs}")  # 디버깅용
        # print(f"[DEBUG] Formatted result: '{result}'")  # 디버깅용
        return result

    def tokenized(self,max_len:int|None=None) -> BatchEncoding:
        return self.tok(self.formatted(),max_length=max_len or self.tok.model_max_length,
                        padding="max_length",truncation=True,return_tensors="pt")
