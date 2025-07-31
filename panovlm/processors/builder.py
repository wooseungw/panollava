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

    def formatted(self):
        if self.has_tpl:
            return self.tok.apply_chat_template(self.msgs, tokenize=False, add_generation_prompt=self.add_gen)
        txt="".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in self.msgs)
        return txt + ("Assistant:" if self.add_gen else "")

    def tokenized(self,max_len:int|None=None) -> BatchEncoding:
        return self.tok(self.formatted(),max_length=max_len or self.tok.model_max_length,
                        padding="max_length",truncation=True,return_tensors="pt")
