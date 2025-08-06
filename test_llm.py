from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
 
model_name = "openai/gpt-oss-20b"
 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)
 
messages = [
    {"role": "user", "content": "Explain what MXFP4 quantization is."},
]
 
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)
 
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7
)
 
print(tokenizer.decode(outputs[0]))