import torch
from transformers import AutoModelForCausalLM

def _choose_dtype():
    if torch.cuda.is_available():
        # Ampere(H100/A100/RTX30+) 이상이면 보통 bf16 OK
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32

def _fa2_possible():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    if major < 8:  # SM80 이상 (Ampere+) 권장
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False

def load_causal_lm_safe(lm_name: str, enable_gc: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = _choose_dtype()

    # 시도 순서: flash_attention_2 → sdpa → eager
    attn_candidates = []
    if _fa2_possible():
        attn_candidates.append("flash_attention_2")
    # PyTorch SDPA가 대부분 기본 제공
    attn_candidates.append("sdpa")
    attn_candidates.append("eager")

    last_err = None
    for impl in attn_candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                lm_name,
                torch_dtype=dtype,
                attn_implementation=impl,
                low_cpu_mem_usage=True,
            )
            if enable_gc and device == "cuda":
                # 길이 긴 시퀀스 학습 시 메모리 여유 확보
                model.gradient_checkpointing_enable()
            model.to(device)
            # 확인용: 현재 attention 구현과 dtype 로깅
            print(f"✓ Loaded {lm_name} with attn='{impl}', dtype={dtype}, device={device}")
            # (선택) config에 실제 사용 impl 기록
            try:
                model.config._attn_implementation = impl
            except Exception:
                pass
            return model
        except Exception as e:
            print(f"↪︎ Fallback from attn='{impl}' due to: {e}")
            last_err = e

    raise RuntimeError(f"모든 attention 구현 로드 실패: {last_err}")

if __name__ == "__main__":
    # 테스트
    model = load_causal_lm_safe("Qwen/Qwen2.5-0.5B-Instruct")
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
    outputs = model(input_ids)
    print("Output logits shape:", outputs.logits.shape)