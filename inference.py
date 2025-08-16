# coding: utf-8
import os
import sys
import torch
import argparse
from pathlib import Path

# 내부 모듈 임포트
from train import VLMModule, safe_load_checkpoint
from panovlm.dataset import VLMDataModule

def load_model_with_lora(ckpt_path, lora_weights_path, device, **model_kwargs):
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = safe_load_checkpoint(ckpt_path)
    if not checkpoint:
        raise ValueError(f"Checkpoint not found: {ckpt_path}")

    model = VLMModule.load_from_checkpoint(
        ckpt_path,
        stage="finetune",
        map_location=device,
        strict=False,
        **model_kwargs
    )
    # LoRA 가중치 로드 (필요시)
    if lora_weights_path and Path(lora_weights_path).exists():
        print(f"Loading LoRA weights: {lora_weights_path}")
        lora_path = Path(lora_weights_path)
        adapter_config = lora_path / "adapter_config.json"
        adapter_model = lora_path / "adapter_model.safetensors"
        if adapter_config.exists() and adapter_model.exists():
            # LoRA 적용 코드 (사용하는 LoRA 라이브러리에 따라 다름)
            from peft import PeftModel
            model.model = PeftModel.from_pretrained(model.model, lora_weights_path)
        else:
            print("LoRA adapter files not found, skipping LoRA load.")
    else:
        print("No LoRA weights found, using base model.")

    model.eval()
    model = model.to(device)
    model.model.requires_grad_(False)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, required=True, help='체크포인트와 LoRA가 있는 폴더')
    parser.add_argument('--vision-name', default='google/siglip-base-patch16-224')
    parser.add_argument('--lm-name', default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--resampler', default='mlp')
    parser.add_argument('--max-text-length', type=int, default=256)
    parser.add_argument('--overlap-ratio', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = next(ckpt_dir.glob("*.ckpt"))
    lora_path = ckpt_dir / "lora_weights"

    model = load_model_with_lora(
        str(ckpt_path),
        str(lora_path) if lora_path.exists() else None,
        device,
        vision_name=args.vision_name,
        lm_name=args.lm_name,
        resampler=args.resampler,
        max_text_length=args.max_text_length,
        overlap_ratio=args.overlap_ratio
    )

    print("\n모델이 준비되었습니다. 질문을 입력하세요. (종료하려면 'exit' 입력)")

    user_input = ""

    # 입력 데이터 가공 (예시: 단일 샘플)
    batch = {
        "pixel_values": None,  # 실제 이미지 입력이 필요하다면 여기에 추가
        "input_ids": model.model.tokenizer(user_input, return_tensors="pt", truncation=True, max_length=args.max_text_length).input_ids.to(device),
        "attention_mask": None,
        "labels": None
    }
    # 모델 추론
    with torch.no_grad():
        output = model(**batch)

    print("\n[Model Output]")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape} | {v.detach().cpu().numpy() if v.numel() < 20 else 'Tensor'}")
        else:
            print(f"{k}: {v}")

    # 텍스트 생성 결과가 있으면 출력
    if "logits" in output:
        # 토크나이저로 디코딩 (예시)
        pred_ids = output["logits"].argmax(-1)
        try:
            decoded = model.model.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            print(f"\n[Generated Text]: {decoded}")
        except Exception:
            print("[Generated Text]: 디코딩 불가")

if __name__ == "__main__":
    main()