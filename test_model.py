from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
from transformers import default_data_collator
from panovlm.dataset import ChatPanoDataset
from panovlm.dataset import ChatPanoTestDataset
from panovlm.processors.pano_llava_processor import PanoLLaVAProcessor
from panovlm.processors.image import PanoramaImageProcessor
from panovlm.processors.text import TextTokenizer
from panovlm.model import PanoramaVLM
from train import VLMModule  # LightningModule 래퍼 사용

print("--- 1. 가상 데이터 및 환경 설정 ---")
csv_path = "data/quic360/downtest.csv"
if not Path(csv_path).exists():
    raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_path}")
print("가상 CSV 파일 경로:", csv_path)
print("\n--- 2. 데이터 로딩 파이프라인 테스트 ---")
VISION_NAME = "google/siglip-base-patch16-224"
LM_NAME = "Qwen/Qwen3-0.6B"
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "mps" if torch.backends.mps.is_available() else DEVICE  # MPS 지원 여부 확인

img_proc = PanoramaImageProcessor()
txt_tok = TextTokenizer(LM_NAME)
processor = PanoLLaVAProcessor(img_proc, txt_tok, max_length=128)

dataset = ChatPanoDataset(csv_path, processor, txt_tok.tok, flatten=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=__import__('panovlm.dataset').dataset.ChatPanoDataModule.custom_collate_fn)

# --- 테스트용 generate 데이터셋 준비 ---
test_dataset = ChatPanoTestDataset(csv_path, processor, txt_tok.tok, flatten=False)
test_sample = test_dataset[0]  # 첫 샘플만 사용
print(f"데이터셋 샘플 수: {len(dataset)}, 배치 크기: {BATCH_SIZE}")

print("\n--- 3. 모델 학습 과정 테스트 (VLMModule 래퍼 기반) ---")
try:
    # VLMModule은 LightningModule이지만, 내부적으로 PanoramaVLM을 래핑하며 stage별 freeze 로직을 포함
    model = VLMModule(
        vision_name=VISION_NAME,
        lm_name=LM_NAME,
        resampler="mlp",
        stage="vision",
        lr=1e-5
    )
    model = model.to(DEVICE)
    model.train()

    batch = next(iter(dataloader))
    print(f"배치 크기: {batch['pixel_values'].shape}, 입력 ID 크기: {batch['input_ids'].shape}")
    batch = {k: (v.to(DEVICE) if hasattr(v, 'to') else v) for k, v in batch.items()}
    print("=======입력 텍스트=======")
    for i in batch["input_text"]:
        print(i[:200])
        print("-----------------------------------")
    print("======================")
    # Vision stage 테스트
    print("\n=== Vision Stage 테스트 ===")
    model._freeze_for_stage("vision")
    optimizer_vision = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    outputs = model.forward(**batch)
    loss = outputs["loss"]
    print(f"✅ Vision 순전파 성공! Loss: {loss.item():.4f}")
    optimizer_vision.zero_grad()
    loss.backward()
    print("✅ Vision 역전파 성공! 그래디언트 계산 완료.")
    optimizer_vision.step()
    print("✅ Vision 옵티마이저 스텝 성공!")
    
    # Finetune stage 테스트 (freeze 자동 적용)
    print("\n=== Finetune Stage 테스트 (LLM & Vision Encoder Frozen) ===")
    model._freeze_for_stage("finetune")
    optimizer_finetune = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    model._stage_key = "finetune"
    outputs = model.forward(**batch)
    loss = outputs["loss"]
    print(f"✅ Finetune 순전파 성공! Loss: {loss.item():.4f}")
    optimizer_finetune.zero_grad()
    loss.backward()
    print("✅ Finetune 역전파 성공! 그래디언트 계산 완료.")
    optimizer_finetune.step()
    print("✅ Finetune 옵티마이저 스텝 성공!")

    # Generation 테스트
    print("\n=== Generation Stage 테스트 ===")
    model.eval()
    with torch.no_grad():
        # 새로운 독립적인 generate 함수 사용
        pixel_values = batch["pixel_values"]
        input_ids = batch.get("input_ids", None)
        # 1. 질문 없이 캡셔닝 테스트
        print("1. 캡셔닝 테스트 (질문 없음):")
        caption_result = model.model.generate(
            pixel_values=pixel_values,
            max_new_tokens=50,
            temperature=0.7
        )
        print(f"✅ 캡셔닝 성공! 생성된 텍스트: {caption_result['text'][0][:100]}...")
        
        # 2. 질문-답변 VQA 테스트 (input_ids가 있는 경우)
        if input_ids is not None:
            print("\n2. VQA 테스트 (질문-답변):")
            vqa_result = model.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=64,
                temperature=0.8
            )
            print(f"✅ VQA 성공! 생성된 답변: {vqa_result['text'][0][:100]}...")
        
        # 3. 배치 크기 1로 테스트
        print("\n3. 단일 이미지 테스트:")
        single_result = model.model.generate(
            pixel_values=pixel_values[:1],
            max_new_tokens=32,
            temperature=0.5
        )
        print(f"✅ 단일 이미지 생성 성공! 텍스트: {single_result['text'][0][:100]}...")

    print("\n🎉 모든 테스트 통과: 데이터 로딩 및 학습 파이프라인이 정상적으로 작동합니다.")

except Exception as e:
    import traceback
    print(f"\n❌ 테스트 실패: 모델 학습 과정에서 오류가 발생했습니다.")
    traceback.print_exc()
