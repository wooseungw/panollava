
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
from transformers import default_data_collator
from panovlm.dataset import ChatPanoDataset
from panovlm.processors.pano_llava_processor import PanoLLaVAProcessor
from panovlm.processors.image import PanoramaImageProcessor
from panovlm.processors.text import TextTokenizer
from panovlm.model import PanoramaVLM


print("--- 1. 가상 데이터 및 환경 설정 ---")

csv_path = "data/quic360/downtest.csv"


print("\n--- 2. 데이터 로딩 파이프라인 테스트 ---")
# 빠른 테스트를 위해 작은 모델 사용
VISION_NAME = "google/siglip-base-patch16-224"
LM_NAME = "Qwen/Qwen3-0.6B"
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

img_proc = PanoramaImageProcessor()
txt_tok = TextTokenizer(LM_NAME)
processor = PanoLLaVAProcessor(img_proc, txt_tok)

dataset = ChatPanoDataset(csv_path, processor, txt_tok.tok,flatten=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=default_data_collator)
print(f"데이터셋 샘플 수: {len(dataset)}, 배치 크기: {BATCH_SIZE}")

print("\n--- 3. 모델 학습 과정 테스트 ---")
try:
    model = PanoramaVLM(vision_name=VISION_NAME, lm_name=LM_NAME).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    batch = next(iter(dataloader))
    print(f"배치 크기: {batch['pixel_values'].shape}, 입력 ID 크기: {batch['input_ids'].shape}")
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    # print("배치 데이터 로드 완료. 모델에 입력을 전달합니다...")

    # 순전파 (Forward Pass)
    outputs = model(stage="vision", **batch)
    loss = outputs["loss"]
    print(f"✅ 순전파(Forward) 성공! Loss: {loss.item():.4f}")
    
    # 역전파 (Backward Pass)
    optimizer.zero_grad()
    loss.backward()
    print("✅ 역전파(Backward) 성공! 그래디언트 계산 완료.")

    outputs = model(stage="finetune", **batch)
    loss = outputs["loss"]
    print(f"✅ 순전파(Forward) 성공! Loss: {loss.item():.4f}")
    
    # outputs = model(stage="generate", **batch)
    # loss = outputs["loss"]
    # print(f"✅ 순전파(Forward) 성공! Loss: {loss.item():.4f}")
    
    

    # 파라미터 업데이트
    optimizer.step()
    print("✅ 옵티마이저(Optimizer) 스텝 성공!")
    print("\n🎉 모든 테스트 통과: 데이터 로딩 및 학습 파이프라인이 정상적으로 작동합니다.")

except Exception as e:
    import traceback
    print(f"\n❌ 테스트 실패: 모델 학습 과정에서 오류가 발생했습니다.")
    traceback.print_exc()
