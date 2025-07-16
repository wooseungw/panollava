from panovlm import VLMModule
from panovlm.utils import DEVICE, VISION_NAME, LM_NAME
from panovlm.dataset import ChatPanoDataset
from panovlm.processors import PanoramaImageProcessor, TextTokenizer
from panovlm.pano_llava_processor import PanoLLaVAProcessor

# ① 파노라마 → 멀티뷰 변환
img_proc = PanoramaImageProcessor(
    image_size    =(224, 224),
    crop_strategy ="e2p",   # 또는 "e2p" / "cubemap"
    fov_deg       =90,
    overlap_ratio =0.5,
    normalize= True,
)

# ② 텍스트 토크나이저
txt_tok  = TextTokenizer("Qwen/Qwen3-0.6B")

# ③ (선택) CLIP·ViT용 비전 프로세서
vis_proc = VisionProcessorWrapper("google/siglip-so400m-patch14-384")

# ④ 최종 통합 Processor
pano_proc = PanoLLaVAProcessor(img_proc, txt_tok, vis_proc)

# ── (2) 대화 프롬프트 빌더 ───────────────────────────────────
builder = ConversationPromptBuilder(
    txt_tok.tok,
    system_msg="당신은 360° 파노라마 분석 전문가입니다.",
)

builder.push("user", "이 방은 어떤 스타일인가요?")

# ── (3) 실제 데이터 전처리 ──────────────────────────────────
image = Image.open(sample_image).convert("RGB")

batch = pano_proc(
    pil     = image,
    builder = builder,
    flatten = True                 # (B, V, C, H, W) 그대로 유지
)

print(batch.keys())
print("pixel_values shape:", batch["pixel_values"].shape)  # (B, V, C, H, W)
print("input_ids shape:", batch["input_ids"].shape)  # (B, L)
print("attention_mask shape:", batch["attention_mask"].shape)  # (B, L
# dict_keys(['pixel_values', 'pixel_values_5d', 'input_ids', 'attention_mask'

num_v = batch["pixel_values"].shape[0]

visualize_views(
    batch["pixel_values"],
    title = f"Sliding Window – {num_v} Views",
    titles= [ f"View {i+1}" for i in range(num_v) ],
    show_plot= True
)