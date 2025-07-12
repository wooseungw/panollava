# coding: utf-8
"""
PanoramaVLM – 수정 버전
▪ 주요 수정 사항
  1. Vision 모델 호출 방식 통일 (keyword arg 사용)
  2. _vis_dim 로직 강화 – hidden_size 탐색 실패 시 fallback 제거
  3. Resampler 출력 규격 통일  → (B*V, N, D)
  4. CLIP Temperature 계산 오류 수정  (sim / T)
  5. Finetune 레이블 시프트 및 반복 로직 보강
  6. BOS 토큰 ID 안전 처리
  7. 기타 shape / 타입 불일치 예외 처리
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertModel,
)
from transformers.modeling_outputs import BaseModelOutput
from .resampler.qformer import Qformer, BertConfig   # ← 새로 추가

# ---------------------------------------------------------------------------
# Resamplers
# ---------------------------------------------------------------------------
class MLPResampler(nn.Module):
    """Linear → GELU → Linear, shape 유지(B*V, N, Dl)."""
    def __init__(self, vis_dim: int, latent_dim: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(vis_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*V, N, D_v)
        return self.ff(x)


# ---------------------------------------------------------------------------
# VICReg Loss
# ---------------------------------------------------------------------------

class VicRegLoss(nn.Module):
    def __init__(self, s: float = 25.0, v: float = 25.0, c: float = 1.0):
        super().__init__()
        self.s, self.v, self.c = s, v, c

    @staticmethod
    def _off_diag(mat):
        n = mat.size(0)
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _cov(self, z):
        z = z - z.mean(0)
        N = z.size(0)
        cov = (z.T @ z) / (N - 1)
        return self._off_diag(cov).pow(2).sum() / z.size(1)

    def forward(self, x, y):  # (⋯, D)
        x = x.reshape(-1, x.size(-1))
        y = y.reshape(-1, y.size(-1))

        sim = F.mse_loss(x, y)
        std = torch.sqrt(x.var(0) + 1e-4).mean() + torch.sqrt(y.var(0) + 1e-4).mean()
        cov = self._cov(x) + self._cov(y)
        return self.s * sim + self.v * std + self.c * cov


# ---------------------------------------------------------------------------
# PanoramaVLM
# ---------------------------------------------------------------------------

class PanoramaVLM(nn.Module):
    def __init__(
        self,
        vision_name: str = "google/siglip-base-patch16-224",
        lm_name: str = "Qwen/Qwen3-0.6B",
        resampler: str = "qformer",
        num_query_tokens: int = 32,
        latent_dim: int = 768,
    ):
        super().__init__()

        # ---------------- Vision Backbone ----------------
        self.vision = AutoModel.from_pretrained(vision_name, trust_remote_code=True)
        if hasattr(self.vision, "vision_model"):
            self.vision = self.vision.vision_model  # ViT wrapper 제거
        vis_dim = self._get_vis_dim(self.vision)

        # ---------------- Resampler ----------------------
        resampler = resampler.lower()
        if resampler == "mlp":
            self.resampler = MLPResampler(vis_dim, latent_dim)
        # elif resampler == "identity":
        #     self.resampler = IdentityResampler(vis_dim, latent_dim)
        # elif resampler == "avg":
        #     self.resampler = AvgPoolResampler(vis_dim, latent_dim)
        # elif resampler == "conv":
        #     self.resampler = Conv1DResampler(vis_dim, latent_dim, num_query_tokens)
        elif resampler == "qformer":
            self.resampler = Qformer(vis_dim, latent_dim, num_query_tokens)
        else:
            raise ValueError(f"Unknown resampler: {resampler}")

        # ---------------- Text / Language ----------------
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        self.lm_proj = nn.Linear(latent_dim, self.lm.config.hidden_size)

        # ---------------- Heads & Misc -------------------
        
        self.itm_head = nn.Linear(self.lm.config.hidden_size, 2)
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.vicreg = VicRegLoss()

    # ---------------------------------------------------------------------
    # util
    # ---------------------------------------------------------------------
    @staticmethod
    def _get_vis_dim(model: nn.Module) -> int:
        cfg = model.config
        for key in (
            "hidden_size",
            "vision_hidden_size",
            "hidden_dim",
            "embed_dim",
            "projection_dim",
        ):
            if hasattr(cfg, key):
                return getattr(cfg, key)
        raise AttributeError("Cannot infer vision hidden size from config")
    def get_text_embeddings(self, input_ids) -> nn.Module:
        """텍스트 임베딩을 반환하는 헬퍼 함수"""
        if hasattr(self.lm, "get_text_features"):
            return self.lm.get_text_features(input_ids)
        else:
            return self.lm.get_input_embeddings()(input_ids)
    # ---------------------------------------------------------------------
    # forward entry
    # ---------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: str = "finetune",
        **gen_kwargs,
    ):
        # ------------------------------------------------ Vision Pass ----------------------------------------
        if pixel_values.ndim == 5:  # (B, V, C, H, W)
            B, V, C, H, W = pixel_values.shape
            flat = pixel_values.view(B * V, C, H, W)
        else:
            B, V = pixel_values.shape[0], 1
            flat = pixel_values  # (B, C, H, W)
        
        vis_hidden_state = self.vision(pixel_values=flat, return_dict=True).last_hidden_state  # (B*V, N, Dv)

        # ---------------- Stage: vision ------------------
        if stage == "vision":
            return {
                "loss": self._overlap_loss(BaseModelOutput(last_hidden_state=vis_hidden_state), B, V, 1.0)
            }

        # ---------------- Resample -----------------------
        vis_hidden_state = vis_hidden_state.view(B, V * -1, vis_hidden_state.size(-1))  # (B*V, Nq, Dv)
        img_feat_flat = self.resampler(vis_hidden_state)  # (B, V, Nq, Dl)

        # ---------------- Stage: contrastive -------------
        if stage == "contrastive":
            return self._contrastive(img_feat_flat, input_ids, attention_mask, B, V)

        # ---------------- Stage: finetune / generate -----
        if V > 1:
            Nq, Dl = img_feat_flat.shape[1:]
            img_feat = img_feat_flat.view(B, V, Nq, Dl).flatten(1, 2)  # (B, V*Nq, Dl)
        else:
            img_feat = img_feat_flat  # (B, Nq, Dl)

        if stage == "generate":
            return self._generate(img_feat, **gen_kwargs)
        else:
            return self._finetune(img_feat, input_ids, attention_mask, labels)
    # ---------------------------------------------------------------------
    # Overlap loss (VICReg)
    # ---------------------------------------------------------------------
    def _overlap_loss(self, vis_out: BaseModelOutput, B: int, V: int, weight: float):
        if V <= 1:
            return torch.zeros((), device=vis_out.last_hidden_state.device)

        patches = vis_out.last_hidden_state[:, 1:]          # (B·V, N, D)
        N, D = patches.size(1), patches.size(2)
        s = int(math.sqrt(N))
        if s * s != N:
            return torch.zeros((), device=patches.device)

        # (B, V, H, W, D)
        feat = patches.view(B, V, s, s, D)

        # ── ① 각 뷰의 오른쪽 절반
        right_half = feat[:, :, :, s // 2 :, :]             # (B, V, H, W/2, D)

        # ── ② roll-shift 하여 “다음 뷰”의 왼쪽 절반을 맞춤
        next_view = torch.roll(feat, shifts=-1, dims=1)     # v+1 (마지막→첫 뷰로 순환)
        left_half  = next_view[:, :, :, : s // 2, :]        # (B, V, H, W/2, D)

        # 이제 left_half[i, v] ↔ right_half[i, v] 가
        # 모든 인접(순환) 뷰 쌍이 됨 → 총 V 쌍
        loss = self.vicreg(right_half, left_half) * weight
        return loss
    # ---------------------------------------------------------------------
    # Stage helpers
    # ---------------------------------------------------------------------
    def _contrastive(self, img_feat_flat, input_ids, attention_mask, B, V):
        # 텍스트 CLS feature
        txt_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        txt_feat = txt_output.last_hidden_state[:, 0, :]  # (B, D)

        if V > 1:
            img_feat_loss = img_feat_flat
            txt_feat_loss = txt_feat.repeat_interleave(V, dim=0)
            bs = B * V
        else:
            img_feat_loss = img_feat_flat
            txt_feat_loss = txt_feat
            bs = B

        # --- ITC ---
        img_emb = F.normalize(img_feat_loss[:, 0, :], dim=-1)
        txt_emb = F.normalize(txt_feat_loss, dim=-1)
        sim = img_emb @ txt_emb.T  # (bs, bs)
        sim_i2t = sim / self.temp
        sim_t2i = sim_i2t.T

        target = torch.arange(bs, device=sim.device)
        loss_itc = (F.cross_entropy(sim_i2t, target) + F.cross_entropy(sim_t2i, target)) / 2

        # --- ITM (hard negative mining) ---
        with torch.no_grad():
            w_i2t = F.softmax(sim_i2t, dim=1)
            w_t2i = F.softmax(sim_t2i, dim=1)
            w_i2t.fill_diagonal_(0)
            w_t2i.fill_diagonal_(0)

        neg_txt_idx = torch.multinomial(w_t2i, 1).squeeze()

        pos_ids = input_ids.repeat_interleave(V, 0) if V > 1 else input_ids
        pos_mask = attention_mask.repeat_interleave(V, 0) if V > 1 else attention_mask
        neg_ids = pos_ids[neg_txt_idx]
        neg_mask = pos_mask[neg_txt_idx]

        itm_ids = torch.cat([pos_ids, neg_ids])
        itm_mask = torch.cat([pos_mask, neg_mask])
        itm_img = torch.cat([img_feat_loss, img_feat_loss])  # (2*bs, Nq, Dl)
        itm_img_mask = torch.ones(itm_img.shape[:-1], device=itm_img.device, dtype=torch.long)

        cross_out = self.text_encoder(
            input_ids=itm_ids,
            attention_mask=itm_mask,
            encoder_hidden_states=itm_img,
            encoder_attention_mask=itm_img_mask,
            return_dict=True,
        ).last_hidden_state[:, 0, :]

        itm_logits = self.itm_head(cross_out)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long, device=itm_logits.device), torch.zeros(bs, dtype=torch.long, device=itm_logits.device)])
        loss_itm = F.cross_entropy(itm_logits, itm_labels)

        return {"loss": loss_itc + loss_itm, "loss_itc": loss_itc, "loss_itm": loss_itm}

    # ---------------------------------------------------
    def _generate(self, img_feat, max_new_tokens: int = 32, temperature: float = 0.7, **kwargs):
        vis_tok = self.lm_proj(img_feat)

        # BOS token safe retrieval
        bos_id = self.lm.config.bos_token_id
        if bos_id is None:
            tokenizer = getattr(self.lm, "tokenizer", None)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        bos_embed = self.lm.get_input_embeddings()(torch.tensor([[bos_id]], device=vis_tok.device))

        emb = torch.cat([vis_tok, bos_embed], dim=1)
        attn_mask = torch.ones(emb.shape[:-1], device=emb.device, dtype=torch.long)

        gen_ids = self.lm.generate(inputs_embeds=emb, attention_mask=attn_mask, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)
        return {"generated_ids": gen_ids}

    # ---------------------------------------------------
    def _finetune(self, img_feat, input_ids, attention_mask, labels):
        vis_tok = self.lm_proj(img_feat)
        txt_emb = self.lm.get_input_embeddings()(input_ids)

        emb = torch.cat([vis_tok, txt_emb], dim=1)
        vis_mask = torch.ones(vis_tok.shape[:-1], device=vis_tok.device, dtype=torch.long)
        attn_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # labels: shift + padding
        pad_val = -100
        padding = torch.full((emb.size(0), vis_tok.size(1)), pad_val, device=labels.device)
        repeats = math.ceil(emb.size(0) / labels.size(0))
        exp_lbl = labels.repeat_interleave(repeats, 0)[: emb.size(0)]
        shifted_lbl = torch.cat([padding, exp_lbl[:, :-1]], dim=1)
        shifted_lbl = shifted_lbl[:, : emb.size(1)]

        out = self.lm(inputs_embeds=emb, attention_mask=attn_mask, labels=shifted_lbl, return_dict=True)
        return {"loss": out.loss, "logits": out.logits}

    

# ----------------------- Example Usage -----------------------
if __name__ == '__main__':
    # 모델과 토크나이저 로딩 시간을 줄이기 위해 작은 모델로 대체하여 테스트할 수 있습니다.
    # 예: vision_name="google/vit-base-patch16-224", lm_name="prajjwal1/bert-tiny"
    # 여기서는 제공된 설정대로 진행합니다. (모델 다운로드에 시간이 소요될 수 있습니다)
    # NOTE: `unsloth/gemma-2b-it-bnb-4bit`는 transformers > 4.41.0 필요.
    # Colab/GPU 환경에서 실행하는 것을 권장합니다.
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. 모델 초기화 ---
    # `lm_name`을 로딩이 빠른 소형 모델로 변경하여 테스트
    # `Qwen/Qwen2-0.5B` 또는 `gpt2` 등
    model = PanoramaVLM(
        vision_name="google/siglip-base-patch16-224",
        lm_name="Qwen/Qwen2-0.5B", # "unsloth/gemma-2b-it-bnb-4bit",
        resampler="qformer"
    ).to(device)
    model.eval() # 추론 모드로 설정

    # --- 2. 더미 데이터 생성 ---
    B, V, C, H, W = 2, 8, 3, 224, 224 # 배치=2, 4개 파노라마 뷰
    pano_pixels = torch.randn(B, V, C, H, W, device=device)
    single_pixels = torch.randn(B, C, H, W, device=device)
    
    # 텍스트 데이터 (BERT 토크나이저 기준)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dummy_text = ["a photo of a cat", "a dog playing in the park"]
    tokenized = tokenizer(dummy_text, return_tensors='pt', padding=True, truncation=True).to(device)
    input_ids, attention_mask = tokenized.input_ids, tokenized.attention_mask
    
    # Finetune용 레이블 (LLM 토크나이저 기준)
    from transformers import AutoTokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    finetune_text = ["The cat is sleeping on the couch.", "The dog is fetching a ball."]
    labels = llm_tokenizer(finetune_text, return_tensors='pt', padding=True).input_ids.to(device)


    print("\n--- Testing Stages ---")
    
    # --- 3. 각 Stage별 Forward Pass 테스트 ---
    with torch.no_grad():
        # Stage 1: Vision (Overlap Consistency Loss)
        try:
            vision_output = model(pixel_values=pano_pixels, stage="vision")
            vision_loss = vision_output['loss']
            print(f"✅ [Vision Stage] Overlap Loss: {vision_loss.item():.4f}")
        except Exception as e:
            print(f"❌ [Vision Stage] Error: {e}")

        # Stage 2: Contrastive (ITC + ITM Loss)
        # NOTE: contrastive stage는 identity/avg resampler를 사용하는 것이 일반적입니다.
        # 여기서는 qformer 출력을 (B,D)로 변환하여 테스트합니다.
        try:
            # contrastive는 4D (B, C, H, W) 이미지 입력을 가정
            contrastive_output = model(pixel_values=single_pixels, input_ids=input_ids, attention_mask=attention_mask, stage="contrastive")
            contrastive_loss = contrastive_output['loss']
            print(f"✅ [Contrastive Stage] Total Loss: {contrastive_loss.item():.4f}")
        except Exception as e:
            print(f"❌ [Contrastive Stage] Error: {e}")

        # Stage 3: Finetune (Language Modeling Loss)
        try:
            finetune_output = model(pixel_values=single_pixels, input_ids=labels, attention_mask=torch.ones_like(labels), labels=labels, stage="finetune")
            finetune_loss = finetune_output['loss']
            print(f"✅ [Finetune Stage] LM Loss: {finetune_loss.item():.4f}")
        except Exception as e:
            print(f"❌ [Finetune Stage] Error: {e}")

        # Stage 4: Generate (Text Generation)
        try:
            # 생성을 위해 LLM 토크나이저를 모델에 연결
            model.lm.tokenizer = llm_tokenizer 
            generate_output = model(pixel_values=single_pixels[0:1], stage="generate", max_new_tokens=20)
            generated_ids = generate_output['generated_ids']
            generated_text = llm_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"✅ [Generate Stage] Generated Text: '{generated_text.strip()}'")
        except Exception as e:
            print(f"❌ [Generate Stage] Error: {e}")