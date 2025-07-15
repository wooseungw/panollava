# coding: utf-8
"""
PanoramaVLM – 2-Stage (VICReg ➜ AR) 평가용 레퍼런스 구현
────────────────────────────────────────────────────────
● **Stage "vicreg"** : 파노라마 인접 뷰 간 중첩 일관성 정렬(VICReg loss)
● **Stage "train"**  : Vision → Resampler → LLM 단일 autoregressive CE loss
● **Stage "generate"** : 프롬프트 없는 zero-shot 캡션/응답 생성

본 파일은 **데이터셋-호환 & 그래디언트 검증용 테스트 코드**를 포함합니다.
(데이터셋이 dict 형태로
  `{pixel_values, input_ids, attention_mask, labels}`
  를 반환한다고 가정)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

def _infer_hw(num_patches: int) -> tuple[int, int]:
    """
    주어진 패치 토큰 개수(`num_patches`)로부터 (H, W) 그리드 크기를 추정합니다.
    1) 완전제곱 → 정사각형  
    2) 아니면 √N 이하에서 가장 큰 약수를 찾아 (h, w = N//h) 반환  
       (예: 256 → 16×16, 140 → 10×14)
    """
    h = int(math.sqrt(num_patches))
    if h * h == num_patches:
        return h, h

    for h in range(h, 0, -1):
        if num_patches % h == 0:
            return h, num_patches // h
    raise ValueError(f"grid 추정 실패: N={num_patches}")
# ---------------------------------------------------------------------------
# ‣ Resampler
# ---------------------------------------------------------------------------
class MLPResampler(nn.Module):
    """(B·V, N, Dv) → (B·V, N_q, Dl)"""

    def __init__(self, vis_dim: int, latent_dim: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(vis_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)

# ---------------------------------------------------------------------------
# ‣ VICReg Loss
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

    def forward(self, x, y):
        x = x.reshape(-1, x.size(-1))
        y = y.reshape(-1, y.size(-1))
        sim = F.mse_loss(x, y)
        std = torch.sqrt(x.var(0) + 1e-4).mean() + torch.sqrt(y.var(0) + 1e-4).mean()
        cov = self._cov(x) + self._cov(y)
        return self.s * sim + self.v * std + self.c * cov

# ---------------------------------------------------------------------------
# ‣ PanoramaVLM (VICReg + AR)
# ---------------------------------------------------------------------------
class PanoramaVLM(nn.Module):
    def __init__(
        self,
        vision_name: str = "google/siglip-base-patch16-224",
        lm_name: str = "Qwen/Qwen2-0.5B",
        latent_dim: int = 768,
        num_query_tokens: int = 16,
        vicreg_weight: float = 1.0,
    ):
        super().__init__()

        # Vision ---------------------------------------------------------
        self.vision = AutoModel.from_pretrained(vision_name, trust_remote_code=True)
        if hasattr(self.vision, "vision_model"):
            self.vision = self.vision.vision_model
        vis_dim = self._vis_dim(self.vision)

        # Resampler ------------------------------------------------------
        self.num_query_tokens = num_query_tokens
        self.resampler = MLPResampler(vis_dim, latent_dim)

        # Language Model -------------------------------------------------
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        self.lm_proj = nn.Linear(latent_dim, self.lm.config.hidden_size)

        # Tokenizer ------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # VICReg ---------------------------------------------------------
        self.vicreg = VicRegLoss()
        self.vicreg_weight = vicreg_weight

    # ---------------- util ---------------------------------------------
    @staticmethod
    def _vis_dim(model: nn.Module) -> int:
        for k in ["hidden_size", "vision_hidden_size", "hidden_dim", "embed_dim", "projection_dim"]:
            if hasattr(model.config, k):
                return getattr(model.config, k)
        raise AttributeError("Cannot infer vision hidden size")

    # ---------------- main forward -------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: str = "train",  # "vicreg" | "train" | "generate"
        max_new_tokens: int = 32,
        temperature: float = 0.7,
    ):
        if pixel_values.ndim != 5:
            raise ValueError("pixel_values should be (B,V,C,H,W)")
        B, V, C, H, W = pixel_values.shape
        flat = pixel_values.view(B * V, C, H, W)
        print(f"Input pixel values shape: {flat.shape}")
        # Vision --------------------------------------------------------
        vis_hs = self.vision(pixel_values=flat, return_dict=True).last_hidden_state  # (B·V,N,Dv)
        print(f"Vision output shape: {vis_hs.shape}")
        _, N, Dv = vis_hs.shape
        # Stage ❶ VICReg --------------------------------------------------
        if stage == "vision":
            loss_dict = {"loss": self._vicreg_overlap(vis_hs, B, V, overlap_ratio=0.5)}
            return loss_dict

        # Resample ------------------------------------------------------
        # (B,V*N_q,Dl)
        
        img_feat = vis_hs.view(B, V, N, Dv)                   
        img_feat = img_feat.reshape(B, V * N, Dv)  # (B·V,N,Dv)
        print(f"Flattened image features shape: {img_feat.shape}")
        img_feat = self.resampler(img_feat)                                           
        print(f"Resampled image features shape: {img_feat.shape}")
        if stage == "generate":
            return self._generate(img_feat, max_new_tokens, temperature)
        elif stage == "finetune":
            return self._ar_loss(img_feat, input_ids, attention_mask, labels)
        else:
            raise ValueError("stage must be 'vision', 'finetune', or 'generate'")

    # ---------------- VICReg helper ------------------------------------
    def _vicreg_overlap(
        self,
        vis_out: torch.Tensor,   # (B·V, S, D)
        B: int,
        V: int,
        overlap_ratio: float = 0.5,   # 파노라마 뷰 간 중첩 비율(열 기준)
    ):
        """
        ▸ vis_out : Vision encoder 출력 (CLS 포함 여부 무관)  
        ▸ overlap_ratio : 0.0 < r ≤ 0.5 권장  
            예) 0.25 → 오른쪽 25% ↔ 다음 뷰 왼쪽 25%
        """
        # 뷰 1개면 손실 없음 --------------------------------------------------
        if V <= 1:
            return torch.zeros((), device=vis_out.device)

        # 1) CLS 토큰 유무 판단 ---------------------------------------------
        has_cls = (vis_out.shape[1] % 2 == 1)  # ViT 계열이면 S=패치+1(CLS) → 홀수
        patches = vis_out[:, 1:] if has_cls else vis_out          # (B·V, N, D)
        N = patches.size(1)

        # 2) (H, W) 그리드 복원 ---------------------------------------------
        H, W = _infer_hw(N)                                       # 예) 14×14, 16×16 …
        feat = patches.view(B, V, H, W, -1)                       # (B, V, H, W, D)

        # 3) 오버랩 열(col) 개수 k 계산 --------------------------------------
        k = max(1, int(W * overlap_ratio))                        # 최소 1 col 보장

        # 4) 현재 뷰 오른쪽 k열 ↔ 다음 뷰 왼쪽 k열 ---------------------------
        right_cols = feat[..., -k:, :]                            # (B,V,H,k,D)
        left_cols  = torch.roll(feat, shifts=-1, dims=1)[..., :k, :]

        # 5) VICReg 손실 -----------------------------------------------------
        return self.vicreg(right_cols, left_cols)

    # ---------------- AR loss ------------------------------------------
    def _ar_loss(self, img_feat, input_ids, attention_mask, labels):
        vis_tok = self.lm_proj(img_feat)
        print(f"Image features projected to LM input shape: {vis_tok.shape}")
        print(f"Input IDs shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}")
        txt_emb = self.lm.get_input_embeddings()(input_ids)
        print(f"Text embeddings shape: {txt_emb.shape}")
        emb_seq = torch.cat([vis_tok, txt_emb], dim=1)
        attn_seq = torch.cat([
            torch.ones(vis_tok.shape[:2], dtype=torch.long, device=vis_tok.device),
            attention_mask,
        ], dim=1)
        pad = emb_seq.new_full((emb_seq.size(0), vis_tok.size(1)), -100, dtype=torch.long)
        lbl = torch.cat([pad, labels], dim=1)
        lbl = torch.roll(lbl, shifts=-1, dims=1)
        lbl[:, -1] = -100
        out = self.lm(inputs_embeds=emb_seq, attention_mask=attn_seq, labels=lbl)
        return {"loss": out.loss, "logits": out.logits}

    # ---------------------------------------------------------------------
    # generation
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def _generate(self, img_feat, max_new_tokens: int, temperature: float):
        vis_tok = self.lm_proj(img_feat)                       # (B,T_v,H)
        bos_id = self.lm.config.bos_token_id or self.tokenizer.eos_token_id
        bos_emb = self.lm.get_input_embeddings()(torch.tensor([[bos_id]], device=vis_tok.device))
        emb_seq = torch.cat([vis_tok, bos_emb.expand(vis_tok.size(0), -1, -1)], dim=1)
        attn_seq = torch.ones(emb_seq.shape[:2], dtype=torch.long, device=vis_tok.device)

        ids = self.lm.generate(
            inputs_embeds=emb_seq,
            attention_mask=attn_seq,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {"generated_ids": ids, "text": self.tokenizer.batch_decode(ids, skip_special_tokens=True)}

