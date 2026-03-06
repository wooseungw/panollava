# PanoAdapt — 실험 계획 및 결과

> 최종 업데이트: 2026-03-03
> 목적: 상용 VLM에 파노라마 적응 기법(PanoRoPE + Overlap Loss) 적용 효과 검증

---

## 1. 연구 개요

**PanoAdapt**: 기존 상용 VLM을 재학습 없이 파노라마 이미지에 적응시키는 경량 방법론.

- **PanoRoPE**: 파노라마의 연속적 yaw 구조를 반영한 1D/3D positional encoding
- **Overlap Loss**: 인접 뷰 간 겹치는 영역의 feature consistency를 강제하는 self-supervised loss
- **LoRA**: 전체 파라미터의 ~0.6%만 학습 (r=32, alpha=64)

**PanoAdapt 3계층 구조:**

| Layer | 구성 요소 | 설명 |
|:---:|---|---|
| L1 | **Multi-view 입력 (AnyRes-E2P)** | 파노라마 → 1 global + 8 tiles (45° pitch=0°) |
| L2 | **PanoRoPE (spatial PE)** | 인접 타일 간 연속 position ID 부여 |
| L3 | **Overlap SSL Loss** | 겹치는 영역 feature consistency (DenseCL / VICReg) |

**대상 모델:**

| Model | Params | Vision Encoder |
|-------|:------:|---------------|
| InternVL3.5-2B | 2B | InternViT-300M (별도 q/k/v proj) |
| Qwen2.5-VL-3B | 3B | Qwen2-VL ViT (fused qkv) |
| Gemma3-4B | 4B | SigLIP2 SO400M (별도 q/k/v proj) |

---

## 2. 공통 실험 설정

| 항목 | 설정 |
|------|------|
| 입력 전략 | anyres_e2p 9-view (1 global + 8 tiles, pitch=0°, stride=45°) |
| Physical overlap | 50% |
| LoRA r / alpha | 32 / 64 |
| Epochs | 1 |
| Precision | fp16 (Gemma3: bf16) |
| GPU | 1× RTX 3090 (24GB) |
| Dataset | QuIC-360 |
| Test set | 5,349 samples |
| Decoding | Greedy, max_new_tokens=128 |
| Eval metrics | BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE |

> **입력 전략 선정 근거**: Qwen2.5-VL-3B 기준 anyres_e2p가 CIDEr=0.3389로 4개 전략 중 최고 (부록 A 참조).

---

## 3. Native Baseline (PanoAdapt 없음)

각 VLM의 native image processor + anyres_e2p 9-view 입력, LoRA 1 epoch.

| Model | BLEU-4 ↑ | METEOR ↑ | ROUGE-L ↑ | CIDEr ↑ | SPICE ↑ | LLM-Judge ↑ | 상태 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **InternVL3.5-2B** | **0.0443** | 0.1111 | **0.2462** | **0.3405** | **0.1661** | **69.2** | ✅ |
| Gemma3-4B | 0.0420 | 0.1081 | 0.2453 | 0.3363 | 0.1636 | 68.6 | ✅ |
| Qwen2.5-VL-3B | 0.0434 | **0.1125** | 0.2427 | 0.3306 | 0.1548 | 67.9 | ✅ |
| InternVL3.5-1B | 0.0389 | 0.1065 | 0.2394 | 0.3171 | 0.1606 | — | ✅ |
| BLIP2-OPT-2.7B | 0.0051 | 0.0448 | 0.1230 | 0.0715 | 0.0848 | — | ✅ (256² only) |

---

## 4. PanoAdapt 실험

### 4.1 Overlap Loss 종류

| Loss | 방식 | 핵심 |
|------|------|------|
| **DenseCL** | Symmetric InfoNCE @ patch level | 인접 뷰 overlap strip의 대응 패치를 N×N similarity matrix로 학습 |
| **VICReg-pairwise** | MSE + variance + covariance @ pair level | 각 인접 뷰 쌍 내부에서 통계 계산, negatives 불필요 |

### 4.2 결과 — DenseCL (overlap 50%)

| Model | Native CIDEr | PanoAdapt CIDEr | Δ | BLEU-4 | METEOR | ROUGE-L | SPICE | LLM-Judge ↑ |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| InternVL3.5-2B | 0.3405 | **0.3603** | **+5.8%** ✅ | 0.0457 | 0.1137 | 0.2492 | 0.1720 | 70.6 |
| Qwen2.5-VL-3B | 0.3306 | 0.3423 | +3.5% | 0.0426 | 0.1135 | 0.2448 | 0.1613 | 68.0 |
| Gemma3-4B | 0.3363 | 0.3362 | -0.03% | 0.0437 | 0.1162 | 0.2415 | 0.1685 | 69.3 |

### 4.3 결과 — VICReg-pairwise

| Model | Overlap | Native CIDEr | PanoAdapt CIDEr | Δ | BLEU-4 | METEOR | ROUGE-L | SPICE | LLM-Judge ↑ |
|-------|:-------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| InternVL3.5-2B | 25% | 0.3405 | 0.3594 | +5.5% ✅ | 0.0457 | 0.1136 | 0.2489 | 0.1713 | — |
| InternVL3.5-2B | 50% | 0.3405 | **0.3609** | **+6.0%** ✅ | 0.0456 | 0.1137 | 0.2491 | 0.1718 | 70.2 |
| Qwen2.5-VL-3B | 50% | 0.3306 | 0.3425 | +3.6% | 0.0426 | 0.1137 | 0.2446 | 0.1627 | 68.3 |
| Gemma3-4B | 50% | 0.3363 | 0.3357 | -0.2% | 0.0439 | 0.1181 | 0.2412 | 0.1698 | **71.8** |

### 4.4 종합 비교 (CIDEr 기준)

| Model | Native | DenseCL 50% | VICReg-pw 25% | VICReg-pw 50% |
|-------|:---:|:---:|:---:|:---:|
| InternVL3.5-2B | 0.3405 | 0.3603 (+5.8%) | 0.3594 (+5.5%) | **0.3609 (+6.0%)** ✅ |
| Qwen2.5-VL-3B | 0.3306 | 0.3423 (+3.5%) | — | 0.3425 (+3.6%) |
| Gemma3-4B | 0.3363 | 0.3362 (-0.03%) | — | 0.3357 (-0.2%) |

---

## 5. 핵심 발견

### F1. InternVL에서 PanoAdapt 효과 명확 (+5.5~6.0%)
모든 loss/overlap 조합에서 일관되게 +5.5~6.0% 개선. DenseCL과 VICReg-pairwise의 차이는 0.5%p 이내로 loss 종류보다 overlap 구조 자체가 핵심일 가능성.

| Loss | Overlap | CIDEr | Δ |
|------|:-------:|:-----:|:--:|
| DenseCL | 50% | 0.3603 | +5.8% |
| VICReg-pw | 25% | 0.3594 | +5.5% |
| VICReg-pw | 50% | 0.3609 | +6.0% |

### F2. Qwen의 overlap loss 효과는 InternVL 대비 제한적 (+3.5~3.6%)
Qwen2.5-VL의 vision encoder attention은 **fused `attn.qkv = nn.Linear(dim, dim*3)`** 구조.
실험 당시 `target_modules`에 **`attn.qkv`, `attn.proj`를 포함**하여 vision encoder에도 LoRA가 정상 삽입됨 (adapter_config.json 확인).

| 모델 | Vision Encoder Attn | LoRA 매칭 | Overlap Loss 효과 |
|------|-------------------|:---:|:---:|
| InternVL3.5 | 별도 `q_proj`/`k_proj`/`v_proj` | ✅ | ✅ +5.5~6.0% |
| Gemma3 (SigLIP2) | 별도 `q_proj`/`k_proj`/`v_proj` | ✅ | ✅ (미미) |
| Qwen2.5-VL | fused `attn.qkv` + `attn.proj` | ✅ (수정됨) | △ +3.5~3.6% |

→ **Qwen +3.5%는 vision LoRA가 포함된 Full PanoAdapt 효과.** InternVL(+6%)보다 낮은 이유는 별도 분석 필요.
→ **Gemma3 -0.03%는 별도 분석 필요** (LoRA 매칭 정상, overlap loss 효과 없음).

### F3. Gemma3는 반드시 bf16
Gemma3-4B-IT는 bfloat16으로 pretrain됨. fp16 학습 시 9-view × 256 tokens = 2304 image tokens attention에서 overflow → `loss=0.0, grad_norm=nan` 전체 실패. **bf16 필수.**

### F4. VICReg-pw 50% > DenseCL 50% > VICReg-pw 25% (InternVL 기준)
50% overlap이 25%보다 근소하게 우세. loss 종류별 차이보다 overlap 비율 차이가 더 의미있을 수 있음.

### F5. DenseCL과 VICReg-pw는 Qwen에서도 동일 효과
Qwen DenseCL=0.3423 (+3.5%), VICReg-pw 50%=0.3425 (+3.6%). 차이 0.0002로 loss 종류보다 모델 구조 차이가 지배적.

### F6. 3계층 기여도 분해 (E1 Ablation — **완료**)

**E1-PE 실험 완료. PanoRoPE 단독 기여는 사실상 없음(-0.0003, 측정 오차 수준).**

| 단계 | 추가 구성요소 | CIDEr | 순증분 | 의미 |
|---|---|:---:|:---:|---|
| Native (L0 baseline) | — | 0.3405 | — | 기준선 |
| + AnyRes-E2P (L1) | 멀티뷰 분해만 | 0.3598 | **+0.0193 (+5.7%)** | 뷰 전략 단독 효과 (지배적) |
| + PanoRoPE (L2) | PE 추가 | 0.3595 | **-0.0003 (~0%)** | PE 단독 효과 — **사실상 없음** |
| + DenseCL 50% (L3) | SSL Loss 추가 | 0.3603 | **+0.0008 (+0.2%)** | Loss 단독 효과 — 소폭 개선 |

- AnyRes-E2P 단독 기여: **+0.0193 (+5.7%)** — 성능 향상의 ~99%
- PanoRoPE 단독 기여: **-0.0003** — 측정 오차 수준, 실질적으로 없음
- SSL Loss 단독 기여: **+0.0008 (+0.2%)** — 소폭이지만 일관된 개선



→ 논문 서사: "성능 향상의 ~99%는 멀티뷰 분해(AnyRes-E2P, L1)에서 기인. PanoRoPE(L2)는 단독으로는 중립, SSL Loss(L3)는 +0.2%p의 일관된 미세 개선을 제공."

### F7. Ortho-LoRA (gradient surgery)는 joint 학습보다 열등

**Ortho-LoRA CIDEr=0.3591** — 표준 PanoAdapt DenseCL(0.3603)보다 낮음. SFT-only E1-B(0.3598)보다도 낮음.

| 학습 전략 | CIDEr | vs DenseCL |
|------|:-----:|:---------:|
| PanoAdapt DenseCL 50% (직접 합산) | 0.3603 | baseline |
| Ortho-LoRA DenseCL (gradient surgery) | 0.3591 | **-0.0012 (-1.2pp)** |
| Ortho-LoRA VICReg-pw (gradient surgery) | 🔄 eval 예정 | — |

→ gradient 충돌 회피가 오히려 성능을 저해. 가능한 원인:
- 두 backward pass의 `retain_graph=True`가 최적화 안정성 저해
- 충돌을 제거하면서 beneficial interference도 제거
- SFT+SSL gradient는 방향 충돌보다 시너지 위주임 시사 (joint 학습이 이미 적절한 해결책)
### F8. TIES merge는 joint 학습보다 미세하게 우세

**TIES (SFT+DenseCL) CIDEr=0.3607** — DenseCL joint(0.3603)보다 +0.0004(+0.4pp).
**TIES (SFT+VICReg-pw) CIDEr=🔄** — eval 진행 중.
SFT-only adapter + SSL adapter를 독립 학습 후 TIES 병합하면 joint 학습보다 손해 없이 두 능력 결합 가능.

---

## 6. 실행 현황

### ✅ 완료

| 실험 | CIDEr | SPICE |
|------|:-----:|:-----:|
| Native InternVL3.5-2B | 0.3405 | 0.1661 |
| Native Qwen2.5-VL-3B | 0.3306 | 0.1548 |
| Native Gemma3-4B | 0.3363 | 0.1636 |
| Native InternVL3.5-1B | 0.3171 | 0.1606 |
| Native BLIP2-OPT-2.7B (256²) | 0.0715 | 0.0848 |
| PanoAdapt DenseCL — InternVL3.5-2B | 0.3603 | 0.1720 |
| PanoAdapt DenseCL — Qwen2.5-VL-3B | 0.3423 | 0.1613 |
| PanoAdapt DenseCL — Gemma3-4B | 0.3362 | 0.1685 |
| PanoAdapt VICReg-pw 25% — InternVL3.5-2B | 0.3594 | 0.1713 |
| PanoAdapt VICReg-pw 50% — InternVL3.5-2B | **0.3609** | 0.1718 |
| PanoAdapt VICReg-pw 50% — Qwen2.5-VL-3B | 0.3425 | 0.1627 |
| PanoAdapt VICReg-pw 50% — Gemma3-4B | 0.3357 | 0.1698 |
| **E1-A** Cubemap no loss (no PE) — InternVL3.5-2B | 0.3579 | — |
| **E1-B** AnyRes-E2P no loss (no PE) — InternVL3.5-2B | 0.3598 | — |
| **E1-PE** AnyRes-E2P + PanoRoPE (no loss) — InternVL3.5-2B | **0.3595** | — |
| **E1-C** AnyRes-E2P + DenseCL 25% — InternVL3.5-2B | 0.3600 | — |
| **TIES merge** SFT+DenseCL — InternVL3.5-2B | **0.3607** | — |
| **Ortho-LoRA DenseCL** gradient surgery — InternVL3.5-2B | 0.3591 | 0.1714 |
| **TIES merge** SFT+VICReg-pw — InternVL3.5-2B | **0.3612** | 0.1716 |



### 🔄 진행중

| 실험 | GPU | 진행률 | 예상 완료 |
|------|:---:|:------:|----------|
| ~~VICReg-pw 50% Gemma3-4B~~ | GPU 1 | ✅ 완료 (CIDEr=0.3357) → LLM-Judge 🔄 | — |
| ~~TIES merge SFT+VICReg-pw eval~~ | — | ✅ 완료 (CIDEr=0.3612) | — |

### LLM-Judge 결과 — 12/12 ✅

> **점수 체계: 0–100점** (각 차원 원점수 1–5 → weighted score × 20)
> - 가중 합산: Spatial Coherence×0.30 + Query Relevance×0.25 + Factual Accuracy×0.20 + Completeness×0.15 + Fluency×0.10
> - 샘플링: dedup(image_path, query) 후 300개 (seed=42, 재현 가능)
> - Judge 모델: gpt-5.2, 이미지 포함, OpenAI Batch API

| 모델 | 방법 | **Judge ↑** | Overall | Spatial | Query | Factual | Complete | Fluency |
|------|------|:-----------:|:-------:|:-------:|:-----:|:-------:|:--------:|:-------:|
| InternVL3.5-2B | Resize (256²) | 67.6 | 59.7 | 58.5 | 81.7 | 63.7 | 47.4 | 97.9 |
| InternVL3.5-2B | Native        | 69.2 | 61.2 | 60.5 | 82.6 | 66.3 | 48.7 | 98.4 |
| InternVL3.5-2B | +DenseCL      | **70.6** | 63.1 | 61.7 | 84.5 | 67.9 | 50.2 | 98.1 |
| InternVL3.5-2B | +VICReg-pw    | 70.2 | 62.1 | 60.7 | 84.5 | 67.5 | 50.1 | 98.3 |
| Qwen2.5-VL-3B  | Resize (256²) | 65.3 | 56.5 | 58.7 | 78.5 | 57.7 | 44.5 | 98.7 |
| Qwen2.5-VL-3B  | Native        | 67.9 | 59.4 | 60.1 | 80.6 | 63.9 | 47.1 | 98.4 |
| Qwen2.5-VL-3B  | +DenseCL      | 68.0 | 59.3 | 60.3 | 80.9 | 63.8 | 47.5 | 98.1 |
| Qwen2.5-VL-3B  | +VICReg-pw    | **68.3** | 60.5 | 60.5 | 82.3 | 63.1 | 47.7 | 98.4 |
| Gemma3-4B      | Resize (256²) | 68.8 | 60.8 | 59.5 | 82.0 | 66.6 | 48.2 | 98.7 |
| Gemma3-4B      | Native        | 68.6 | 60.8 | 59.5 | 81.8 | 66.4 | 47.5 | 98.7 |
| Gemma3-4B      | +DenseCL      | 69.3 | 61.5 | 60.9 | 81.3 | 66.5 | 50.1 | 98.7 |
| Gemma3-4B      | +VICReg-pw    | **71.8** | 62.9 | 61.0 | 85.9 | 72.0 | 51.9 | 97.9 |

> 굵은 숫자 = 해당 백본 내 최고점. Judge ↑ = weighted score (100점 만점, 높을수록 좋음)

### ⏳ 대기 중 (실험 설계 완료, 아직 미시작)

| 실험 | 목적 | 예상 시작 |
|------|------|----------|
| ~~Ortho-LoRA + VICReg-pw~~ | VICReg-pw에서 gradient surgery | 학습 완료, **eval 진행중** (GPU 1) |
| ~~E2: Qwen B1-fix (vision LoRA `attn.qkv` 추가)~~ | ~~Qwen overlap loss 활성화~~ | **이미 적용됨** — trackB 실험에 `attn.qkv`, `attn.proj` 포함 확인 |

### ❌ DROPPED

| 실험 | 사유 |
|------|------|
| Native InternVL2.5-4B | PEFT `prepare_inputs_for_generation` crash, 미해결 |
| Native InternVL2.5-2B | 동일 PEFT crash |

---

## 7. E1. 완전 Ablation (InternVL3.5-2B)

**목적**: PanoAdapt 3계층(뷰 전략 + PanoRoPE + SSL Loss) 각각의 기여 분리.

| # | 입력 방식 | PanoRoPE | Loss | Overlap | 상태 | BLEU-4 ↑ | METEOR ↑ | ROUGE-L ↑ | CIDEr ↑ | SPICE ↑ |
|:---:|----------|:---:|------|:-------:|:----:|:--------:|:--------:|:---------:|:-------:|:-------:|
| 1 | Resize 256² | ✗ | None | — | ✅ | 0.0403 | 0.1096 | 0.2402 | 0.3054 | 0.1566 |
| 2 | Native | ✗ | None | — | ✅ | 0.0443 | 0.1111 | 0.2462 | 0.3405 | 0.1661 |
| 3 | Cubemap | ✗ | None | — | ✅ | 0.0452 | 0.1136 | 0.2494 | 0.3579 | 0.1706 |
| 4 | AnyRes-E2P | ✗ | None | — | ✅ | 0.0456 | 0.1136 | 0.2490 | 0.3598 | 0.1718 |
| **4-PE** | **AnyRes-E2P** | **✅** | **None** | **—** | **✅** | **0.0455** | **0.1137** | **0.2490** | **0.3595** | **0.1714** |
| 5 | AnyRes-E2P | ✅ | DenseCL | 25% | ✅ | 0.0455 | 0.1135 | 0.2488 | 0.3600 | 0.1713 |
| 6 | AnyRes-E2P | ✅ | DenseCL | 50% | ✅ | 0.0457 | 0.1137 | 0.2492 | 0.3603 | 0.1720 |
| 7 | AnyRes-E2P | ✅ | VICReg-pw | 25% | ✅ | 0.0457 | 0.1136 | 0.2489 | 0.3594 | 0.1713 |
| 8 | AnyRes-E2P | ✅ | VICReg-pw | 50% | ✅ | 0.0456 | 0.1137 | 0.2491 | **0.3609** | **0.1718** |

> ⚠️ **주의**: 기존 E1 #5~8은 PanoRoPE(PE)가 **이미 포함**된 Full PanoAdapt 설정임.

**3계층 기여 분해 (CIDEr 기준):**
```
L1 (럄 전략  Resize→AnyRes-E2P):  0.3598 - 0.3405 = +0.0193 (+5.7%p)
L2 (+ PanoRoPE):                      0.3595 - 0.3598 = -0.0003 (없음)
L3 (+ VICReg-pw 50%):                 0.3609 - 0.3595 = +0.0014 (+0.4%p)
```
> → 주 기여는 **럄 전략(L1)** 이며, PanoRoPE/Loss는 수치적으로는 미미함. LLM-Judge로 잡히지 않는 콩텐츠 품질 차이 반영 가능성 있음.

---

## 8. E3. 새 학습 전략 실험

**목적**: E1에서 SSL Loss 단독 기여가 미미한 이유(~+0.1%p)를 극복하기 위한 대안 학습 전략 탐색.

### 8.1 E3-A: LoRA Merging (TIES)

**가설**: SFT(파노라마 적응)와 SSL(시각적 일관성)을 독립 학습 후 병합 → gradient 간섭 없이 두 능력 결합.

**방법**: PEFT `add_weighted_adapter(combination_type="ties", density=0.5)`.

| 실험 | Adapter A | Adapter B | Method | Density | BLEU-4 ↑ | METEOR ↑ | ROUGE-L ↑ | CIDEr ↑ | SPICE ↑ | 상태 |
|------|-----------|-----------|:------:|:-------:|:--------:|:--------:|:---------:|:-------:|:-------:|:----:|
| TIES merge SFT+DenseCL | E1-B (0.3598) | DenseCL 50% (0.3603) | TIES | 0.5 | 0.0463 | 0.1135 | 0.2494 | **0.3607** | 0.1712 | ✅ 완료 |
| TIES merge SFT+VICReg-pw | E1-B (0.3598) | VICReg-pw 50% (0.3609) | TIES | 0.5 | 0.0462 | 0.1135 | 0.2492 | **0.3612** | 0.1716 | ✅ 완료 |
**구현**: `scripts/merge_lora.py`

### 8.2 E3-B: Ortho-LoRA (Gradient Surgery)

**가설**: SFT gradient와 SSL gradient의 방향 충돌 시 직교 투영으로 해소 → 두 signal이 상호 강화.

**방법**: arXiv:2601.09684 Algorithm 1. LoRA A/B 행렬별 독립적으로 직교 투영:
```
if (g_sft · g_ssl) < 0:   # 방향 충돌
    g_ssl ← g_ssl - (g_sft · g_ssl / ||g_sft||²) · g_sft
g_combined = g_sft + g_ssl
```

| 실험 | Config | BLEU-4 ↑ | METEOR ↑ | ROUGE-L ↑ | CIDEr ↑ | SPICE ↑ | 상태 |
|------|--------|:--------:|:--------:|:---------:|:-------:|:-------:|:----:|
| Ortho-LoRA DenseCL — InternVL3.5-2B | ortholora_internvl35_2b.yaml | 0.0437 | 0.1107 | 0.2494 | 0.3591 | **0.1714** | ✅ |
| Ortho-LoRA VICReg-pw — InternVL3.5-2B | ortholora_vicreg_internvl35_2b.yaml | — | — | — | — | — | ⏳ eval 진행중 (GPU 1) |
**구현**: `src/cora/baseline/finetune.py::_OrthoLoRATrainer`
**Config 플래그**: `panoadapt.ortho_lora: true`

---

## 9. 버그 이력

| 날짜 | 버그 | 수정 |
|------|------|------|
| 2026-02-24 | `_unwrap_to_rope_model()` 무한 루프 | `base_model` 자기참조 사이클 체크 추가 |
| 2026-02-24 | `max_length: 1024` 부족 (9 views × 256 = 2304 tokens) | `max_length: 3072` 수정 |
| 2026-02-25 | Qwen DenseCL `Expected features with 2 or 4 dims, got 3` | `DenseCLLoss.forward` ndim==3 처리 추가 |
| 2026-02-26 | Gemma3 DenseCL `loss=0.0, grad_norm=nan` 전체 실패 | `dtype: float16 → bfloat16`, `mixed_precision: fp16 → bf16` |
| 2026-02-26 | Gemma3 `multi_modal_projector` output 3D `[N, 256, 2560]` | `_compute_densecl` else-branch ndim==3 직접 인덱싱 |
| 2026-02-27 | InternVL PEFT `prepare_inputs_for_generation` 없음 | monkey-patching으로 stub 추가 |
| 2026-03-01 | `merge_lora.py` adapter name에 `.` 포함 불가 | `d{density}` → `d{str(density).replace('.','_')}` |
| 2026-03-01 | `merge_lora.py` PEFT `save_pretrained`이 adapter_name 서브디렉토리에 저장 | tmp dir → shutil.move로 flat 구조로 이동 |
| 2026-03-02 | `merge-lora` tmux 세션이 완료 후에도 살아있어 `pe-ablation` 무한 대기 | `tmux kill-session -t merge-lora` 로 수동 해제 |
| 2026-03-02 | **Gemma3 eval Meteor JAR 데드락** (22시간 stuck) | `metrics.py`에서 Meteor 제거, `finetune.py`에서 predictions 먼저 저장 후 metrics 계산 순서 변경 |
---

## 부록 A. 입력 전략 비교 (Qwen2.5-VL-3B 기준)

> LoRA 1 epoch, QuIC-360 test 5,349 samples

| 전략 | Views | CIDEr ↑ | BLEU-4 ↑ | METEOR ↑ | ROUGE-L ↑ | SPICE ↑ |
|------|-------|:---:|:---:|:---:|:---:|:---:|
| resize (256²) | 1 | 0.2809 | 0.0382 | 0.1113 | 0.2334 | 0.1435 |
| native (dynamic) | dynamic | 0.3285 | 0.0431 | 0.1124 | 0.2421 | 0.1554 |
| cubemap | 5 (4+global) | 0.3303 | 0.0424 | 0.1119 | 0.2424 | 0.1575 |
| **anyres_e2p** | **9 (8+global)** | **0.3389** | 0.0420 | **0.1138** | **0.2441** | **0.1613** |

→ anyres_e2p 채택. pinhole (CIDEr=0.3384)은 anyres_e2p와 차이 0.0005로 제외.

---

## 부록 B. 파일 위치

```
configs/baseline/
├── panoadapt_internvl35_2b.yaml                            # DenseCL InternVL ✅
├── panoadapt_qwen25_3b.yaml                                # DenseCL Qwen ✅
├── panoadapt_gemma3_4b.yaml                                # DenseCL Gemma3 (bf16) ✅
├── panoadapt_vicreg_pairwise_internvl35_2b_25overlap.yaml  # VICReg-pw 25% InternVL ✅
├── panoadapt_vicreg_pairwise_internvl35_2b.yaml            # VICReg-pw 50% InternVL ✅
├── panoadapt_vicreg_pairwise_qwen25_3b.yaml                # VICReg-pw 50% Qwen ✅
├── panoadapt_vicreg_pairwise_gemma3_4b.yaml                # VICReg-pw 50% Gemma3 ❌ 학습 실패
├── ablation_internvl35_2b_cubemap_noloss.yaml              # E1-A ✅
├── ablation_internvl35_2b_anyrese2p_noloss.yaml            # E1-B ✅
├── ablation_internvl35_2b_densecl_25overlap.yaml           # E1-C ✅
├── ablation_internvl35_2b_anyrese2p_pe_only.yaml           # E1-PE ✅
├── ortholora_internvl35_2b.yaml                            # E3-B Ortho-LoRA DenseCL ✅
└── ortholora_vicreg_internvl35_2b.yaml                     # E3-B Ortho-LoRA VICReg-pw ⏳ (eval 진행중)

scripts/
├── baseline_finetune.py    # 학습 진입점
├── baseline_eval.py        # 평가 진입점
└── merge_lora.py           # PEFT TIES/DARE merge + eval (신규)

runs/baseline/
|─ trackA_internvl35-2b_native/eval/metrics.json              # CIDEr=0.3405 ✅
|─ trackA_qwen25-3b_native/eval/metrics.json                 # CIDEr=0.3306 ✅
|─ trackA_gemma3-4b_native/eval/metrics.json                 # CIDEr=0.3363 ✅
|─ trackA_internvl35-1b_native/eval/metrics.json             # CIDEr=0.3171 ✅
|─ trackB_internvl35-2b_densecl50/eval/metrics.json         # CIDEr=0.3603 ✅
|─ trackB_qwen25-3b_densecl50/eval/metrics.json             # CIDEr=0.3423 ✅
|─ trackB_gemma3-4b_densecl50/eval/metrics.json             # CIDEr=0.3362 ✅
|─ trackB_internvl35-2b_vicreg25/eval/metrics.json          # CIDEr=0.3594 ✅
|─ trackB_internvl35-2b_vicreg50/eval/metrics.json          # CIDEr=0.3609 ✅
|─ trackB_qwen25-3b_vicreg50/eval/metrics.json              # CIDEr=0.3425 ✅
|─ e1a_internvl35-2b_cubemap_noloss/eval/metrics.json       # CIDEr=0.3579 ✅
|─ e1b_internvl35-2b_anyres_e2p_noloss/eval/metrics.json   # CIDEr=0.3598 ✅
|─ e1c_internvl35-2b_anyres_e2p_densecl25/eval/metrics.json # CIDEr=0.3600 ✅
|─ e1pe_internvl35-2b_anyres_e2p_pe_only/eval/metrics.json # CIDEr=0.3595 ✅
|─ e3a_internvl35-2b_ties_densecl/eval/metrics.json        # CIDEr=0.3607 ✅
|─ e3a_internvl35-2b_ties_vicreg50/eval/metrics.json       # CIDEr=0.3612 ✅
|─ e3b_internvl35-2b_ortholora_densecl/eval/metrics.json   # CIDEr=0.3591 ✅
└─ e3b_internvl35-2b_ortholora_vicreg50/eval/metrics.json  # ⏳ eval 진행중
```
