# PanoAdapt — 실험 계획 및 결과

> 최종 업데이트: 2026-02-27
> 목적: 상용 VLM에 파노라마 적응 기법(PanoRoPE + Overlap Loss) 적용 효과 검증

---

## 1. 연구 개요

**PanoAdapt**: 기존 상용 VLM을 재학습 없이 파노라마 이미지에 적응시키는 경량 방법론.

- **PanoRoPE**: 파노라마의 연속적 yaw 구조를 반영한 1D/3D positional encoding
- **Overlap Loss**: 인접 뷰 간 겹치는 영역의 feature consistency를 강제하는 self-supervised loss
- **LoRA**: 전체 파라미터의 ~0.6%만 학습 (r=32, alpha=64)

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
| Precision | bf16 |
| GPU | 1× RTX 3090 (24GB) |
| Dataset | QuIC-360 |
| Test set | 5,349 samples |
| Decoding | Greedy, max_new_tokens=128 |
| Eval metrics | BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE |

> **입력 전략 선정 근거**: Qwen2.5-VL-3B 기준 anyres_e2p가 CIDEr=0.3389로 4개 전략 중 최고 (부록 A 참조).

---

## 3. Native Baseline (PanoAdapt 없음)

각 VLM의 native image processor + anyres_e2p 9-view 입력, LoRA 1 epoch.

| Model | BLEU-4 ↑ | METEOR ↑ | ROUGE-L ↑ | CIDEr ↑ | SPICE ↑ |
|-------|:---:|:---:|:---:|:---:|:---:|
| **InternVL3.5-2B** | 0.0443 | 0.1111 | 0.2462 | **0.3405** | **0.1661** |
| Gemma3-4B | 0.0420 | 0.1081 | 0.2453 | 0.3363 | 0.1636 |
| Qwen2.5-VL-3B | **0.0443** | **0.1125** | 0.2427 | 0.3306 | 0.1548 |
| InternVL3.5-1B | 0.0389 | 0.1065 | **0.2462** | 0.3171 | 0.1606 |

---

## 4. PanoAdapt 실험

### 4.1 Overlap Loss 종류

| Loss | 방식 | 핵심 |
|------|------|------|
| **DenseCL** | Symmetric InfoNCE @ patch level | 인접 뷰 overlap strip의 대응 패치를 N×N similarity matrix로 학습 |
| **VICReg-pairwise** | MSE + variance + covariance @ pair level | 각 인접 뷰 쌍 내부에서 통계 계산, negatives 불필요 |

### 4.2 결과 — DenseCL (overlap 50%)

| Model | Native CIDEr | PanoAdapt CIDEr | Δ | BLEU-4 | METEOR | ROUGE-L | SPICE |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| InternVL3.5-2B | 0.3405 | **0.3603** | **+5.8%** ✅ | 0.0457 | 0.1137 | 0.2492 | 0.1720 |
| Qwen2.5-VL-3B | 0.3306 | 0.3396 | +2.7% | 0.0424 | 0.1140 | 0.2449 | 0.1619 |
| Gemma3-4B | 0.3363 | 0.3362 | -0.03% | 0.0438 | 0.1162 | 0.2509 | 0.1685 |

### 4.3 결과 — VICReg-pairwise

| Model | Overlap | Native CIDEr | PanoAdapt CIDEr | Δ | BLEU-4 | METEOR | ROUGE-L | SPICE |
|-------|:-------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| InternVL3.5-2B | 25% | 0.3405 | 0.3594 | +5.5% ✅ | 0.0457 | 0.1136 | 0.2601 | 0.1713 |
| InternVL3.5-2B | 50% | 0.3405 | 🔵 학습중 | — | — | — | — | — |
| Qwen2.5-VL-3B | 50% | 0.3306 | ⏳ 큐 | — | — | — | — | — |
| Gemma3-4B | 50% | 0.3363 | ⏳ 큐 | — | — | — | — | — |

### 4.4 종합 비교 (CIDEr 기준)

| Model | Native | DenseCL | VICReg-pw 25% | VICReg-pw 50% |
|-------|:---:|:---:|:---:|:---:|
| InternVL3.5-2B | 0.3405 | **0.3603** (+5.8%) | 0.3594 (+5.5%) | 🔵 |
| Qwen2.5-VL-3B | 0.3306 | 0.3396 (+2.7%) | — | ⏳ |
| Gemma3-4B | 0.3363 | 0.3362 (-0.03%) | — | ⏳ |

---

## 5. 핵심 발견

### F1. InternVL에서 PanoAdapt 효과 명확 (+5.8%)
InternVL3.5-2B에서 DenseCL이 CIDEr 0.3405→0.3603으로 유의미한 개선.
VICReg-pairwise 25% (0.3594)와 DenseCL (0.3603)의 차이는 0.09%p — loss 종류보다 overlap 구조 자체가 핵심일 가능성.

### F2. Qwen의 overlap loss는 사실상 무효 (구조적 문제)
Qwen2.5-VL의 vision encoder attention이 **fused `qkv = nn.Linear(dim, dim*3)`** 를 사용.
`target_modules=["q_proj", "k_proj", "v_proj"]`와 이름이 매칭되지 않아 **vision encoder에 LoRA가 삽입되지 않음**.

| 모델 | Vision Encoder Attn | LoRA 매칭 | Overlap Loss 효과 |
|------|-------------------|:---:|:---:|
| InternVL3.5 | 별도 `q_proj`/`k_proj`/`v_proj` | ✅ | ✅ vision LoRA 학습 |
| Gemma3 (SigLIP2) | 별도 `q_proj`/`k_proj`/`v_proj` | ✅ | ✅ vision LoRA 학습 |
| Qwen2.5-VL | fused `qkv` + `proj` | ❌ | ❌ dead loss |

→ **Qwen +2.7%는 PanoRoPE(spatial PE)만의 효과.**
→ **Gemma3 -0.03%는 별도 분석 필요** (LoRA는 매칭되나 효과 없음).

### F3. Gemma3는 반드시 bf16
Gemma3-4B-IT는 bfloat16으로 pretrain됨. fp16 학습 시 9-view × 256 tokens = 2304 image tokens attention에서 overflow → `loss=0.0, grad_norm=nan` 전체 실패. **bf16 필수.**

### F4. DenseCL vs VICReg-pairwise — InternVL 25%에서 거의 동일
CIDEr 차이 0.09%p. 50% overlap 결과 나오면 추가 비교 가능.

---

## 6. 실행 현황

### ✅ 완료

| 실험 | CIDEr |
|------|:-----:|
| Native InternVL3.5-2B | 0.3405 |
| Native Qwen2.5-VL-3B | 0.3306 |
| Native Gemma3-4B | 0.3363 |
| Native InternVL3.5-1B | 0.3171 |
| PanoAdapt DenseCL — InternVL3.5-2B | **0.3603** |
| PanoAdapt DenseCL — Qwen2.5-VL-3B | 0.3396 |
| PanoAdapt DenseCL — Gemma3-4B | 0.3362 |
| PanoAdapt VICReg-pw 25% — InternVL3.5-2B | 0.3594 |

### 🔵 진행중 (tmux: `gpu1-trackb:phase2-watcher`, CUDA_VISIBLE_DEVICES=1)

- VICReg-pw 50% InternVL3.5-2B 학습 — 22% (427/1983), ~1.5h 남음
- 이후 자동 체이닝: Qwen 50% → Gemma3 50%

| 시각 (KST) | 완료 예정 |
|-----------|---------|
| ~20:00 | VICReg-pw 50% InternVL 학습+eval |
| ~23:00 | VICReg-pw 50% Qwen 학습+eval |
| ~02:00 (28일) | VICReg-pw 50% Gemma3 학습+eval |

---

## 7. 향후 실험 계획

### E1. InternVL3.5-2B 완전 Ablation

**목적**: 입력 전략 효과와 Overlap Loss 효과를 독립적으로 분리.

| # | 입력 방식 | Loss | Overlap | 상태 | CIDEr |
|:---:|----------|------|:-------:|:----:|:-----:|
| 1 | Resize 256² | None | — | ✅ | 0.3054 |
| 2 | Native | None | — | ✅ | 0.3405 |
| 3 | Cubemap | None | — | ⏳ 대기 | — |
| 4 | AnyRes-E2P | None | — | ⏳ 대기 | — |
| 5 | AnyRes-E2P | DenseCL | 25% | ⏳ 대기 | — |
| 6 | AnyRes-E2P | DenseCL | 50% | ✅ | 0.3603 |
| 7 | AnyRes-E2P | VICReg-pw | 25% | ✅ | 0.3594 |
| 8 | AnyRes-E2P | VICReg-pw | 50% | 🔵 진행중 | — |

> **현재 VICReg-pw 큐 완료 후 실행 (실험 3개 추가필요: #3, #4, #5)**

**논문 서사:**
```
입력 전략 효과 (loss 없음, 수직 비교):
  Resize(0.305) → Native(0.340) → Cubemap(?) → AnyRes-E2P(?)

Overlap Loss 효과 (AnyRes-E2P 고정, 수평 비교):
  No loss(?) → DenseCL 25%(?) → DenseCL 50%(0.360)
              → VICReg-pw 25%(0.359) → VICReg-pw 50%(?)
```

### E2. Qwen B1-fix — Vision LoRA 활성화
`target_modules`에 `attn.qkv`, `attn.proj` 추가하여 vision encoder에도 LoRA 삽입.

```yaml
lora:
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "attn.qkv"    # Qwen vision encoder fused QKV
    - "attn.proj"   # Qwen vision encoder output projection
```

| 실험 | 목적 |
|------|------|
| B1-fix | vision LoRA 활성화 후 DenseCL 재실험 |
| B1-pe-only | overlap loss 제거, PanoRoPE만 → B1과 동일해야 함 (ablation) |

예상: B1-fix가 InternVL 수준(+5~6%)으로 개선되면 overlap loss 효과 입증.
B1-pe-only ≈ B1이면 기존 B1의 overlap loss가 무효였음을 실험적으로 확인.

### E3. Dense VICReg
DenseCL(InfoNCE)과 동일한 patch-level granularity에서 VICReg loss 적용.

| | InfoNCE | VICReg |
|---|:---:|:---:|
| **Dense (patch)** | ✅ DenseCL (완료) | ⬜ Dense VICReg (미구현) |
| **Coarse (pair)** | — | ✅ VICReg-pairwise (진행중) |

주의: VICReg variance/covariance population이 overlap 패치들(공간적으로 유사)로 구성되어 invariance ↔ variance 충돌 가능. E1 + VICReg-pw 50% 결과 확인 후 판단.
---

## 8. 버그 이력

| 날짜 | 버그 | 수정 |
|------|------|------|
| 2026-02-24 | `_unwrap_to_rope_model()` 무한 루프 | `base_model` 자기참조 사이클 체크 추가 |
| 2026-02-24 | `max_length: 1024` 부족 (9 views × 256 = 2304 tokens) | `max_length: 3072` 수정 |
| 2026-02-25 | Qwen DenseCL `Expected features with 2 or 4 dims, got 3` | `DenseCLLoss.forward` ndim==3 처리 추가 |
| 2026-02-26 | Gemma3 DenseCL `loss=0.0, grad_norm=nan` 전체 실패 | `dtype: float16 → bfloat16`, `mixed_precision: fp16 → bf16` |
| 2026-02-26 | Gemma3 `multi_modal_projector` output 3D `[N, 256, 2560]` | `_compute_densecl` else-branch ndim==3 직접 인덱싱 |

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
├── panoadapt_vicreg_pairwise_internvl35_2b.yaml            # VICReg-pw 50% InternVL 🔵
├── panoadapt_vicreg_pairwise_qwen25_3b.yaml                # VICReg-pw 50% Qwen ⏳
└── panoadapt_vicreg_pairwise_gemma3_4b.yaml                # VICReg-pw 50% Gemma3 ⏳

runs/baseline/
├── native_internvl35-2b/eval/metrics.json                  # CIDEr=0.3405
├── native_qwen25-vl-3b/eval/metrics.json                   # CIDEr=0.3306
├── native_gemma3-4b/eval/metrics.json                      # CIDEr=0.3363
├── native_internvl35-1b/eval/metrics.json                  # CIDEr=0.3171
├── panoadapt_internvl35-2b/eval/metrics.json               # CIDEr=0.3603 ✅
├── panoadapt_qwen25-vl-3b/eval/metrics.json                # CIDEr=0.3396 ✅
├── panoadapt_gemma3-4b/eval/metrics.json                   # CIDEr=0.3362 ✅
└── panoadapt_vicreg_pairwise_internvl35-2b_25overlap/
    └── eval/metrics.json                                   # CIDEr=0.3594 ✅
```
