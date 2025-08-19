# VICReg-L 설정 가이드

## 📋 설정 파일 개요

| 파일 | 용도 | VICReg-L | 적용 시기 |
|------|------|----------|-----------|
| `config.json` | 기존 기본 설정 | ❌ 비활성화 | 안정적인 베이스라인 |
| `config_mixed_transition.json` | 점진적 전환 | 🟡 낮은 가중치 (0.2) | 기존 → VICReg-L 전환 |
| `config_vicreg_local.json` | VICReg-L 주도 | ✅ 활성화 (0.5) | VICReg-L 본격 적용 |

## 🔧 VICReg-L 설정 파라미터

### 기본 설정
```json
{
  "use_vicreg_local": true,           // VICReg-L 활성화 여부
  "vicreg_local_weight": 0.5,         // 전체 손실에서 VICReg-L 비중
  "vicreg_loss_weight": 0.8           // 기존 VICReg 가중치 (조절)
}
```

### 세부 파라미터
```json
{
  "vicreg_local_inv_weight": 1.0,     // INV 손실 가중치 (불변성)
  "vicreg_local_var_weight": 1.0,     // VAR 손실 가중치 (분산)
  "vicreg_local_cov_weight": 0.01,    // COV 손실 가중치 (공분산, 보통 작게)
  "vicreg_local_inv_type": "l2",      // "l2" 또는 "cos" (불변성 손실 타입)
  "vicreg_local_gamma": 1.0           // 분산 정규화 목표값
}
```

## 🚀 사용 방법

### 1. 기존 VICReg 유지 (안전)
```bash
python train.py --config config.json --stage vision
```

### 2. 점진적 전환 (권장)
```bash
# 단계 1: 낮은 가중치로 시작
python train.py --config config_mixed_transition.json --stage vision

# 단계 2: 가중치 점진적 증가
python train.py --config config_vicreg_local.json --stage vision
```

### 3. 직접 VICReg-L (고급)
```bash
python train.py --config config_vicreg_local.json --stage vision
```

## 📊 모니터링 메트릭

### 기존 메트릭
- `vicreg_loss`: 기존 VICReg 손실
- `vicreg_raw`: 가중치 적용 전 VICReg 손실

### 새로운 메트릭 (VICReg-L)
- `vicreg_local_loss`: VICReg-L 총 손실
- `vicreg_local_inv`: INV 손실 (낮을수록 좋음)
- `vicreg_local_var`: VAR 손실 (낮을수록 좋음)  
- `vicreg_local_cov`: COV 손실 (낮을수록 좋음)
- `vicreg_local_pairs`: 유효한 뷰 쌍 개수

## ⚙️ 하이퍼파라미터 튜닝 가이드

### 시작 값 (권장)
```json
{
  "vicreg_local_weight": 0.2,         // 낮게 시작
  "vicreg_local_inv_weight": 1.0,     // 표준
  "vicreg_local_var_weight": 1.0,     // 표준
  "vicreg_local_cov_weight": 0.01     // VICReg 논문 기준
}
```

### 문제별 조정
| 문제 | 조정 방법 |
|------|-----------|
| 학습 불안정 | `vicreg_local_weight` ↓ (0.1~0.3) |
| 표현 붕괴 | `vicreg_local_var_weight` ↑ (1.5~2.0) |
| 과도한 유사성 | `vicreg_local_inv_weight` ↓ (0.5~0.8) |
| 특징 상관관계 | `vicreg_local_cov_weight` ↑ (0.02~0.05) |

### INV 손실 타입 선택
- `"l2"`: 일반적, 안정적 (기본값)
- `"cos"`: 방향성 중시, 각도 기반 유사성

## 🎯 단계별 전환 전략

### Phase 1: 검증 (1-2 에포크)
```json
{
  "vicreg_local_weight": 0.1,
  "vicreg_loss_weight": 1.0
}
```
→ VICReg-L이 정상 작동하는지 확인

### Phase 2: 점진적 증가 (2-3 에포크)
```json
{
  "vicreg_local_weight": 0.3,
  "vicreg_loss_weight": 0.9
}
```
→ VICReg-L 비중 증가, 기존 VICReg 유지

### Phase 3: 주도권 전환 (3+ 에포크)
```json
{
  "vicreg_local_weight": 0.5,
  "vicreg_loss_weight": 0.7
}
```
→ VICReg-L이 주도하되 기존 손실도 보완

## 🔍 디버깅 체크리스트

### 1. 메타데이터 확인
- [ ] `return_metadata=True` 설정
- [ ] 메타데이터에 `yaw`, `effective_fov` 포함
- [ ] `crop_strategy="e2p"` 사용

### 2. 손실 값 모니터링
- [ ] `vicreg_local_pairs > 0` (유효 쌍 존재)
- [ ] `vicreg_local_loss` 값이 합리적 (0.1~10 범위)
- [ ] INV/VAR/COV 균형 확인

### 3. 성능 지표
- [ ] 총 손실이 안정적으로 감소
- [ ] 기존 성능 대비 유지/개선
- [ ] OCS (Overlap Consistency Score) 향상

## 💡 Pro Tips

1. **점진적 적용**: 갑작스런 변화보다 단계적 전환
2. **메트릭 모니터링**: WandB 등에서 세부 손실들 추적  
3. **하이퍼파라미터 실험**: 데이터셋별로 최적값 다를 수 있음
4. **백업**: 기존 체크포인트 보관 후 실험
5. **A/B 테스트**: 기존 vs VICReg-L 성능 비교