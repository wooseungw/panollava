# 🐛 Validation 크래시 버그 수정

## 문제 상황
- **증상**: Validation 단계 시작 직후 학습이 중단됨
- **로그**: `training.log`의 마지막 줄에서 validation batch 정보 출력 후 멈춤
  ```
  2025-10-19 18:27:04,856 - [VAL] First validation batch keys: [...]
  2025-10-19 18:27:04,858 - [VAL] pixel_values shape: torch.Size([16, 9, 3, 256, 256])
  ```
- **원인**: `validation_step` 및 `training_step`의 **잘못된 에러 처리 반환값**

## 근본 원인

### 문제 코드
```python
# ❌ 잘못된 코드 (빈 텐서 반환)
return torch.zeros([], device=self.device)
```

이 코드는 **빈 텐서 (empty tensor)**를 반환하는데, PyTorch Lightning이 기대하는 것은 **스칼라 텐서**입니다.

### 정상 동작 비교
```python
# 정상적인 loss 반환
loss = out["loss"]  # Shape: torch.Size([])  <- 스칼라 텐서
return loss

# ❌ 에러 처리 시 반환 (문제)
return torch.zeros([], device=self.device)  # Shape: torch.Size([]) <- 빈 텐서 (스칼라가 아님!)
```

### 기술적 설명
```python
# 빈 텐서 vs 스칼라 텐서
empty = torch.zeros([])       # Shape: torch.Size([]), numel: 0
scalar = torch.tensor(0.0)    # Shape: torch.Size([]), numel: 1

# PyTorch Lightning은 스칼라를 기대
# 빈 텐서는 aggregation 시 문제 발생
```

## 수정 내역

### 파일: `scripts/train.py`

#### 1. training_step (3곳)
```python
# Before (Line ~320, ~365, ~371)
return torch.zeros([], device=self.device, requires_grad=True)

# After
return torch.tensor(0.0, device=self.device, requires_grad=True)
```

#### 2. validation_step (4곳)
```python
# Before (Line ~389, ~394, ~407, ~413, ~418)
return torch.zeros([], device=self.device)

# After  
return torch.tensor(0.0, device=self.device, requires_grad=True)
```

### 수정 위치 상세

| 라인 | 위치 | 에러 타입 | 설명 |
|-----|-----|----------|------|
| ~320 | training_step | Non-finite loss | loss가 NaN/Inf일 때 |
| ~365 | training_step | OOM | GPU 메모리 부족 |
| ~371 | training_step | Runtime error | 기타 런타임 에러 |
| ~377 | training_step | General exception | 예상치 못한 에러 |
| ~389 | validation_step | No loss key | 모델 출력에 'loss' 없음 |
| ~394 | validation_step | Non-finite loss | validation loss NaN/Inf |
| ~407 | validation_step | OOM | validation OOM |
| ~413 | validation_step | Runtime error | validation 런타임 에러 |
| ~418 | validation_step | General exception | validation 예상치 못한 에러 |

## 영향 분석

### 1. **Lightning Aggregation**
PyTorch Lightning은 모든 step의 반환값을 collect하여 epoch-level metrics 계산:
```python
# Lightning 내부 (simplified)
epoch_losses = []
for batch in dataloader:
    loss = model.validation_step(batch, idx)
    epoch_losses.append(loss)

# ❌ 빈 텐서가 섞이면
mean_loss = torch.mean(torch.stack(epoch_losses))  # 계산 오류!

# ✅ 스칼라 텐서만 있으면
mean_loss = torch.mean(torch.stack(epoch_losses))  # 정상 동작
```

### 2. **DDP/Multi-GPU 환경**
분산 학습에서 rank 간 텐서 동기화 시 shape 불일치:
```python
# Rank 0: torch.Size([])  (빈 텐서, numel=0)
# Rank 1: torch.Size([])  (스칼라, numel=1)
# → torch.distributed.all_reduce 실패 가능
```

### 3. **Gradient Computation**
`requires_grad=True`가 필요한 이유:
```python
# Gradient flow를 유지하여 optimizer step이 정상 동작
# (에러 발생 시에도 학습 루프가 깨지지 않도록)
```

## 테스트 결과

### 수정 전
```bash
# Validation 시작 → 즉시 중단
[VAL] First validation batch keys: [...]
[VAL] pixel_values shape: torch.Size([16, 9, 3, 256, 256])
<프로세스 종료>
```

### 수정 후 (예상)
```bash
[VAL] First validation batch keys: [...]
[VAL] pixel_values shape: torch.Size([16, 9, 3, 256, 256])
[VAL][Epoch 0] mean loss: 0.XXXX  # 정상 완료
```

## 추가 개선 사항

### 1. Traceback 로깅 추가
```python
# Before
logger.error(f"Runtime error: {e}")

# After
import traceback
logger.error("Traceback:\n" + traceback.format_exc())
```

### 2. 에러 타입별 로깅 강화
- OOM: 명확한 메모리 부족 메시지
- Runtime error: 상세 traceback
- General exception: 전체 컨텍스트

## 재현 방법 (디버깅용)

문제를 재현하려면:
```python
# 의도적으로 빈 텐서 반환
def validation_step(self, batch, batch_idx):
    return torch.zeros([], device=self.device)
```

정상 동작 확인:
```python
# 스칼라 텐서 반환
def validation_step(self, batch, batch_idx):
    return torch.tensor(0.0, device=self.device, requires_grad=True)
```

## 체크리스트

### 완료 ✅
- [x] training_step 에러 처리 수정 (3곳)
- [x] validation_step 에러 처리 수정 (4곳)
- [x] Traceback 로깅 추가
- [x] 문서화

### 권장 사항 📋
- [ ] 유닛 테스트 추가 (에러 처리 경로)
- [ ] CI/CD에 validation step 테스트 추가
- [ ] 다른 LightningModule 메서드도 점검 (test_step 등)

## 참고 자료

- [PyTorch Lightning - LightningModule API](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)
- [PyTorch - Tensor Operations](https://pytorch.org/docs/stable/tensors.html)
- [Debugging DDP](https://pytorch.org/docs/stable/notes/ddp.html)

---

**수정일**: 2025년 10월 19일 (올바른 날짜: 10월, 오타 수정)  
**파일**: `scripts/train.py`  
**변경**: 7개 return 문 (빈 텐서 → 스칼라 텐서)
