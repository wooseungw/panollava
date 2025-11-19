# VLM 평가 스크립트 수정 요약

## 문제점

`scripts/evaluate_vlm_models.py`에서 Gemma3와 LLaVA-OneVision 모델의 예측(pred) 결과가 생성되지 않고 평가 메트릭도 작동하지 않는 문제가 있었습니다.

## 수정 사항

### 1. LLaVA-OneVision 처리 개선 (라인 820-930)

**문제**: LLaVA-OneVision은 `requires_vision_utils=True`로 설정되어 있었지만, 배치 처리 방식이 불안정했습니다.

**수정**: 
- 배치 처리에서 **개별 샘플 처리**로 변경
- 각 샘플마다 `process_vision_info`를 호출하여 이미지 입력 처리
- 명확한 디버그 로깅 추가
- 예외 발생 시 상세한 에러 정보 출력

```python
# LLaVA-OneVision과 Qwen2.5-VL은 개별 처리가 더 안정적
for sample_idx, (inst, img, ref, path) in enumerate(zip(...)):
    try:
        # 개별 샘플 처리
        messages = [{"role": "user", "content": [...]}]
        text = self.processor.apply_chat_template(...)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, ...)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        # ... 디코딩 및 결과 저장
    except Exception as e:
        # 상세한 에러 로깅
        logging.error(f"❌ 샘플 처리 실패: {e}")
        # 빈 예측 추가하여 레퍼런스와 매칭 유지
        predictions.append("")
```

### 2. Gemma3 처리 개선 (라인 930-1050)

**문제**: Gemma3의 개별 처리는 구현되어 있었지만, 예외 처리가 불충분했습니다.

**수정**:
- 예외 발생 시 더 상세한 정보 출력 (이미지 경로, instruction 등)
- 에러 로깅 개선

### 3. 메트릭 계산 강화 (라인 300-370)

**문제**: 
- 빈 예측이 있을 때 메트릭 계산이 실패하거나 경고 없이 넘어감
- eval.py 메트릭 계산 실패 시 traceback이 출력되지 않음

**수정**:
```python
def compute_text_metrics(predictions, references):
    logging.info(f"📊 메트릭 계산 시작: {len(predictions)} predictions")
    
    # 데이터 검증
    valid_count = sum(1 for p, r in zip(predictions, references) 
                     if p.strip() and r.strip())
    empty_pred_count = sum(1 for p in predictions if not p.strip())
    
    logging.info(f"  - 유효한 쌍: {valid_count}/{len(predictions)}")
    logging.info(f"  - 빈 예측: {empty_pred_count}")
    
    if valid_count == 0:
        logging.error("❌ 유효한 예측-정답 쌍이 없습니다!")
        return {}
```

### 4. 평가 완료 요약 개선 (라인 1115-1140)

**문제**: 빈 예측이 있을 때 어떤 샘플에서 문제가 발생했는지 알 수 없음

**수정**:
```python
empty_pred_count = sum(1 for p in predictions if not p.strip())
if empty_pred_count > 0:
    logging.warning(f"⚠️ {empty_pred_count}개의 빈 예측이 발견되었습니다!")
    # 처음 5개의 빈 예측 샘플 정보 출력
    empty_indices = [i for i, p in enumerate(predictions) if not p.strip()][:5]
    for idx in empty_indices:
        logging.warning(f"  - 샘플 {idx}: image={image_paths[idx]}")
        logging.warning(f"    instruction={instructions[idx][:80]}...")
```

## 테스트 방법

### 1. 소규모 테스트 (권장)

```bash
# Gemma3 테스트 (1개 샘플만)
python scripts/evaluate_vlm_models.py \
  --data_csv data/train_stanford_dummy_anno.csv \
  --models gemma-3-4b \
  --max_samples 1 \
  --log_level DEBUG \
  --output_dir eval_results/debug

# LLaVA-OneVision 테스트 (1개 샘플만)
python scripts/evaluate_vlm_models.py \
  --data_csv data/train_stanford_dummy_anno.csv \
  --models llava-onevision-0.5b \
  --max_samples 1 \
  --log_level DEBUG \
  --output_dir eval_results/debug
```

### 2. 전체 테스트

```bash
# 여러 모델 평가 (10개 샘플)
python scripts/evaluate_vlm_models.py \
  --data_csv data/train_stanford_dummy_anno.csv \
  --models gemma-3-4b llava-onevision-0.5b llava-onevision-7b \
  --max_samples 10 \
  --log_level INFO \
  --output_dir eval_results
```

## 디버깅 팁

### 1. 로그 확인

디버그 모드(`--log_level DEBUG`)에서 다음 로그를 확인하세요:

```
🔍 [DEBUG] Using vision_utils path for llava-onevision-0.5b
🔍 [DEBUG] Processing 1 samples individually
🔍 [DEBUG] === Sample 0/1 ===
🔍 [DEBUG] Instruction: ...
🔍 [DEBUG] Chat template output length: 123
🔍 [DEBUG] image_inputs=1, video_inputs=0
🔍 [DEBUG] Processor output keys: dict_keys([...])
🔍 [DEBUG] Starting generation...
🔍 [DEBUG] Generation complete. Output shape: torch.Size([1, 456])
🔍 [DEBUG] Decoded prediction: ...
```

### 2. 에러 발생 시

에러가 발생하면 다음 정보가 출력됩니다:

```
❌ 샘플 처리 실패 (batch=0, sample=0, model=gemma-3-4b)
   Error: ...
   Image: data/path/to/image.jpg
   Instruction: What is shown in this image?...
   Traceback:
   ...
```

### 3. 빈 예측 확인

평가 완료 후 다음과 같은 요약이 출력됩니다:

```
✅ 평가 완료: gemma-3-4b
총 예측 수: 10
총 레퍼런스 수: 10
빈 예측 수: 3
⚠️ 3개의 빈 예측이 발견되었습니다!
  - 샘플 2: image=data/...
    instruction=Describe the scene...
```

### 4. 메트릭 계산 확인

```
📊 메트릭 계산 시작: 10 predictions, 10 references
  - 유효한 쌍: 7/10
  - 빈 예측: 3
  - 빈 참조: 0
📊 로컬 메트릭 계산: 7 유효 쌍
```

## 예상 결과

수정 후 다음과 같은 개선을 기대할 수 있습니다:

1. **Gemma3**: 예측이 정상적으로 생성되며, 에러 발생 시 상세한 정보 출력
2. **LLaVA-OneVision**: 개별 샘플 처리로 안정성 향상, 예측 생성 성공
3. **메트릭 계산**: 빈 예측이 있어도 유효한 샘플에 대해 메트릭 계산
4. **디버깅**: 명확한 로그로 문제 원인 파악 용이

## 결과 파일

평가 완료 후 다음 파일이 생성됩니다:

```
eval_results/
└── ablation/
    ├── gemma-3-4b/
    │   ├── metrics.json        # 메트릭 결과
    │   └── predictions.csv     # 예측 결과 (instruction, reference, prediction)
    └── llava-onevision-0.5b/
        ├── metrics.json
        └── predictions.csv
```

## 추가 개선 사항 (향후)

1. **배치 처리 최적화**: Gemma3와 LLaVA-OneVision의 배치 처리 안정화
2. **재시도 로직**: 네트워크 에러 등 일시적 문제 시 자동 재시도
3. **캐싱**: 이미지 전처리 결과 캐싱으로 속도 향상
4. **병렬 처리**: 여러 모델 평가 시 병렬 실행

## 문의사항

문제가 계속되면 다음 정보와 함께 문의하세요:

1. 전체 로그 (`--log_level DEBUG`)
2. 사용한 명령어
3. `predictions.csv` 파일 (처음 몇 행)
4. GPU 메모리 사용량 (`nvidia-smi`)
