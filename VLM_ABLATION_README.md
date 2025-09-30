# HF VLM LoRA Ablation

이 스크립트는 Hugging Face에 등록된 VLM 래퍼(`llava`, `blip2`, `qwen-vl`, …)에 대해
LoRA 하이퍼파라미터를 바꾸어가며 학습/평가하는 실험용 드라이버입니다.

## 1. 설정 파일 점검
```bash
cat configs/vlm_ablation.yaml
```
- `models`: 실험에 사용할 모델을 정의합니다 (`hf_model_id`, `model_type`, LoRA 타겟 모듈 등).
- `lora_variants`: LoRA rank/alpha/dropout 조합을 추가하거나 수정합니다.
- `data`: CSV 경로와 이미지/텍스트 컬럼명을 지정합니다.
- `training`: Hugging Face `Trainer`에서 사용하는 공통 하이퍼파라미터입니다.

## 2. Ablation 실행
```bash
python vlm_ablation_study.py --config configs/vlm_ablation.yaml --output-dir ./results/vlm_lora_runs
```
- `--output-dir`를 지정하면 설정 파일의 `output_dir` 대신 사용됩니다.
- GPU 환경에서 `mixed_precision`을 `fp16`/`bf16`으로 설정하면 VRAM을 절약할 수 있습니다.

## 3. 결과 확인
```bash
ls -R ./results/vlm_lora_runs
```
- 각 실험(`모델명__LoRA조합`) 하위에 체크포인트, LoRA 어댑터, 메트릭 요약이 저장됩니다.
- 전체 결과는 `ablation_summary.json`에서 한번에 확인할 수 있습니다.

## 4. 모델 추가 방법
1. `models` 블록에 새 항목을 추가합니다. (예: 다른 Qwen-VL, LLaVA, BLIP-2 변형)
2. 필요 시 `processor_id`, `torch_dtype`, `lora_target_modules`를 조정합니다.
3. 이미지/텍스트 포맷이 동일하다면 추가 작업 없이 실행됩니다.

> 참고: `model_type`은 현재 `llava`, `qwen_vl`, `blip2`를 지원합니다. 새로운 유형이 필요하면
> `vlm_ablation_study.py`의 `build_adapter`와 Adapter 클래스를 확장하세요.
