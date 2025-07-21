@echo off
REM Stage 3: End-to-End Fine-tuning with LoRA
REM ===========================================

echo === Panorama VLM - Stage 3: Fine-tuning with LoRA ===
echo Starting end-to-end fine-tuning with LoRA...

REM 기본 설정
if not defined CUDA_VISIBLE_DEVICES set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=%PYTHONPATH%;%CD%

REM 이전 stage 체크포인트 자동 탐지
set RESAMPLER_CKPT_DIR=.\runs\e2p_resampler_mlp
if exist "%RESAMPLER_CKPT_DIR%\best.ckpt" (
    set RESUME_FROM=%RESAMPLER_CKPT_DIR%\best.ckpt
    echo ✓ Found resampler checkpoint: %RESUME_FROM%
) else if exist "%RESAMPLER_CKPT_DIR%\last.ckpt" (
    set RESUME_FROM=%RESAMPLER_CKPT_DIR%\last.ckpt
    echo ✓ Found resampler checkpoint: %RESUME_FROM%
) else (
    echo ⚠️  No resampler checkpoint found in %RESAMPLER_CKPT_DIR%
    echo    Make sure to run resampler training first or specify --resume-from manually
    set RESUME_FROM=
)

REM LoRA 설정 (환경 변수로 오버라이드 가능)
if not defined USE_LORA set USE_LORA=true
if not defined LORA_R set LORA_R=16
if not defined LORA_ALPHA set LORA_ALPHA=32
if not defined LORA_DROPOUT set LORA_DROPOUT=0.1

set PANO_VLM_MODEL_LORA_ENABLED=%USE_LORA%
set PANO_VLM_MODEL_LORA_R=%LORA_R%
set PANO_VLM_MODEL_LORA_ALPHA=%LORA_ALPHA%
set PANO_VLM_MODEL_LORA_DROPOUT=%LORA_DROPOUT%

REM 학습 설정 (환경 변수로 오버라이드 가능)
REM set PANO_VLM_TRAINING_LEARNING_RATE=2e-4
REM set PANO_VLM_DATA_BATCH_SIZE=2
REM set PANO_VLM_TRAINING_EPOCHS=3

echo LoRA Configuration:
echo   - Enabled: %PANO_VLM_MODEL_LORA_ENABLED%
echo   - Rank: %PANO_VLM_MODEL_LORA_R%
echo   - Alpha: %PANO_VLM_MODEL_LORA_ALPHA%
echo   - Dropout: %PANO_VLM_MODEL_LORA_DROPOUT%

REM 기본값 설정
if not defined CSV_TRAIN set CSV_TRAIN=data/quic360/train.csv
if not defined CSV_VAL set CSV_VAL=data/quic360/valid.csv
if not defined WANDB_PROJECT set WANDB_PROJECT=panorama-vlm

REM Python 스크립트 실행
python train.py ^
    --config-stage finetune ^
    --csv-train "%CSV_TRAIN%" ^
    --csv-val "%CSV_VAL%" ^
    --wandb-project "%WANDB_PROJECT%" ^
    --resume-from "%RESUME_FROM%" ^
    %*

if %ERRORLEVEL% neq 0 (
    echo ❌ Fine-tuning failed!
    exit /b %ERRORLEVEL%
)

echo ✓ Fine-tuning completed!
echo Training pipeline finished successfully!

REM LoRA 어댑터 위치 안내
set FINETUNE_CKPT_DIR=.\runs\e2p_finetune_mlp
echo.
echo LoRA adapter saved in: %FINETUNE_CKPT_DIR%\
echo Use this for inference or further fine-tuning.