@echo off
REM Stage 2: Resampler Training
REM ===========================

echo === Panorama VLM - Stage 2: Resampler Training ===
echo Starting resampler training with vision encoder...

REM 기본 설정
if not defined CUDA_VISIBLE_DEVICES set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=%PYTHONPATH%;%CD%

REM 이전 stage 체크포인트 자동 탐지
set VISION_CKPT_DIR=.\runs\e2p_vision_mlp
if exist "%VISION_CKPT_DIR%\best.ckpt" (
    set RESUME_FROM=%VISION_CKPT_DIR%\best.ckpt
    echo ✓ Found vision checkpoint: %RESUME_FROM%
) else if exist "%VISION_CKPT_DIR%\last.ckpt" (
    set RESUME_FROM=%VISION_CKPT_DIR%\last.ckpt
    echo ✓ Found vision checkpoint: %RESUME_FROM%
) else (
    echo  No vision checkpoint found in %VISION_CKPT_DIR%
    echo    Make sure to run vision training first or specify --resume-from manually
    set RESUME_FROM=
)

REM 환경 변수를 통한 설정 오버라이드 (선택사항)
REM set PANO_VLM_TRAINING_LEARNING_RATE=5e-5
REM set PANO_VLM_DATA_BATCH_SIZE=6
REM set PANO_VLM_VICREG_LOSS_WEIGHT=0.5

REM 기본값 설정
if not defined CSV_TRAIN set CSV_TRAIN=data/quic360/train.csv
if not defined CSV_VAL set CSV_VAL=data/quic360/valid.csv
if not defined WANDB_PROJECT set WANDB_PROJECT=panorama-vlm

REM Python 스크립트 실행
python train.py ^
    --config-stage resampler ^
    --csv-train "%CSV_TRAIN%" ^
    --csv-val "%CSV_VAL%" ^
    --wandb-project "%WANDB_PROJECT%" ^
    --resume-from "%RESUME_FROM%" ^
    %*

if %ERRORLEVEL% neq 0 (
    echo  Resampler training failed!
    exit /b %ERRORLEVEL%
)

echo ✓ Resampler training completed!
echo Next step: run scripts\train_finetune.bat