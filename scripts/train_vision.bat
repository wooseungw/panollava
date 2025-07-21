@echo off
REM Stage 1: Vision Encoder Training with VICReg Loss
REM ==================================================

echo === Panorama VLM - Stage 1: Vision Training ===
echo Starting vision encoder training with VICReg loss...

REM 기본 설정
if not defined CUDA_VISIBLE_DEVICES set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=%PYTHONPATH%;%CD%

REM 환경 변수를 통한 설정 오버라이드 (선택사항)
REM set PANO_VLM_TRAINING_LEARNING_RATE=1e-4
REM set PANO_VLM_DATA_BATCH_SIZE=8
REM set PANO_VLM_TRAINING_EPOCHS=5

REM 기본값 설정
if not defined CSV_TRAIN set CSV_TRAIN=data/quic360/train.csv
if not defined CSV_VAL set CSV_VAL=data/quic360/valid.csv
if not defined WANDB_PROJECT set WANDB_PROJECT=panorama-vlm
if not defined RESUME_FROM set RESUME_FROM=

REM Python 스크립트 실행
python train.py ^
    --config-stage vision ^
    --csv-train "%CSV_TRAIN%" ^
    --csv-val "%CSV_VAL%" ^
    --wandb-project "%WANDB_PROJECT%" ^
    --resume-from "%RESUME_FROM%" ^
    %*

if %ERRORLEVEL% neq 0 (
    echo Vision training failed!
    exit /b %ERRORLEVEL%
)

echo ✓ Vision training completed!
echo Next step: run scripts\train_resampler.bat