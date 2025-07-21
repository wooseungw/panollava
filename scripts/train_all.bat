@echo off
REM Complete 3-Stage Training Pipeline
REM ===================================

echo ==========================================
echo   Panorama VLM - Complete Training Pipeline
echo ==========================================
echo.

REM 기본 설정 확인
if not defined CSV_TRAIN (
    if not defined CSV_VAL (
        echo ⚠️  Please set CSV_TRAIN and CSV_VAL environment variables:
        echo    set CSV_TRAIN=path\to\train.csv
        echo    set CSV_VAL=path\to\val.csv
        echo.
        echo Using default paths...
    )
)

if not defined CSV_TRAIN set CSV_TRAIN=data/quic360/train.csv
if not defined CSV_VAL set CSV_VAL=data/quic360/valid.csv
if not defined CUDA_VISIBLE_DEVICES set CUDA_VISIBLE_DEVICES=0
if not defined WANDB_PROJECT set WANDB_PROJECT=panorama-vlm

echo Configuration:
echo   - Training data: %CSV_TRAIN%
echo   - Validation data: %CSV_VAL%
echo   - CUDA devices: %CUDA_VISIBLE_DEVICES%
echo   - WandB project: %WANDB_PROJECT%
echo.

REM Stage 1: Vision Training
echo 🚀 Starting Stage 1: Vision Encoder Training...
echo ==============================================
call scripts\train_vision.bat %*
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo.

REM Stage 2: Resampler Training
echo 🚀 Starting Stage 2: Resampler Training...
echo ==========================================
call scripts\train_resampler.bat %*
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo.

REM Stage 3: Fine-tuning
echo 🚀 Starting Stage 3: Fine-tuning with LoRA...
echo =============================================
call scripts\train_finetune.bat %*
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo.

echo 🎉 Complete training pipeline finished successfully!
echo.
echo Results:
echo   - Vision model: .\runs\e2p_vision_mlp\
echo   - Resampler model: .\runs\e2p_resampler_mlp\
echo   - Fine-tuned model: .\runs\e2p_finetune_mlp\
echo.
echo Ready for inference! 🚀