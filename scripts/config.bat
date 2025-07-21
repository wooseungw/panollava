@echo off
REM =============================================================================
REM PanoLLaVA Training Configuration for Windows
REM =============================================================================

REM GPU 설정
set CUDA_VISIBLE_DEVICES=0
set WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974

REM 모델 설정
set VISION_MODEL=google/siglip-base-patch16-224
set LM_MODEL=Qwen/Qwen2.5-0.5B
set RESAMPLER=mlp
set CROP_STRATEGY=e2p

REM 데이터 설정
set CSV_TRAIN=data/quic360/train.csv
set CSV_VAL=data/quic360/valid.csv

REM 학습 설정
set NUM_WORKERS=8
set WANDB_PROJECT=panollava-training

REM Stage별 배치 크기 및 에포크
set VISION_BATCH_SIZE=16
set VISION_EPOCHS=3

set RESAMPLER_BATCH_SIZE=4
set RESAMPLER_EPOCHS=1

set FINETUNE_BATCH_SIZE=4
set FINETUNE_EPOCHS=1

REM 생성 설정 (평가용)
set MAX_NEW_TOKENS=64
set TEMPERATURE=0.7

REM 디렉토리 설정
set LOG_DIR=logs
set RUNS_DIR=runs
set EVAL_OUTPUT_DIR=eval_results

REM 체크포인트 경로 템플릿
set VISION_CHECKPOINT_DIR=runs/%CROP_STRATEGY%_vision_%RESAMPLER%
set RESAMPLER_CHECKPOINT_DIR=runs/%CROP_STRATEGY%_resampler_%RESAMPLER%
set FINETUNE_CHECKPOINT_DIR=runs/%CROP_STRATEGY%_finetune_%RESAMPLER%

REM 디렉토리 생성
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RUNS_DIR%" mkdir "%RUNS_DIR%"
if not exist "%VISION_CHECKPOINT_DIR%" mkdir "%VISION_CHECKPOINT_DIR%"
if not exist "%RESAMPLER_CHECKPOINT_DIR%" mkdir "%RESAMPLER_CHECKPOINT_DIR%"
if not exist "%FINETUNE_CHECKPOINT_DIR%" mkdir "%FINETUNE_CHECKPOINT_DIR%"
if not exist "%EVAL_OUTPUT_DIR%" mkdir "%EVAL_OUTPUT_DIR%"

REM 설정 출력 함수가 call 없이 끝나지 않도록 return 추가
exit /b 0

:print_config
echo ========================================
echo PanoLLaVA Configuration
echo ========================================
echo Vision Model: %VISION_MODEL%
echo Language Model: %LM_MODEL%
echo Resampler: %RESAMPLER%
echo Crop Strategy: %CROP_STRATEGY%
echo Training Data: %CSV_TRAIN%
echo Validation Data: %CSV_VAL%
echo Workers: %NUM_WORKERS%
echo WandB Project: %WANDB_PROJECT%
echo ========================================
goto :eof
