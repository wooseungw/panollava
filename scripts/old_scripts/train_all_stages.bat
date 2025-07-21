@echo off
REM =============================================================================
REM Full 3-Stage Training Pipeline for Windows
REM =============================================================================

REM Load common configuration
call "%~dp0config.bat"

echo ========================================
echo PanoLLaVA Full Training Pipeline
echo ========================================

REM Print configuration
call :print_config

REM Validate data files
if not exist "%CSV_TRAIN%" (
    echo Error: Training data file not found: %CSV_TRAIN%
    pause
    exit /b 1
)

if not exist "%CSV_VAL%" (
    echo Error: Validation data file not found: %CSV_VAL%
    pause
    exit /b 1
)

echo Starting full 3-stage training pipeline...
echo Training data: %CSV_TRAIN%
echo Validation data: %CSV_VAL%

REM Generate timestamp
for /f "delims=" %%i in ('powershell -command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set TIMESTAMP=%%i

REM =============================================================================
REM Stage 1: Vision Training
REM =============================================================================
echo.
echo =========================================
echo Stage 1/3: Vision Training (VICReg)
echo =========================================

python train.py ^
    --stage vision ^
    --vision-name "%VISION_MODEL%" ^
    --lm-name "%LM_MODEL%" ^
    --epochs "%VISION_EPOCHS%" ^
    --batch-size "%VISION_BATCH_SIZE%" ^
    --resampler "%RESAMPLER%" ^
    --crop-strategy "%CROP_STRATEGY%" ^
    --csv-train "%CSV_TRAIN%" ^
    --csv-val "%CSV_VAL%" ^
    --num-workers "%NUM_WORKERS%" ^
    --wandb-project "%WANDB_PROJECT%"

if %errorlevel% neq 0 (
    echo Error: Stage 1 training failed
    pause
    exit /b %errorlevel%
)

echo Stage 1 completed. Checkpoint: %VISION_CHECKPOINT_DIR%/best.ckpt

REM =============================================================================
REM Stage 2: Resampler Training
REM =============================================================================
echo.
echo =========================================
echo Stage 2/3: Resampler Training
echo =========================================

python train.py ^
    --stage resampler ^
    --vision-name "%VISION_MODEL%" ^
    --lm-name "%LM_MODEL%" ^
    --resampler "%RESAMPLER%" ^
    --epochs "%RESAMPLER_EPOCHS%" ^
    --batch-size "%RESAMPLER_BATCH_SIZE%" ^
    --crop-strategy "%CROP_STRATEGY%" ^
    --csv-train "%CSV_TRAIN%" ^
    --csv-val "%CSV_VAL%" ^
    --num-workers "%NUM_WORKERS%" ^
    --wandb-project "%WANDB_PROJECT%" ^
    --resume-from "%VISION_CHECKPOINT_DIR%/best.ckpt"

if %errorlevel% neq 0 (
    echo Error: Stage 2 training failed
    pause
    exit /b %errorlevel%
)

echo Stage 2 completed. Checkpoint: %RESAMPLER_CHECKPOINT_DIR%/best.ckpt

REM =============================================================================
REM Stage 3: Finetune Training
REM =============================================================================
echo.
echo =========================================
echo Stage 3/3: Finetune Training
echo =========================================

python train.py ^
    --stage finetune ^
    --vision-name "%VISION_MODEL%" ^
    --lm-name "%LM_MODEL%" ^
    --resampler "%RESAMPLER%" ^
    --epochs "%FINETUNE_EPOCHS%" ^
    --batch-size "%FINETUNE_BATCH_SIZE%" ^
    --crop-strategy "%CROP_STRATEGY%" ^
    --csv-train "%CSV_TRAIN%" ^
    --csv-val "%CSV_VAL%" ^
    --num-workers "%NUM_WORKERS%" ^
    --wandb-project "%WANDB_PROJECT%" ^
    --resume-from "%RESAMPLER_CHECKPOINT_DIR%/best.ckpt"

if %errorlevel% neq 0 (
    echo Error: Stage 3 training failed
    pause
    exit /b %errorlevel%
)

echo Stage 3 completed. Checkpoint: %FINETUNE_CHECKPOINT_DIR%/best.ckpt

echo.
echo ========================================
echo All stages completed successfully!
echo Final model: %FINETUNE_CHECKPOINT_DIR%/best.ckpt
echo ========================================

pause
