@echo off
REM =============================================================================
REM Stage 3: Finetune Training for Windows
REM =============================================================================

REM Load common configuration
call "%~dp0config.bat"

REM Print configuration
call :print_config
echo Stage: Finetune Training
echo Batch Size: %FINETUNE_BATCH_SIZE%
echo Epochs: %FINETUNE_EPOCHS%
echo ========================================

set RESAMPLER_CHECKPOINT=%RESAMPLER_CHECKPOINT_DIR%/best.ckpt

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
    --num-workers "%NUM_WORKERS%"

if %errorlevel% neq 0 (
    echo Error: Stage 3 training failed
    pause
    exit /b %errorlevel%
)

echo Stage 3 completed successfully
pause
