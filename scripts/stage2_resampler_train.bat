@echo off
REM =============================================================================
REM Stage 2: Resampler Training for Windows
REM =============================================================================

REM Load common configuration
call "%~dp0config.bat"

REM Print configuration
call :print_config
echo Stage: Resampler Training
echo Batch Size: %RESAMPLER_BATCH_SIZE%
echo Epochs: %RESAMPLER_EPOCHS%
echo ========================================

set VISION_CHECKPOINT=%VISION_CHECKPOINT_DIR%/best.ckpt

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
    --num-workers "%NUM_WORKERS%"

if %errorlevel% neq 0 (
    echo Error: Stage 2 training failed
    pause
    exit /b %errorlevel%
)

echo Stage 2 completed successfully
pause
