@echo off
REM =============================================================================
REM Stage 1: Vision Encoder Training for Windows
REM =============================================================================

REM Load common configuration
call "%~dp0config.bat"

REM Print configuration
call :print_config
echo Stage: Vision Training
echo Batch Size: %VISION_BATCH_SIZE%
echo Epochs: %VISION_EPOCHS%
echo ========================================

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
    --num-workers "%NUM_WORKERS%"

if %errorlevel% neq 0 (
    echo Error: Stage 1 training failed
    pause
    exit /b %errorlevel%
)

echo Stage 1 completed successfully
pause
