@echo off
REM =============================================================================
REM PanoLLaVA Finetune Model Evaluation Script for Windows
REM =============================================================================

REM Load common configuration
call "%~dp0config.bat"

echo ==========================================
echo PanoLLaVA Finetune Model Evaluation
echo ==========================================

REM Configuration
set CSV_VAL=%1
if "%CSV_VAL%"=="" set CSV_VAL=data/quic360/test.csv
set BATCH_SIZE=1
set OUTPUT_DIR=%EVAL_OUTPUT_DIR%/finetune_eval_results

echo Searching for finetune model checkpoint...

REM Find checkpoint files (simplified for Windows)
set CHECKPOINT_FOUND=0
if exist "%FINETUNE_CHECKPOINT_DIR%\best.ckpt" (
    set FINETUNE_CHECKPOINT=%FINETUNE_CHECKPOINT_DIR%\best.ckpt
    set CHECKPOINT_FOUND=1
) else if exist "%FINETUNE_CHECKPOINT_DIR%\last.ckpt" (
    set FINETUNE_CHECKPOINT=%FINETUNE_CHECKPOINT_DIR%\last.ckpt
    set CHECKPOINT_FOUND=1
)

if %CHECKPOINT_FOUND%==0 (
    echo Error: No finetune checkpoints found in %FINETUNE_CHECKPOINT_DIR%
    echo Make sure you have completed the finetune stage training
    pause
    exit /b 1
)

echo Found checkpoint: %FINETUNE_CHECKPOINT%

REM Validate input file
if not exist "%CSV_VAL%" (
    echo Error: Evaluation data file not found: %CSV_VAL%
    pause
    exit /b 1
)

echo Evaluation data: %CSV_VAL%
echo Output directory: %OUTPUT_DIR%

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ==========================================
echo Starting Finetune Model Evaluation
echo ==========================================

python eval.py ^
    --checkpoint "%FINETUNE_CHECKPOINT%" ^
    --csv-val "%CSV_VAL%" ^
    --vision-name "%VISION_MODEL%" ^
    --lm-name "%LM_MODEL%" ^
    --resampler "%RESAMPLER%" ^
    --crop-strategy "%CROP_STRATEGY%" ^
    --batch-size %BATCH_SIZE% ^
    --max-new-tokens %MAX_NEW_TOKENS% ^
    --temperature %TEMPERATURE% ^
    --output-dir "%OUTPUT_DIR%" ^
    --num-workers 1

if %errorlevel% neq 0 (
    echo Error: Evaluation failed
    pause
    exit /b %errorlevel%
)

echo ==========================================
echo Evaluation completed successfully!
echo Results saved to: %OUTPUT_DIR%
echo ==========================================

pause
