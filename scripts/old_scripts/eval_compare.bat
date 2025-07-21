@echo off
REM =============================================================================
REM PanoLLaVA Model Comparison Evaluation Script for Windows
REM =============================================================================

REM Load common configuration
call "%~dp0config.bat"

echo ==========================================
echo PanoLLaVA Model Comparison Evaluation
echo ==========================================

REM Configuration
set CSV_VAL=%1
if "%CSV_VAL%"=="" set CSV_VAL=data/quic360/test.csv
set BATCH_SIZE=1
set OUTPUT_DIR=%EVAL_OUTPUT_DIR%/comparison_results

echo Validation data: %CSV_VAL%
echo Output directory: %OUTPUT_DIR%

REM Validate input file
if not exist "%CSV_VAL%" (
    echo Error: Evaluation data file not found: %CSV_VAL%
    pause
    exit /b 1
)

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ==========================================
echo Checking available checkpoints...
echo ==========================================

REM Check for resampler checkpoint
set RESAMPLER_AVAILABLE=0
if exist "%RESAMPLER_CHECKPOINT_DIR%\best.ckpt" (
    set RESAMPLER_CHECKPOINT=%RESAMPLER_CHECKPOINT_DIR%\best.ckpt
    set RESAMPLER_AVAILABLE=1
    echo Found resampler checkpoint: %RESAMPLER_CHECKPOINT%
) else if exist "%RESAMPLER_CHECKPOINT_DIR%\last.ckpt" (
    set RESAMPLER_CHECKPOINT=%RESAMPLER_CHECKPOINT_DIR%\last.ckpt
    set RESAMPLER_AVAILABLE=1
    echo Found resampler checkpoint: %RESAMPLER_CHECKPOINT%
) else (
    echo Warning: No resampler checkpoint found
)

REM Check for finetune checkpoint
set FINETUNE_AVAILABLE=0
if exist "%FINETUNE_CHECKPOINT_DIR%\best.ckpt" (
    set FINETUNE_CHECKPOINT=%FINETUNE_CHECKPOINT_DIR%\best.ckpt
    set FINETUNE_AVAILABLE=1
    echo Found finetune checkpoint: %FINETUNE_CHECKPOINT%
) else if exist "%FINETUNE_CHECKPOINT_DIR%\last.ckpt" (
    set FINETUNE_CHECKPOINT=%FINETUNE_CHECKPOINT_DIR%\last.ckpt
    set FINETUNE_AVAILABLE=1
    echo Found finetune checkpoint: %FINETUNE_CHECKPOINT%
) else (
    echo Warning: No finetune checkpoint found
)

if %RESAMPLER_AVAILABLE%==0 if %FINETUNE_AVAILABLE%==0 (
    echo Error: No trained models found for comparison
    echo Make sure you have completed at least one training stage
    pause
    exit /b 1
)

echo ==========================================
echo Starting Model Comparison Evaluation
echo ==========================================

REM Evaluate resampler model if available
if %RESAMPLER_AVAILABLE%==1 (
    echo.
    echo Evaluating Resampler Model...
    echo ----------------------------------------
    
    python eval.py ^
        --checkpoint "%RESAMPLER_CHECKPOINT%" ^
        --csv-val "%CSV_VAL%" ^
        --vision-name "%VISION_MODEL%" ^
        --lm-name "%LM_MODEL%" ^
        --resampler "%RESAMPLER%" ^
        --crop-strategy "%CROP_STRATEGY%" ^
        --batch-size %BATCH_SIZE% ^
        --max-new-tokens %MAX_NEW_TOKENS% ^
        --temperature %TEMPERATURE% ^
        --output-dir "%OUTPUT_DIR%/resampler" ^
        --num-workers 1
    
    if %errorlevel% neq 0 (
        echo Warning: Resampler evaluation failed
    ) else (
        echo Resampler evaluation completed
    )
)

REM Evaluate finetune model if available
if %FINETUNE_AVAILABLE%==1 (
    echo.
    echo Evaluating Finetune Model...
    echo ----------------------------------------
    
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
        --output-dir "%OUTPUT_DIR%/finetune" ^
        --num-workers 1
    
    if %errorlevel% neq 0 (
        echo Warning: Finetune evaluation failed
    ) else (
        echo Finetune evaluation completed
    )
)

echo ==========================================
echo Model Comparison Evaluation Completed!
echo Results saved to: %OUTPUT_DIR%
echo ==========================================

if %RESAMPLER_AVAILABLE%==1 echo Resampler results: %OUTPUT_DIR%/resampler
if %FINETUNE_AVAILABLE%==1 echo Finetune results: %OUTPUT_DIR%/finetune

pause
