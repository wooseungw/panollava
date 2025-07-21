@echo off
REM =============================================================================
REM Custom Training Script with Flexible Configuration for Windows
REM =============================================================================

REM Load common configuration
call "%~dp0config.bat"

REM Default values
set STAGE=all
set EPOCHS=
set BATCH_SIZE=
set LEARNING_RATE=
set RESUME_FROM=
set DATA_DIR=data/quic360
set WORKERS=%NUM_WORKERS%
set PROJECT=%WANDB_PROJECT%
set RUN_NAME=

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
if "%~1"=="-s" (
    set STAGE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--stage" (
    set STAGE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-e" (
    set EPOCHS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-b" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--batch-size" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-l" (
    set LEARNING_RATE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--lr" (
    set LEARNING_RATE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-r" (
    set RESUME_FROM=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--resume" (
    set RESUME_FROM=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-d" (
    set DATA_DIR=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--data-dir" (
    set DATA_DIR=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-w" (
    set WORKERS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--workers" (
    set WORKERS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-p" (
    set PROJECT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--project" (
    set PROJECT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-n" (
    set RUN_NAME=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--name" (
    set RUN_NAME=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args

:end_parse

REM Set default epochs and batch sizes based on stage
if "%STAGE%"=="vision" (
    if "%EPOCHS%"=="" set EPOCHS=%VISION_EPOCHS%
    if "%BATCH_SIZE%"=="" set BATCH_SIZE=%VISION_BATCH_SIZE%
)
if "%STAGE%"=="resampler" (
    if "%EPOCHS%"=="" set EPOCHS=%RESAMPLER_EPOCHS%
    if "%BATCH_SIZE%"=="" set BATCH_SIZE=%RESAMPLER_BATCH_SIZE%
)
if "%STAGE%"=="finetune" (
    if "%EPOCHS%"=="" set EPOCHS=%FINETUNE_EPOCHS%
    if "%BATCH_SIZE%"=="" set BATCH_SIZE=%FINETUNE_BATCH_SIZE%
)

REM Set data paths
set CSV_TRAIN_PATH=%DATA_DIR%/train.csv
set CSV_VAL_PATH=%DATA_DIR%/valid.csv

echo ==========================================
echo PanoLLaVA Custom Training
echo ==========================================
echo Stage: %STAGE%
echo Epochs: %EPOCHS%
echo Batch Size: %BATCH_SIZE%
echo Learning Rate: %LEARNING_RATE%
echo Resume From: %RESUME_FROM%
echo Data Directory: %DATA_DIR%
echo Workers: %WORKERS%
echo Project: %PROJECT%
echo Run Name: %RUN_NAME%
echo ==========================================

REM Validate data files
if not exist "%CSV_TRAIN_PATH%" (
    echo Error: Training data file not found: %CSV_TRAIN_PATH%
    pause
    exit /b 1
)

if not exist "%CSV_VAL_PATH%" (
    echo Error: Validation data file not found: %CSV_VAL_PATH%
    pause
    exit /b 1
)

REM Build command arguments
set CMD_ARGS=--csv-train "%CSV_TRAIN_PATH%" --csv-val "%CSV_VAL_PATH%" --vision-name "%VISION_MODEL%" --lm-name "%LM_MODEL%" --resampler "%RESAMPLER%" --crop-strategy "%CROP_STRATEGY%" --num-workers "%WORKERS%" --wandb-project "%PROJECT%"

if not "%EPOCHS%"=="" set CMD_ARGS=%CMD_ARGS% --epochs "%EPOCHS%"
if not "%BATCH_SIZE%"=="" set CMD_ARGS=%CMD_ARGS% --batch-size "%BATCH_SIZE%"
if not "%LEARNING_RATE%"=="" set CMD_ARGS=%CMD_ARGS% --lr "%LEARNING_RATE%"
if not "%RESUME_FROM%"=="" set CMD_ARGS=%CMD_ARGS% --resume-from "%RESUME_FROM%"
if not "%RUN_NAME%"=="" set CMD_ARGS=%CMD_ARGS% --wandb-name "%RUN_NAME%"

REM Execute training based on stage
if "%STAGE%"=="all" (
    set CMD_ARGS=%CMD_ARGS% --stages vision resampler finetune
) else (
    set CMD_ARGS=%CMD_ARGS% --stage "%STAGE%"
)

echo Starting training with command:
echo python train.py %CMD_ARGS%
echo.

python train.py %CMD_ARGS%

if %errorlevel% neq 0 (
    echo Error: Training failed
    pause
    exit /b %errorlevel%
)

echo ==========================================
echo Training completed successfully!
echo ==========================================

pause
goto :eof

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   -s, --stage STAGE          Training stage (vision^|resampler^|finetune^|all)
echo   -e, --epochs EPOCHS        Number of epochs (default: auto per stage)
echo   -b, --batch-size SIZE      Batch size (default: auto per stage)
echo   -l, --lr RATE              Learning rate (default: auto per stage)
echo   -r, --resume PATH          Resume from checkpoint
echo   -d, --data-dir DIR         Data directory (default: data/quic360)
echo   -w, --workers NUM          Number of workers (default: %NUM_WORKERS%)
echo   -p, --project NAME         WandB project name (default: %WANDB_PROJECT%)
echo   -n, --name NAME            WandB run name (default: auto-generated)
echo   -h, --help                 Show this help message
echo.
echo Examples:
echo   %~nx0 --stage vision --epochs 5 --batch-size 16
echo   %~nx0 --stage all --data-dir C:\path\to\data
echo   %~nx0 --stage finetune --resume runs\vlm_resampler\checkpoints\best.ckpt
echo.
pause
goto :eof
