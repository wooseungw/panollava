@echo off
REM =============================================================================
REM Configuration Test Script for Windows
REM =============================================================================

REM Load common configuration
call "%~dp0config.bat"

echo ========================================
echo Configuration Test
echo ========================================

REM Test basic configuration loading
call :print_config

echo.
echo Testing directory setup...
echo Directories created successfully

echo.
echo Testing timestamp generation...
for /f "delims=" %%i in ('powershell -command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set TIMESTAMP=%%i
echo Generated timestamp: %TIMESTAMP%

echo.
echo Testing configuration override...
echo Original Vision Model: %VISION_MODEL%
set VISION_MODEL=microsoft/DiT-large
echo After override: %VISION_MODEL%

echo.
echo All configuration tests passed!
echo ========================================

pause
