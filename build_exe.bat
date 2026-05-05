@echo off
REM ============================================================
REM  Build WaferAligner UI into a single .exe
REM  Run this script from the project root folder
REM ============================================================

echo [Build] Stopping any running WaferAlignerUI instances...
taskkill /IM WaferAlignerUI.exe /F >nul 2>&1

echo [Build] Cleaning previous build...
if exist dist\WaferAlignerUI rmdir /s /q dist\WaferAlignerUI
if exist dist\WaferAlignerUI.zip del /f dist\WaferAlignerUI.zip
if exist build\WaferAligner_UI rmdir /s /q build\WaferAligner_UI

echo [Build] Removing temp cropped template files...
del /f /q temp_cropped_template_*.png 2>nul

echo [Build] Running PyInstaller...
.venv\Scripts\pyinstaller.exe WaferAligner_UI.spec

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed. Check output above.
    pause
    exit /b 1
)

echo.
echo [Build] Zipping output folder...
powershell -Command "Compress-Archive -Path 'dist\WaferAlignerUI' -DestinationPath 'dist\WaferAlignerUI.zip' -Force"
echo.
echo [Build] SUCCESS!
echo Folder : dist\WaferAlignerUI\
echo Archive: dist\WaferAlignerUI.zip
echo.
pause
