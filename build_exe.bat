@echo off
REM ============================================================
REM  Build WaferAligner UI into a single .exe
REM  Run this script from the project root folder
REM ============================================================

echo [Build] Cleaning previous build...
if exist dist\WaferAlignerUI.exe del /f dist\WaferAlignerUI.exe
if exist build\WaferAligner_UI rmdir /s /q build\WaferAligner_UI

echo [Build] Removing temp cropped template files...
del /f /q temp_cropped_template_*.png 2>nul

echo [Build] Running PyInstaller...
pyinstaller WaferAligner_UI.spec

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed. Check output above.
    pause
    exit /b 1
)

echo.
echo [Build] SUCCESS!
echo Output: dist\WaferAlignerUI.exe
echo.
pause
