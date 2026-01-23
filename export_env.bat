@echo off
echo ========================================
echo Export Conda Environment
echo ========================================
echo.

REM Export conda environment (không bao gồm build numbers để dễ cài đặt trên máy khác)
echo Exporting conda environment...
conda env export --name aplr --no-builds > environment.yml
if %errorlevel% equ 0 (
    echo [OK] Da tao file environment.yml
) else (
    echo [ERROR] Khong the export conda environment
    pause
    exit /b 1
)

REM Export pip requirements (backup)
echo.
echo Exporting pip requirements...
pip freeze > requirements_full.txt
if %errorlevel% equ 0 (
    echo [OK] Da tao file requirements_full.txt
) else (
    echo [WARNING] Khong the export pip requirements
)

echo.
echo ========================================
echo Export hoan tat!
echo ========================================
echo.
echo Cac file da tao:
echo - environment.yml (conda environment)
echo - requirements_full.txt (pip packages - backup)
echo.
pause

