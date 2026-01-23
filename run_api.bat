@echo off
REM Activate conda environment and run API server
call conda activate alpr
cd /d C:\Data\ALPR
set PYTHONPATH=C:\Data\ALPR\src
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
pause
