# PowerShell script to run API server
$env:PYTHONPATH = "C:\Data\ALPR\src"
conda activate alpr
cd C:\Data\ALPR
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
