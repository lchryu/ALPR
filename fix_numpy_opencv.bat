@echo off
echo ========================================
echo Fixing numpy/opencv compatibility issue
echo ========================================
echo.
echo This will downgrade numpy to 1.x to fix opencv compatibility
echo.

REM Activate conda environment
call conda activate alpr

echo Step 1: Uninstalling old opencv-python-headless...
pip uninstall -y opencv-python-headless

echo.
echo Step 2: Installing numpy 1.26.4...
pip install numpy==1.26.4

echo.
echo Step 3: Reinstalling opencv-python-headless...
pip install opencv-python-headless==4.7.0.72

echo.
echo Step 4: Verifying installation...
python -c "import numpy; print('numpy:', numpy.__version__); import cv2; print('cv2:', cv2.__version__); print('SUCCESS!')"

echo.
echo Done! You can now run: python start_server.py
pause
