"""
Startup script for ALPR API server
Ensures proper environment setup before starting uvicorn
"""
import os
import sys
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Verify cv2 is available first (critical dependency)
try:
    import cv2
    print(f"✓ cv2 imported successfully (version: {cv2.__version__})")
except ImportError as e:
    print(f"✗ ERROR: Cannot import cv2: {e}")
    print(f"Python executable: {sys.executable}")
    print(f"This is usually a numpy/opencv compatibility issue.")
    print(f"Try: pip install --upgrade opencv-python-headless")
    sys.exit(1)

# Verify easyocr is available
try:
    import easyocr
    print(f"✓ easyocr imported successfully")
except ImportError as e:
    print(f"✗ ERROR: Cannot import easyocr: {e}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# Verify other critical imports
try:
    from alpr.pipeline import run_alpr_on_image
    print(f"✓ ALPR pipeline imported successfully")
except ImportError as e:
    print(f"✗ ERROR: Cannot import ALPR pipeline: {e}")
    sys.exit(1)

# Verify uvicorn is available
try:
    import uvicorn
    print(f"✓ uvicorn imported successfully")
except ImportError as e:
    print(f"✗ ERROR: Cannot import uvicorn: {e}")
    print(f"Please install uvicorn: pip install uvicorn[standard]")
    sys.exit(1)

# Now start uvicorn
if __name__ == "__main__":
    print(f"\n🚀 Starting ALPR API server...")
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"🐍 Python: {sys.executable}")
    print(f"🌐 Server will run at: http://127.0.0.1:8000")
    print(f"📚 API docs at: http://127.0.0.1:8000/docs\n")
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(BASE_DIR)]
    )
