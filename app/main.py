"""
FastAPI Application: Thin Wrapper Around ALPR Pipeline

This module provides the HTTP API interface for the ALPR system.
All processing logic is delegated to the pipeline module.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Add src/ to Python path for imports
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from alpr.pipeline import run_alpr_on_image
from alpr.debug_logger import DebugImageLogger

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho tất cả domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model globally (singleton)
MODEL_PATH = BASE_DIR / "models" / "best.pt"
model = YOLO(str(MODEL_PATH))


@app.post("/alpr")
async def alpr_api(
    file: UploadFile = File(...),
    debug: bool = Query(False, description="Enable debug image logging")
):
    """
    ALPR API endpoint.
    
    Args:
        file: Uploaded image file
        debug: If True, enable image logging for debugging (default: False)
              Usage: POST /alpr?debug=true
    
    Returns:
        JSON response with detection results:
        {
            "results": [
                {
                    "bbox": [x1, y1, x2, y2],
                    "raw": str,           # Raw OCR output (may contain OCR artifacts)
                    "plate": str,         # Normalized plate number (empty string if all passes failed)
                    "det_conf": float,    # YOLO detection confidence
                    "ocr_conf": float,    # OCR confidence (0.0 if all passes failed)
                    "method": str,        # Which OCR pass succeeded:
                                          #   - "pass1_clean": Fast path (high confidence)
                                          #   - "pass2_robust": Moderate difficulty
                                          #   - "pass3_fallback": Last resort (lower confidence)
                                          #   - "none": All passes failed (plate will be empty)
                    "two_line": bool      # Whether plate was classified as two-line
                }
            ],
            "debug": {                    # Only present if debug=true
                "debug_folder": str,
                "debug_folder_name": str,
                "debug_images_count": int,
                "debug_images": List[str]
            }
        }
    
    Design Intent:
    - method="none" + plate="" + ocr_conf=0.0 = Explicit failure (no valid OCR found)
    - Prefer empty result over returning garbage/low-confidence noise
    - method field indicates trust level: pass1 > pass2 > pass3 > none
    """
    # Read uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    
    # Create debug logger if enabled (instrumentation only)
    logger = None
    if debug:
        debug_dir = BASE_DIR / "runs" / "debug"
        logger = DebugImageLogger(enabled=True, root_dir=str(debug_dir))
        print(f"Debug mode enabled. Images will be saved to: {logger.output_dir}")
    
    # Run ALPR pipeline
    response = run_alpr_on_image(img, model, debug_logger=logger)
    
    return response


@app.get("/")
def root():
    """Health check endpoint"""
    return {"message": "ALPR FastAPI is running!"}


@app.get("/debug/status")
def debug_status():
    """Check if debug logger is available"""
    return {
        "debug_logger_available": True,
        "debug_dir": str(BASE_DIR / "runs" / "debug")
    }


@app.get("/debug/images/{folder_name}/{filename}")
async def get_debug_image(folder_name: str, filename: str):
    """
    Serve debug images for frontend preview.
    
    Args:
        folder_name: Debug folder name (e.g., "debug_20250106_123456_789")
        filename: Image filename (e.g., "000_input.jpg")
    
    Returns:
        Image file or 404 if not found
    """
    debug_path = BASE_DIR / "runs" / "debug" / folder_name / filename
    
    # Security: Validate path is within debug directory
    debug_dir = BASE_DIR / "runs" / "debug"
    try:
        debug_path.resolve().relative_to(debug_dir.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path"}, status_code=403)
    
    if debug_path.exists() and debug_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        return FileResponse(str(debug_path))
    
    return JSONResponse({"error": "Image not found"}, status_code=404)

