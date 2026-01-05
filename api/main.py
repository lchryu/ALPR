
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
from ultralytics import YOLO
import sys
import os
from pathlib import Path

# Add src/ to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from utils import (
    ocr_plate_complete,
    is_two_line_plate,
    split_two_line_plate,
    normalize_plate,
    deskew_plate,
    crop_text_region,
    remove_plate_border
)

# Import debug logger (optional instrumentation)
try:
    from debug_logger import DebugImageLogger
except ImportError as e:
    print(f"Warning: Could not import DebugImageLogger: {e}")
    DebugImageLogger = None

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho tất cả domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load YOLO model
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
model = YOLO(MODEL_PATH)


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
        JSON response with detection results
    """
    # Read uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Create debug logger if enabled (instrumentation only)
    logger = None
    debug_info = None
    if debug:
        if DebugImageLogger is None:
            print("Warning: DebugImageLogger not available, debug mode disabled")
        else:
            # Use absolute path from BASE_DIR
            debug_dir = os.path.join(BASE_DIR, "runs", "debug")
            logger = DebugImageLogger(enabled=True, root_dir=debug_dir)
            logger.save("input", img)
            print(f"Debug mode enabled. Images will be saved to: {logger.output_dir}")
            
            # Prepare debug info for response
            if logger.output_dir and logger.output_dir.exists():
                debug_images = sorted([f.name for f in logger.output_dir.glob("*.jpg")])
                debug_info = {
                    "debug_folder": str(logger.output_dir),
                    "debug_folder_name": logger.output_dir.name,
                    "debug_images_count": len(debug_images),
                    "debug_images": debug_images
                }

    # ========================================================================
    # STAGE 1: YOLO Detection Pass #1 - Detect plates on original image
    # ========================================================================
    results = model(img)[0]

    output = []

    for box in results.boxes:
        conf_det = float(box.conf)
        if conf_det < 0.4:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop_stage1 = img[y1:y2, x1:x2]
        
        print(f"[Stage 1] YOLO pass #1: bbox=({x1},{y1},{x2},{y2}), "
              f"crop_size={crop_stage1.shape[1]}x{crop_stage1.shape[0]}, conf={conf_det:.3f}")
        
        # Instrumentation: log initial cropped plate
        if logger:
            logger.save("crop_stage1", crop_stage1)

        # ========================================================================
        # STAGE 2: Plate Classification - Use ORIGINAL crop (before deskew)
        # ========================================================================
        # IMPORTANT: Classify BEFORE deskew to avoid geometry distortion
        # Deskew changes w/h ratio, making classification unreliable
        is_two = is_two_line_plate(crop_stage1)
        
        print(f"[Stage 2] Plate classification: is_two_line={is_two}, "
              f"ratio={crop_stage1.shape[1]/crop_stage1.shape[0]:.2f}")

        # ========================================================================
        # STAGE 3: Deskew - Apply rotation correction ONCE
        # ========================================================================
        crop_deskewed = deskew_plate(
            crop_stage1, 
            angle_threshold=0.8, 
            debug=True, 
            logger=logger
        )
        
        print(f"[Stage 3] Deskew completed: "
              f"crop_size_before={crop_stage1.shape[1]}x{crop_stage1.shape[0]}, "
              f"crop_size_after={crop_deskewed.shape[1]}x{crop_deskewed.shape[0]}")
        
        # Instrumentation: log deskewed plate
        if logger:
            logger.save("crop_deskewed", crop_deskewed)
        
        # ========================================================================
        # STAGE 4: Crop text region - Remove padding/whitespace after deskew
        # ========================================================================
        # Use crop_text_region to get tight bounding box around text
        # This removes padding/whitespace added during deskew rotation
        crop_final = crop_text_region(crop_deskewed, margin_ratio=0.05, logger=logger)
        
        print(f"[Stage 4] Text region cropped: "
              f"crop_size={crop_final.shape[1]}x{crop_final.shape[0]}")

        # ========================================================================
        # STAGE 5: OCR - Process final clean crop with EasyOCR
        # ========================================================================
        # Deskew already applied, skip deskew in OCR to avoid redundant operations
        # Use classification result from Stage 2 (before deskew)
        if is_two:
            # Split into 2 (after deskew and re-crop)
            top, bottom = split_two_line_plate(crop_final)

            # OCR each line separately (deskew already applied, skip deskew in ocr_plate_complete)
            top_raw, top_norm, top_conf, top_method = ocr_plate_complete(top, logger=logger, skip_deskew=True)
            bot_raw, bot_norm, bot_conf, bot_method = ocr_plate_complete(bottom, logger=logger, skip_deskew=True)

            # Handle None values (OCR may return None if all passes fail)
            top_raw = top_raw if top_raw is not None else ""
            bot_raw = bot_raw if bot_raw is not None else ""
            
            raw = top_raw + bot_raw
            plate = normalize_plate(raw) if raw else ""

            ocr_conf = (top_conf + bot_conf) / 2 if top_conf and bot_conf else max(top_conf or 0.0, bot_conf or 0.0)
            method = f"{top_method}+{bot_method}"

        else:
            # Single-line plate (deskew already applied, skip deskew in ocr_plate_complete)
            raw, plate, ocr_conf, method = ocr_plate_complete(crop_final, logger=logger, skip_deskew=True)
            
            # Handle None values (OCR may return None if all passes fail)
            raw = raw if raw is not None else ""
            plate = plate if plate is not None else ""
            ocr_conf = ocr_conf if ocr_conf is not None else 0.0
            method = method if method is not None else "none"

        output.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "raw": raw,
            "plate": plate,
            "det_conf": conf_det,
            "ocr_conf": ocr_conf,
            "method": method,
            "two_line": is_two
        })

    # Prepare response
    response = {"results": output}
    
    # Update debug info after processing (to get final image count)
    if debug and logger and logger.output_dir and logger.output_dir.exists():
        debug_images = sorted([f.name for f in logger.output_dir.glob("*.jpg")])
        debug_info = {
            "debug_folder": str(logger.output_dir),
            "debug_folder_name": logger.output_dir.name,
            "debug_images_count": len(debug_images),
            "debug_images": debug_images
        }
        response["debug"] = debug_info
    
    return response


@app.get("/")
def root():
    return {"message": "ALPR FastAPI is running!"}


@app.get("/debug/status")
def debug_status():
    """Check if debug logger is available"""
    return {
        "debug_logger_available": DebugImageLogger is not None,
        "debug_dir": os.path.join(BASE_DIR, "runs", "debug") if DebugImageLogger else None
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
    debug_path = Path(BASE_DIR) / "runs" / "debug" / folder_name / filename
    
    # Security: Validate path is within debug directory
    debug_dir = Path(BASE_DIR) / "runs" / "debug"
    try:
        debug_path.resolve().relative_to(debug_dir.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path"}, status_code=403)
    
    if debug_path.exists() and debug_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        return FileResponse(str(debug_path))
    
    return JSONResponse({"error": "Image not found"}, status_code=404)
