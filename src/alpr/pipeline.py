"""
ALPR Pipeline: Main Processing Orchestration

Orchestrates all stages of the ALPR pipeline:
1. YOLO detection
2. Plate classification (two-line vs single-line)
3. Deskew
4. Text region cropping
5. OCR (with fallback logic for two-line plates)
"""

import numpy as np
from typing import List, Dict, Optional, Any
from ultralytics import YOLO
from .geometry import is_two_line_plate, split_two_line_plate, deskew_plate, crop_text_region
from .ocr.waterfall import ocr_plate_complete
from .validation import normalize_plate, validate_vn_plate_pattern
from .debug_logger import DebugImageLogger


def run_alpr_on_image(
    img: np.ndarray,
    model: YOLO,
    debug_logger: Optional[DebugImageLogger] = None
) -> Dict[str, Any]:
    """
    Run complete ALPR pipeline on an image.
    
    Processes all detected plates in the image and returns results.
    
    Args:
        img: Input image (BGR format, numpy array)
        model: YOLO model instance for plate detection
        debug_logger: Optional DebugImageLogger for instrumentation
    
    Returns:
        Dictionary with:
        - results: List of detection results, each containing:
          - bbox: [x1, y1, x2, y2]
          - raw: Raw OCR text
          - plate: Normalized plate number
          - det_conf: YOLO detection confidence
          - ocr_conf: OCR confidence
          - method: OCR method used (pass1_clean, pass2_robust, pass3_fallback, none)
          - two_line: Whether plate was classified as two-line
        - debug: Debug info dict (if debug_logger enabled)
    """
    # Instrumentation: log input image
    if debug_logger:
        debug_logger.save("input", img)
    
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
        if debug_logger:
            debug_logger.save("crop_stage1", crop_stage1)
        
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
            logger=debug_logger
        )
        
        print(f"[Stage 3] Deskew completed: "
              f"crop_size_before={crop_stage1.shape[1]}x{crop_stage1.shape[0]}, "
              f"crop_size_after={crop_deskewed.shape[1]}x{crop_deskewed.shape[0]}")
        
        # Instrumentation: log deskewed plate
        if debug_logger:
            debug_logger.save("crop_deskewed", crop_deskewed)
        
        # ========================================================================
        # STAGE 4: Crop text region - Remove padding/whitespace after deskew
        # ========================================================================
        # Use crop_text_region to get tight bounding box around text
        # This removes padding/whitespace added during deskew rotation
        crop_final = crop_text_region(crop_deskewed, margin_ratio=0.05, logger=debug_logger)
        
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
            # Use is_partial_line=True to relax validation for individual lines
            top_raw, top_norm, top_conf, top_method = ocr_plate_complete(
                top, logger=debug_logger, skip_deskew=True, is_partial_line=True
            )
            bot_raw, bot_norm, bot_conf, bot_method = ocr_plate_complete(
                bottom, logger=debug_logger, skip_deskew=True, is_partial_line=True
            )
            
            # Handle None values (OCR may return None if all passes fail)
            top_raw = top_raw if top_raw is not None else ""
            bot_raw = bot_raw if bot_raw is not None else ""
            
            raw = top_raw + bot_raw
            plate = normalize_plate(raw) if raw else ""
            
            # Validate combined result (full plate pattern)
            # If combined result is invalid, try single-line OCR as fallback
            if plate:
                pattern_score = validate_vn_plate_pattern(plate)
                combined_conf = (top_conf + bot_conf) / 2 if top_conf and bot_conf else max(top_conf or 0.0, bot_conf or 0.0)
                
                # If pattern score is too low, might be misclassified - try single-line
                if pattern_score < 0.5 and combined_conf < 0.5:
                    print(f"[Warning] Two-line result has low pattern score ({pattern_score:.2f}), trying single-line fallback...")
                    # Try single-line OCR as fallback
                    fallback_raw, fallback_plate, fallback_conf, fallback_method = ocr_plate_complete(
                        crop_final, logger=debug_logger, skip_deskew=True, is_partial_line=False
                    )
                    if fallback_plate and fallback_conf > combined_conf:
                        fallback_pattern_score = validate_vn_plate_pattern(fallback_plate)
                        if fallback_pattern_score > pattern_score:
                            print(f"[Fallback] Using single-line result: {fallback_plate} (conf={fallback_conf:.2f}, pattern={fallback_pattern_score:.2f})")
                            raw = fallback_raw if fallback_raw else ""
                            plate = fallback_plate
                            ocr_conf = fallback_conf
                            method = f"fallback_{fallback_method}"
                            is_two = False  # Update classification
                        else:
                            # Keep two-line result
                            ocr_conf = combined_conf
                            method = f"{top_method}+{bot_method}"
                    else:
                        # Keep two-line result
                        ocr_conf = combined_conf
                        method = f"{top_method}+{bot_method}"
                else:
                    # Two-line result is valid
                    ocr_conf = combined_conf
                    method = f"{top_method}+{bot_method}"
            else:
                # Both lines failed, try single-line fallback
                print(f"[Warning] Both lines failed OCR, trying single-line fallback...")
                fallback_raw, fallback_plate, fallback_conf, fallback_method = ocr_plate_complete(
                    crop_final, logger=debug_logger, skip_deskew=True, is_partial_line=False
                )
                if fallback_plate:
                    print(f"[Fallback] Using single-line result: {fallback_plate} (conf={fallback_conf:.2f})")
                    raw = fallback_raw if fallback_raw else ""
                    plate = fallback_plate
                    ocr_conf = fallback_conf
                    method = f"fallback_{fallback_method}"
                    is_two = False  # Update classification
                else:
                    ocr_conf = 0.0
                    method = f"{top_method}+{bot_method}"
        
        else:
            # Single-line plate (deskew already applied, skip deskew in ocr_plate_complete)
            raw, plate, ocr_conf, method = ocr_plate_complete(
                crop_final, logger=debug_logger, skip_deskew=True
            )
            
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
    if debug_logger and debug_logger.output_dir and debug_logger.output_dir.exists():
        debug_images = sorted([f.name for f in debug_logger.output_dir.glob("*.jpg")])
        debug_info = {
            "debug_folder": str(debug_logger.output_dir),
            "debug_folder_name": debug_logger.output_dir.name,
            "debug_images_count": len(debug_images),
            "debug_images": debug_images
        }
        response["debug"] = debug_info
    
    return response

