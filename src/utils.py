import cv2
import numpy as np
import re
import easyocr

# Create reader 1 lần (tối ưu)
reader = easyocr.Reader(['en'])

# ----------------------------------------------------------
# Remove border/edge contours (plate borders, padding artifacts)
# ----------------------------------------------------------
def remove_border_contours(contours, img_shape, border_margin=0.05, debug=False):
    """
    Remove contours that are likely plate borders or edge artifacts.
    
    Args:
        contours: List of contours
        img_shape: (height, width) of image
        border_margin: Margin from edge to consider as border (default: 5%)
        debug: Print debug info
    
    Returns:
        Filtered list of contours
    """
    h_img, w_img = img_shape
    border_threshold = min(h_img, w_img) * border_margin
    
    filtered = []
    border_contours = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if contour touches or is very close to image edges
        touches_top = y < border_threshold
        touches_bottom = (y + h) > (h_img - border_threshold)
        touches_left = x < border_threshold
        touches_right = (x + w) > (w_img - border_threshold)
        
        # Check if contour spans most of the image (likely a border)
        spans_width = w > w_img * 0.7  # Spans >70% of width
        spans_height = h > h_img * 0.7  # Spans >70% of height
        
        # Check if contour is very large (likely border or merged text)
        area = w * h
        img_area = h_img * w_img
        is_very_large = area > img_area * 0.3  # >30% of image
        
        # Exclude if it's a border contour
        is_border = (
            (touches_top and touches_bottom) or  # Vertical border
            (touches_left and touches_right) or  # Horizontal border
            (spans_width and spans_height) or    # Spans both dimensions
            (is_very_large and (touches_top or touches_bottom or touches_left or touches_right))  # Large and touches edge
        )
        
        if is_border:
            border_contours.append(contour)
            if debug:
                print(f"    Border contour removed: bbox=({x},{y},{w},{h}), "
                      f"area={area/img_area*100:.1f}%, touches_edges=({touches_top},{touches_bottom},{touches_left},{touches_right})")
        else:
            filtered.append(contour)
    
    if debug:
        print(f"  Removed {len(border_contours)} border/edge contours, kept {len(filtered)}")
    
    return filtered

# ----------------------------------------------------------
# Remove plate border (viền biển số)
# ----------------------------------------------------------
def remove_plate_border(img, border_ratio=0.08, logger=None):
    """
    Remove plate border by cropping inner region.
    Border của biển số có thể ảnh hưởng đến OCR:
    - Làm Otsu threshold sai
    - EasyOCR detect border như text
    - Border merge với characters
    
    Args:
        img: Input plate image (BGR or grayscale)
        border_ratio: Ratio of border to remove (default: 0.08 = 8%)
        logger: Optional DebugImageLogger for instrumentation (default: None)
    
    Returns:
        Image with border removed (cropped inner region)
    """
    h, w = img.shape[:2]
    
    # Calculate border thickness
    border_h = int(h * border_ratio)
    border_w = int(w * border_ratio)
    
    # Validate: don't remove too much (at least keep 50% of image)
    if border_h * 2 >= h or border_w * 2 >= w:
        # Border too thick, return original
        if logger:
            logger.save("border_removed", img)
        return img
    
    # Crop inner region (remove border)
    cropped = img[border_h:h-border_h, border_w:w-border_w]
    
    # Validate crop is not empty
    if cropped.size == 0:
        if logger:
            logger.save("border_removed", img)
        return img
    
    # Instrumentation: log border removed image
    if logger:
        logger.save("border_removed", cropped)
    
    return cropped

# ----------------------------------------------------------
# Detect if plate is 2-line or 1-line based on aspect ratio
# ----------------------------------------------------------
def is_two_line_plate(crop):
    h, w = crop.shape[:2]
    ratio = w / h
    return ratio < 3.2

# ----------------------------------------------------------
# Crop text region after deskew (remove padding/whitespace)
# ----------------------------------------------------------
def crop_text_region(img, margin_ratio=0.05, logger=None):
    """
    Detect and crop text region from deskewed image to remove padding/whitespace.
    
    Args:
        img: Input image (BGR or grayscale) - already deskewed
        margin_ratio: Margin ratio to add around detected text region (default: 0.05 = 5%)
        logger: Optional DebugImageLogger for instrumentation (default: None)
    
    Returns:
        Cropped image containing only text region
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape[:2]
    
    # Create binary image to find text regions
    _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours for both
    contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use the binary with more contours
    if len(contours1) >= len(contours2):
        contours = contours1
    else:
        contours = contours2
    
    if not contours:
        # No contours found, return original
        if logger:
            logger.save("crop_text_region", img)
        return img
    
    # Find bounding box of all text contours
    img_area = h * w
    text_bboxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 30:  # Reduced threshold for better detection
            continue
        
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Filter by size
        area_ratio = area / img_area
        if area_ratio < 0.0005 or area_ratio > 0.4:  # More lenient
            continue
        
        # Filter by aspect ratio - characters are typically not extremely wide/tall
        aspect_ratio = ch / cw if cw > 0 else 0
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:  # More lenient
            continue
        
        # Filter by height - characters should be reasonable height
        height_ratio = ch / h
        if height_ratio < 0.1 or height_ratio > 0.9:  # More lenient
            continue
        
        # Filter out contours touching borders (likely padding/artifacts)
        margin = min(w, h) * 0.03  # 3% margin
        touches_border = (x < margin or y < margin or 
                         (x + cw) > (w - margin) or (y + ch) > (h - margin))
        
        # If touches border AND is large, likely padding (skip)
        if touches_border and area_ratio > 0.15:
            continue
        
        text_bboxes.append((x, y, cw, ch))
    
    if not text_bboxes:
        # No valid text regions found, return original
        if logger:
            logger.save("crop_text_region", img)
        return img
    
    # Find union bounding box of all text regions
    min_x = min(bbox[0] for bbox in text_bboxes)
    min_y = min(bbox[1] for bbox in text_bboxes)
    max_x = max(bbox[0] + bbox[2] for bbox in text_bboxes)
    max_y = max(bbox[1] + bbox[3] for bbox in text_bboxes)
    
    # Add margin
    margin_x = int((max_x - min_x) * margin_ratio)
    margin_y = int((max_y - min_y) * margin_ratio)
    
    x1 = max(0, min_x - margin_x)
    y1 = max(0, min_y - margin_y)
    x2 = min(w, max_x + margin_x)
    y2 = min(h, max_y + margin_y)
    
    # Crop image
    cropped = img[y1:y2, x1:x2]
    
    # Validate crop is not empty
    if cropped.size == 0:
        if logger:
            logger.save("crop_text_region", img)
        return img
    
    # Instrumentation: log cropped text region
    if logger:
        logger.save("crop_text_region", cropped)
    
    return cropped

# ----------------------------------------------------------
# Split 2-line motorcycle plate
# ----------------------------------------------------------
def split_two_line_plate(crop):
    h, w = crop.shape[:2]
    mid = h // 2
    return crop[0:mid, :], crop[mid:h, :]

# ----------------------------------------------------------
# Deskew / Rotation Correction
# ----------------------------------------------------------
def deskew_plate(img, angle_threshold=0.8, debug=False, logger=None, return_angle=False):
    """
    Detect and correct rotation/skew in license plate image.
    
    Improved algorithm that reliably detects small rotations (0.7-3.0°) by:
    1. Filtering contours to find character-like regions (not plate borders)
    2. Using multiple character contours for robust angle estimation
    3. Primary method: Projection profile (most accurate for text skew)
    4. Fallback: minAreaRect on filtered character contours
    
    Args:
        img: Input image (BGR or grayscale)
        angle_threshold: Minimum angle (degrees) to trigger correction (default: 0.8)
        debug: If True, print detected angle (default: False)
        logger: Optional DebugImageLogger for instrumentation (default: None)
        return_angle: If True, return tuple (corrected_image, detected_angle) (default: False)
    
    Returns:
        Corrected image (same format as input), or tuple (image, angle) if return_angle=True
        Angle is in degrees, 0.0 if no correction was applied
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Skip deskew for very small images (not enough data for reliable angle detection)
    h, w = gray.shape[:2]
    if min(h, w) < 30:
        if debug:
            print(f"  Deskew: Image too small ({w}x{h}), skipping")
        if logger:
            logger.save("deskew", img)
        return (img, 0.0) if return_angle else img
    
    # Create binary image for contour detection
    # Try both normal and inverted thresholds to handle different plate styles
    _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours for both
    contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use the binary with more contours (likely has better text detection)
    if len(contours1) >= len(contours2):
        binary = binary1
        contours = contours1
    else:
        binary = binary2
        contours = contours2
    
    if not contours:
        if logger:
            logger.save("deskew", img)
        return img  # No contours found, return original
    
    # ========================================================================
    # METHOD 1: Filter contours to find character-like regions
    # ========================================================================
    # Filter criteria for character-like contours:
    # - Reasonable size (not too small, not too large)
    # - Character-like aspect ratio (not too wide/tall)
    # - Not touching borders (likely not plate border)
    # - Multiple contours for robust angle estimation
    
    img_area = h * w
    character_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:  # Too small, likely noise
            continue
        
        # Get bounding box
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Filter by size: character should be reasonable portion of image
        area_ratio = area / img_area
        if area_ratio < 0.001 or area_ratio > 0.3:  # Too small or too large
            continue
        
        # Filter by aspect ratio: characters are typically not extremely wide/tall
        aspect_ratio = ch / cw if cw > 0 else 0
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Too wide or too tall
            continue
        
        # Filter by position: characters shouldn't touch borders (plate borders do)
        margin = min(w, h) * 0.05  # 5% margin
        touches_border = (x < margin or y < margin or 
                         (x + cw) > (w - margin) or (y + ch) > (h - margin))
        
        # If contour touches border AND is large, likely plate border (skip)
        if touches_border and area_ratio > 0.1:
            continue
        
        # Filter by height: characters should be reasonable height relative to image
        height_ratio = ch / h
        if height_ratio < 0.15 or height_ratio > 0.85:  # Too small or too large
            continue
        
        character_contours.append(contour)
    
    # ========================================================================
    # METHOD 2: Estimate angle using minAreaRect on character contours
    # ========================================================================
    angles_minArea = []
    
    if character_contours:
        # Use top character contours (sorted by area)
        sorted_char_contours = sorted(character_contours, key=cv2.contourArea, reverse=True)
        top_char_contours = sorted_char_contours[:min(10, len(sorted_char_contours))]
        
        for contour in top_char_contours:
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            
            # Normalize angle to [-45, 45] range
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
            angles_minArea.append(angle)
    
    # Use median of angles (robust to outliers)
    angle_minArea = np.median(angles_minArea) if angles_minArea else 0.0
    
    # ========================================================================
    # METHOD 3: Projection profile method (PRIMARY - most accurate for text)
    # ========================================================================
    # This method rotates the binary image and finds the angle that maximizes
    # horizontal projection variance (text lines should align horizontally)
    
    # Reuse binary image from contour detection (optimization)
    # If binary1 was selected, use it; otherwise create new one for projection
    if 'binary' in locals() and binary is not None:
        binary_proj = binary
    else:
        _, binary_proj = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Optimized: Coarse-to-fine search instead of exhaustive scan
    # Step 1: Coarse search (±10° with 1° step) - 20 iterations
    angles_coarse = np.arange(-10, 10, 1.0)
    best_angle_coarse = 0
    best_score_coarse = 0
    
    center = (w // 2, h // 2)
    for test_angle in angles_coarse:
        M = cv2.getRotationMatrix2D(center, test_angle, 1.0)
        rotated = cv2.warpAffine(binary_proj, M, (w, h), 
                               flags=cv2.INTER_LINEAR,  # Faster interpolation for coarse search
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=0)
        h_projection = np.sum(rotated, axis=1)
        score = np.var(h_projection)
        if score > best_score_coarse:
            best_score_coarse = score
            best_angle_coarse = test_angle
    
    # Step 2: Fine search around best coarse angle (±1.5° with 0.15° step) - 20 iterations
    # Total: 40 iterations instead of 133 (3x faster)
    angles_fine = np.arange(best_angle_coarse - 1.5, best_angle_coarse + 1.5, 0.15)
    best_angle = best_angle_coarse
    best_score = best_score_coarse
    
    for test_angle in angles_fine:
        M = cv2.getRotationMatrix2D(center, test_angle, 1.0)
        rotated = cv2.warpAffine(binary_proj, M, (w, h), 
                               flags=cv2.INTER_CUBIC,  # Higher quality for fine search
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=0)
        h_projection = np.sum(rotated, axis=1)
        score = np.var(h_projection)
        if score > best_score:
            best_score = score
            best_angle = test_angle
    
    # ========================================================================
    # COMBINE METHODS: Prefer projection profile (most accurate)
    # ========================================================================
    # Projection profile is the primary method because:
    # - It directly measures text alignment
    # - Works well for small angles (0.5-3°)
    # - Not affected by individual contour noise
    
    # Use projection profile if it detects significant angle
    # Otherwise fallback to minAreaRect (if we have character contours)
    if abs(best_angle) >= 0.3:  # Projection detected meaningful angle
        angle = best_angle
        method_used = "projection_profile"
    elif angles_minArea:  # Fallback to minAreaRect on characters
        angle = angle_minArea
        method_used = "minAreaRect_characters"
    else:
        angle = 0.0
        method_used = "none"
    
    if debug:
        num_chars = len(character_contours)
        print(f"  Deskew: found {num_chars} character contours, "
              f"minAreaRect={angle_minArea:.2f}°, projection={best_angle:.2f}°, "
              f"using {method_used}")
    
    # Only correct if angle exceeds threshold
    if abs(angle) < angle_threshold:
        if debug:
            print(f"  Deskew: Angle {angle:.2f}° < threshold {angle_threshold}°, skipping correction")
        if logger:
            logger.save("deskew", img)
            print(f"[Debug] Deskew logged: angle={angle:.2f}° < threshold, no correction needed")
        return img  # Plate is already horizontal enough
    
    if debug:
        print(f"  Deskew: Detected angle {angle:.2f}° ({method_used}), applying correction")
    
    # Apply rotation correction
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation
    if len(img.shape) == 3:
        corrected = cv2.warpAffine(img, M, (new_w, new_h), 
                                   flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(255, 255, 255))
    else:
        corrected = cv2.warpAffine(img, M, (new_w, new_h), 
                                   flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=255)
    
    # Instrumentation: log deskewed/corrected image
    if logger:
        logger.save("deskew", corrected)
        print(f"[Debug] Deskew logged: angle={angle:.2f}°, corrected image saved")
    
    return corrected

# ----------------------------------------------------------
# Preprocess plate - Optimized for EasyOCR
# ----------------------------------------------------------
def preprocess_plate(img, variant="standard", apply_deskew=True, logger=None):
    """
    Clean preprocessing pipeline optimized for EasyOCR.
    Focuses on clarity without destroying texture.
    
    Args:
        img: Input image (BGR or grayscale)
        variant: "standard", "high_contrast", "sharp", "clean"
        apply_deskew: Whether to apply rotation correction (default: True)
        logger: Optional DebugImageLogger for instrumentation (default: None)
    """
    # 0. Deskew/rotation correction (applied first, before upscaling)
    if apply_deskew:
        img = deskew_plate(img, angle_threshold=0.8, debug=True, logger=logger)
    
    h, w = img.shape[:2]
    
    # 1. Upscale intelligently - larger scale for better OCR (especially for letter/number distinction)
    if min(h, w) < 50:
        scale = 5.0  # Very high scale for tiny images
    elif min(h, w) < 100:
        scale = 4.5  # High scale for small images
    else:
        scale = 4.0  # Higher scale for better character clarity
    
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Instrumentation: log grayscale
    if logger:
        logger.save("gray", gray)
    
    # 3. Add padding to avoid edge artifacts
    # More padding at the end (right side) to ensure last characters are not cut off
    padding_top = 20
    padding_bottom = 20
    padding_left = 20
    padding_right = 40  # Extra padding on right side for last characters
    gray = cv2.copyMakeBorder(gray, padding_top, padding_bottom, padding_left, padding_right, 
                             cv2.BORDER_CONSTANT, value=255)
    
    # 4. Denoising based on variant
    # Reduced denoising to preserve thin characters like '1'
    if variant == "clean":
        # Moderate denoising (reduced from aggressive)
        if min(gray.shape) > 100:
            gray = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
        else:
            gray = cv2.bilateralFilter(gray, 5, 60, 60)
    else:
        # Very gentle denoising to preserve thin characters
        if min(gray.shape) > 100:
            gray = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7, searchWindowSize=21)
        else:
            gray = cv2.bilateralFilter(gray, 3, 40, 40)
    
    # 5. CLAHE for adaptive contrast
    if variant == "high_contrast":
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    else:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Instrumentation: log contrast enhanced
    if logger:
        logger.save("contrast", enhanced)
    
    # 6. Sharpening based on variant
    # Reduced sharpening to preserve thin characters like '1'
    if variant == "sharp":
        # Moderate sharpening (reduced from stronger to preserve details)
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 4.5, -0.5],
            [0, -0.5, 0]
        ])
    else:
        # Light sharpening to preserve thin characters
        kernel = np.array([
            [0, -0.3, 0],
            [-0.3, 3.2, -0.3],
            [0, -0.3, 0]
        ])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    
    # 7. Final contrast enhancement - make text darker and background brighter
    # This helps EasyOCR distinguish characters more clearly
    final = np.clip(sharp, 0, 255).astype(np.uint8)
    
    # Apply threshold to create strong black text on white background
    # Use Otsu's method to automatically find optimal threshold
    _, binary = cv2.threshold(final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Instrumentation: log threshold
    if logger:
        logger.save("threshold", binary)
    
    # IMPORTANT: Add spacing between characters to help EasyOCR detect them separately
    # Use morphological opening with horizontal kernel to separate characters
    # This creates small gaps between characters without breaking them
    kernel_separate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))  # Horizontal kernel
    binary_separated = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_separate, iterations=1)
    
    # Blend: 70% separated binary + 30% original binary
    # Separated helps character detection, original preserves shape
    binary_final = cv2.addWeighted(binary_separated, 0.7, binary, 0.3, 0)
    
    # Blend: 80% binary (strong contrast) + 20% original (preserve some detail)
    # This creates very dark text on very bright background
    final = cv2.addWeighted(binary_final, 0.8, final, 0.2, 0)
    
    # Final normalization
    final = np.clip(final, 0, 255).astype(np.uint8)
    
    # Instrumentation: log final preprocessed image
    if logger:
        logger.save("preprocessed_final", final)
    
    return final

# ----------------------------------------------------------
# 3-PASS WATERFALL OCR PIPELINE
# ----------------------------------------------------------

def _ocr_pass_1_clean(img, logger=None):
    """
    PASS 1 - CLEAN PASS
    Minimal preprocessing for clear, straight plates with good lighting.
    Goal: Fast, high-precision for easy cases.
    
    Args:
        img: Input cropped plate image (ALREADY DESKEWED by ocr_plate())
        logger: Optional DebugImageLogger for instrumentation (default: None)
    
    Returns: (text, confidence) or (None, 0.0) if no result
    
    Note: DO NOT call deskew_plate() here - deskew is global preprocessing.
    """
    # Light preprocessing: minimal enhancement
    # IMPORTANT: img is already deskewed by ocr_plate() before this function is called
    # DO NOT call deskew_plate() here - deskew is global preprocessing, not pass-specific
    img_deskewed = img  # Already deskewed
    h, w = img_deskewed.shape[:2]
    
    # Moderate upscale
    scale = 3.5 if min(h, w) < 100 else 3.0
    img_scaled = cv2.resize(img_deskewed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    if len(img_scaled.shape) == 3:
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_scaled.copy()
    
    # Light padding
    padding = 15
    gray = cv2.copyMakeBorder(gray, padding, padding, padding, padding, 
                              cv2.BORDER_CONSTANT, value=255)
    
    # Light CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Mild threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Instrumentation: log OCR input image
    if logger:
        logger.save("pass1_input", binary)
    
    # OCR with standard parameters (optimized for single-line text)
    ocr_params = {
        'detail': 1,
        'paragraph': False,
        'width_ths': 0.6,  # Standard threshold
        'height_ths': 0.6,
        'slope_ths': 0.1,
        'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    }
    
    try:
        results = reader.readtext(binary, **ocr_params)
    except Exception as e:
        # Retry once on failure
        try:
            results = reader.readtext(binary, **ocr_params)
        except Exception:
            if logger:
                print(f"  OCR Pass 1 failed: {e}")
            return None, 0.0
    
    if not results:
        return None, 0.0
    
    # Sort results by x-coordinate (left to right) to ensure correct order
    results_sorted = sorted(results, key=lambda r: r[0][0][0])  # Sort by leftmost x-coordinate
    text = "".join([r[1] for r in results_sorted])
    confidence = np.mean([r[2] for r in results_sorted])
    
    return text, confidence


def _ocr_pass_2_robust(img, logger=None):
    """
    PASS 2 - ROBUST PASS
    Stronger preprocessing for slight blur, rotation, or uneven lighting.
    Goal: High recall for moderately difficult cases.
    
    Args:
        img: Input cropped plate image (ALREADY DESKEWED by ocr_plate())
        logger: Optional DebugImageLogger for instrumentation (default: None)
    
    Returns: (text, confidence) or (None, 0.0) if no result
    
    Note: DO NOT call deskew_plate() here - deskew is global preprocessing.
    """
    # Apply preprocessing
    # IMPORTANT: img is already deskewed by ocr_plate() before this function is called
    # DO NOT call deskew_plate() here - deskew is global preprocessing, not pass-specific
    img_deskewed = img  # Already deskewed
    h, w = img_deskewed.shape[:2]
    
    # Higher upscale for better clarity
    scale = 4.5 if min(h, w) < 100 else 4.0
    img_scaled = cv2.resize(img_deskewed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    if len(img_scaled.shape) == 3:
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_scaled.copy()
    
    # Add padding
    padding = 20
    gray = cv2.copyMakeBorder(gray, padding, padding, padding, padding, 
                              cv2.BORDER_CONSTANT, value=255)
    
    # Denoising
    if min(gray.shape) > 100:
        gray = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7, searchWindowSize=21)
    else:
        gray = cv2.bilateralFilter(gray, 3, 40, 40)
    
    # Stronger CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Light sharpening
    kernel = np.array([
        [0, -0.3, 0],
        [-0.3, 3.2, -0.3],
        [0, -0.3, 0]
    ])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    
    # Adaptive threshold (more robust to lighting variations)
    adaptive = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )
    
    # Instrumentation: log OCR input image
    if logger:
        logger.save("pass2_input", adaptive)
    
    # OCR with more sensitive parameters
    ocr_params = {
        'detail': 1,
        'paragraph': False,
        'width_ths': 0.4,  # More sensitive
        'height_ths': 0.4,
        'slope_ths': 0.1,
        'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    }
    
    try:
        results = reader.readtext(adaptive, **ocr_params)
    except Exception as e:
        # Retry once on failure
        try:
            results = reader.readtext(adaptive, **ocr_params)
        except Exception:
            if logger:
                print(f"  OCR Pass 2 failed: {e}")
            return None, 0.0
    
    if not results:
        return None, 0.0
    
    # Sort results by x-coordinate (left to right) to ensure correct order
    results_sorted = sorted(results, key=lambda r: r[0][0][0])  # Sort by leftmost x-coordinate
    text = "".join([r[1] for r in results_sorted])
    confidence = np.mean([r[2] for r in results_sorted])
    
    return text, confidence


def _ocr_pass_3_fallback(img, logger=None):
    """
    PASS 3 - FALLBACK PASS
    Aggressive preprocessing for very hard cases.
    Tries both normal and inverted images, chooses the better result.
    May produce noisy results - MUST go through normalization + pattern correction.
    Goal: Salvage attempt for difficult images.
    
    Args:
        img: Input cropped plate image (ALREADY DESKEWED by ocr_plate())
        logger: Optional DebugImageLogger for instrumentation (default: None)
    
    Returns: (text, confidence) or (None, 0.0) if no result
    
    Note: DO NOT call deskew_plate() here - deskew is global preprocessing.
    """
    # Apply preprocessing
    # IMPORTANT: img is already deskewed by ocr_plate() before this function is called
    # DO NOT call deskew_plate() here - deskew is global preprocessing, not pass-specific
    img_deskewed = img  # Already deskewed
    h, w = img_deskewed.shape[:2]
    
    # Very high upscale
    scale = 6.0 if min(h, w) < 100 else 5.0
    img_scaled = cv2.resize(img_deskewed, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to grayscale
    if len(img_scaled.shape) == 3:
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_scaled.copy()
    
    # Extra padding
    padding = 30
    gray = cv2.copyMakeBorder(gray, padding, padding, padding, padding, 
                              cv2.BORDER_CONSTANT, value=255)
    
    # Strong denoising
    if min(gray.shape) > 100:
        gray = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
    else:
        gray = cv2.bilateralFilter(gray, 5, 60, 60)
    
    # Maximum CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Strong sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 6, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    
    # Try both normal and inverted
    # Normal threshold
    _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR with very sensitive parameters
    ocr_params = {
        'detail': 1,
        'paragraph': False,
        'width_ths': 0.3,  # Very sensitive
        'height_ths': 0.3,
        'slope_ths': 0.1,
        'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    }
    
    # Try both normal and inverted, choose the better one
    inverted = cv2.bitwise_not(binary)
    
    # Instrumentation: log OCR input images
    if logger:
        logger.save("pass3_input", binary)
        logger.save("pass3_input_inverted", inverted)
    
    # Try normal
    results_normal = []
    conf_normal = 0.0
    try:
        results_normal = reader.readtext(binary, **ocr_params)
        if results_normal:
            results_normal_sorted = sorted(results_normal, key=lambda r: r[0][0][0])
            conf_normal = np.mean([r[2] for r in results_normal_sorted])
    except Exception as e:
        if logger:
            print(f"  OCR Pass 3 (normal) failed: {e}")
    
    # Try inverted
    results_inverted = []
    conf_inverted = 0.0
    try:
        results_inverted = reader.readtext(inverted, **ocr_params)
        if results_inverted:
            results_inverted_sorted = sorted(results_inverted, key=lambda r: r[0][0][0])
            conf_inverted = np.mean([r[2] for r in results_inverted_sorted])
    except Exception as e:
        if logger:
            print(f"  OCR Pass 3 (inverted) failed: {e}")
    
    # Choose the better result (higher confidence)
    if conf_normal >= conf_inverted and results_normal:
        results = sorted(results_normal, key=lambda r: r[0][0][0])
    elif results_inverted:
        results = sorted(results_inverted, key=lambda r: r[0][0][0])
    else:
        return None, 0.0
    
    text = "".join([r[1] for r in results])
    confidence = np.mean([r[2] for r in results])
    
    return text, confidence


def _is_valid_result(text, confidence, min_confidence=0.5, min_pattern_score=0.7):
    """
    Validate OCR result based on confidence and pattern matching.
    
    Args:
        text: Raw OCR text
        confidence: OCR confidence score (0.0-1.0)
        min_confidence: Minimum confidence threshold
        min_pattern_score: Minimum pattern validation score
    
    Returns:
        bool: True if result is valid, False otherwise
    """
    # Edge case: Check text validity
    if not text:
        return False
    
    # Edge case: Check text length (VN plates are typically 7-10 chars after normalization)
    clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(clean_text) < 5 or len(clean_text) > 15:
        return False
    
    # Edge case: Check confidence validity
    if confidence <= 0.0 or np.isnan(confidence) or np.isinf(confidence):
        return False
    
    if confidence < min_confidence:
        return False
    
    # Edge case: Check if text has both letters and numbers (VN plates have both)
    has_letter = any(c.isalpha() for c in clean_text)
    has_number = any(c.isdigit() for c in clean_text)
    if not (has_letter and has_number):
        return False
    
    pattern_score = validate_vn_plate_pattern(text)
    if pattern_score < min_pattern_score:
        return False
    
    return True


def ocr_plate(img, use_multi_pass=True, return_all_attempts=False, logger=None, skip_deskew=False):
    """
    3-PASS WATERFALL OCR PIPELINE
    
    Waterfall logic:
    1. Run Pass 1 (clean) → if valid → STOP
    2. Run Pass 2 (robust) → if valid → STOP
    3. Run Pass 3 (fallback) → validate with lower thresholds → return if valid
    
    Args:
        img: Input cropped license plate image
        use_multi_pass: If False, only run Pass 1
        return_all_attempts: If True, return all attempts (for debugging)
        logger: Optional DebugImageLogger for instrumentation (default: None)
        skip_deskew: If True, skip deskew (image already deskewed at higher level)
    
    Returns:
        (text, confidence, method_used) or (text, confidence, method, all_attempts) if return_all_attempts=True
    """
    # Deskew ONCE before OCR passes (unless already deskewed at higher level)
    if skip_deskew:
        img_deskewed = img  # Already deskewed
    else:
        # Deskew ONCE (global preprocessing, not pass-specific)
        # This ensures:
        # 1. Consistent deskew across all passes
        # 2. Debug logger logs the exact image used for OCR
        # 3. No redundant deskew operations
        img_deskewed = deskew_plate(img, angle_threshold=0.8, debug=True, logger=logger)
    
    # IMPORTANT: Remove plate border BEFORE OCR passes
    # Border có thể ảnh hưởng đến:
    # - Otsu thresholding (border đen làm threshold sai)
    # - EasyOCR detection (border được detect như text)
    # - Character segmentation (border merge với characters)
    img_no_border = remove_plate_border(img_deskewed, border_ratio=0.08, logger=logger)
    
    if not use_multi_pass:
        # Single pass mode - only Pass 1
        text, confidence = _ocr_pass_1_clean(img_no_border, logger=logger)
        if text:
            return text, confidence, "pass1_clean"
        return "", 0.0, "pass1_clean"
    
    # Waterfall: Pass 1 → Pass 2 → Pass 3
    all_attempts = []
    
    # PASS 1: Clean pass (use image without border)
    text, confidence = _ocr_pass_1_clean(img_no_border, logger=logger)
    if text:
        normalized = normalize_plate(text)
        all_attempts.append((text, confidence, "pass1_clean"))
        
        # Validate using normalized text, but return raw text
        if _is_valid_result(normalized, confidence, min_confidence=0.6, min_pattern_score=0.8):
            if return_all_attempts:
                return text, confidence, "pass1_clean", all_attempts
            return text, confidence, "pass1_clean"
    
    # PASS 2: Robust pass (use image without border)
    text, confidence = _ocr_pass_2_robust(img_no_border, logger=logger)
    if text:
        normalized = normalize_plate(text)
        all_attempts.append((text, confidence, "pass2_robust"))
        
        # Validate using normalized text, but return raw text
        if _is_valid_result(normalized, confidence, min_confidence=0.5, min_pattern_score=0.7):
            if return_all_attempts:
                return text, confidence, "pass2_robust", all_attempts
            return text, confidence, "pass2_robust"
    
    # PASS 3: Fallback pass (use image without border, validate with lower thresholds)
    text, confidence = _ocr_pass_3_fallback(img_no_border, logger=logger)
    if text:
        normalized = normalize_plate(text)
        all_attempts.append((text, confidence, "pass3_fallback"))
        
        # Validate with lower thresholds but still validate (don't return garbage)
        if _is_valid_result(normalized, confidence, min_confidence=0.3, min_pattern_score=0.5):
            if return_all_attempts:
                return text, confidence, "pass3_fallback", all_attempts
            return text, confidence, "pass3_fallback"
        # If Pass 3 result is too bad, don't return it
        if logger:
            print(f"  Pass 3 result rejected: text='{text}', normalized='{normalized}', conf={confidence:.3f}")
    
    # All passes failed
    if return_all_attempts:
        return "", 0.0, "none", all_attempts
    return "", 0.0, "none"

# ----------------------------------------------------------
# Validate Vietnamese plate pattern
# ----------------------------------------------------------
def validate_vn_plate_pattern(text):
    """
    Validate if text matches Vietnamese plate pattern: XXY-XXXXX
    - Position 3 (index 2) should be a LETTER
    - Length should be reasonable (7-9 characters after removing special chars)
    Returns: score from 0.0 to 1.0
    """
    if not text or len(text) < 3:
        return 0.0
    
    # Remove special characters for validation
    clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())
    
    if len(clean_text) < 7 or len(clean_text) > 10:
        return 0.0  # Invalid length
    
    # Check if position 3 is a letter (most important)
    if len(clean_text) > 2:
        if clean_text[2].isalpha():
            return 1.0  # Perfect pattern match
        else:
            return 0.3  # Position 3 is number (common mistake)
    
    return 0.5  # Neutral score

# ----------------------------------------------------------
# Post-process OCR result based on Vietnamese plate patterns
# ----------------------------------------------------------
def post_process_vn_plate(text):
    """
    Post-process OCR result using Vietnamese license plate patterns.
    VN plate format: XXY-XXXXX (e.g., 51G-316.91, 60A-359.81)
    - Position 3 (index 2) is typically a LETTER (A-Z)
    - Other positions are typically NUMBERS (0-9)
    """
    if not text or len(text) < 3:
        return text
    
    # Convert to list for easier manipulation
    chars = list(text.upper())
    
    # Fix position 3 (index 2) - should be a letter, not a number
    # Common mistakes: 6 -> G, 4 -> A, 0 -> O, 1 -> I
    if len(chars) > 2 and chars[2].isdigit():
        # Common OCR mistakes at position 3 (number misread as letter)
        fixes = {
            '6': 'G',  # 6 is often misread as G (most common)
            '4': 'A',  # 4 is often misread as A (common in VN plates like 60A)
            '0': 'A',  # 0 is sometimes misread as A (e.g., 60A -> 600)
            '1': 'I',  # 1 is often misread as I
            '5': 'S',  # 5 is sometimes misread as S
            '8': 'B',  # 8 is sometimes misread as B
        }
        if chars[2] in fixes:
            chars[2] = fixes[chars[2]]
    
    return ''.join(chars)

# ----------------------------------------------------------
# Normalize Vietnamese license plate
# ----------------------------------------------------------
def normalize_plate(text):
    """
    Normalize Vietnamese license plate text.
    Removes special characters and fixes common OCR mistakes.
    """
    if not text:
        return ""
    
    # Convert to uppercase
    text = text.upper()
    
    # Remove all special characters using regex (keep only A-Z and 0-9)
    text = re.sub(r"[^A-Z0-9]", "", text)
    
    # Post-process based on VN plate patterns FIRST (before other replacements)
    # This fixes position 3 (index 2) which should be a letter
    text = post_process_vn_plate(text)
    
    # Common OCR mistakes for Vietnamese plates
    # Note: G is a valid character in VN plates (e.g., 51G-316.91)
    # Only replace characters that are clearly mistakes (but NOT at position 3)
    replacements = {
        "O": "0",  # Letter O -> Number 0 (common mistake, but not at pos 3)
        "I": "1",  # Letter I -> Number 1 (common mistake, but not at pos 3)
        "Z": "2",  # Letter Z -> Number 2 (common mistake)
        "S": "5",  # Letter S -> Number 5 (common mistake)
        "B": "8",  # Letter B -> Number 8 (common mistake)
        # Don't replace G, D as they can be valid in VN plates
    }
    
    # Apply replacements with context awareness
    # Only replace O→0, I→1 when they're between numbers (context: number-O-number or number-I-number)
    result = []
    for i, char in enumerate(text):
        if i == 2 and char.isalpha():
            # Position 3: ALWAYS keep as letter (don't replace)
            result.append(char)
        elif char in replacements:
            # Check context: only replace if surrounded by numbers or at start/end
            prev_char = text[i-1] if i > 0 else None
            next_char = text[i+1] if i < len(text)-1 else None
            
            # Replace O→0, I→1 only if:
            # 1. At position 0-1 (first 2 chars are numbers)
            # 2. Between numbers (prev and next are digits)
            # 3. At end if prev is digit
            should_replace = False
            if i < 2:
                # First 2 positions: replace if next char is digit
                should_replace = next_char and next_char.isdigit()
            elif i >= len(text) - 2:
                # Last 2 positions: replace if prev char is digit
                should_replace = prev_char and prev_char.isdigit()
            else:
                # Middle positions: replace if both prev and next are digits
                should_replace = (prev_char and prev_char.isdigit() and 
                                next_char and next_char.isdigit())
            
            # For Z→2, S→5, B→8: always replace (less ambiguous)
            if char in ["Z", "S", "B"]:
                should_replace = True
            
            if should_replace:
                result.append(replacements[char])
            else:
                result.append(char)
        else:
            result.append(char)
    
    return ''.join(result)

# ----------------------------------------------------------
# Complete OCR pipeline for plate
# ----------------------------------------------------------
def ocr_plate_complete(img, use_multi_pass=True, return_all_attempts=False, logger=None, skip_deskew=False):
    """
    Complete OCR pipeline: preprocess -> OCR -> normalize
    Returns: (raw_text, normalized_text, confidence, method)
    If return_all_attempts=True, also returns list of all attempts
    
    Args:
        img: Input cropped plate image
        use_multi_pass: If False, only run Pass 1
        return_all_attempts: If True, return all attempts (for debugging)
        logger: Optional DebugImageLogger for instrumentation (default: None)
        skip_deskew: If True, skip deskew (image already deskewed at higher level)
    """
    if return_all_attempts and use_multi_pass:
        # Get all attempts for visualization
        raw_text, confidence, method, all_attempts = ocr_plate(img, use_multi_pass=use_multi_pass, return_all_attempts=True, logger=logger, skip_deskew=skip_deskew)
        normalized = normalize_plate(raw_text)
        return raw_text, normalized, confidence, method, all_attempts
    else:
        raw_text, confidence, method = ocr_plate(img, use_multi_pass=use_multi_pass, logger=logger, skip_deskew=skip_deskew)
        normalized = normalize_plate(raw_text)
        return raw_text, normalized, confidence, method
