import cv2
import numpy as np
import re
import easyocr

# Create reader 1 lần (tối ưu)
reader = easyocr.Reader(['en'])

# ----------------------------------------------------------
# Detect individual characters using contour detection
# ----------------------------------------------------------
def detect_individual_characters(preprocessed_img, debug=False):
    """
    Detect and separate individual characters using contour detection.
    Returns: list of (x, y, w, h, char_img) sorted left to right
    """
    # Create binary image - try both normal and inverted
    # For inverted images (white text on black), we need normal threshold
    # For normal images (black text on white), we need inverted threshold
    
    # Check if image is mostly dark (inverted) or mostly light (normal)
    mean_val = np.mean(preprocessed_img)
    is_inverted = mean_val < 127
    
    if is_inverted:
        # Inverted image: white text on black background
        _, binary = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Normal image: black text on white background
        _, binary = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to separate characters
    # Use horizontal kernel to create gaps between characters
    # Try stronger separation for merged characters
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Larger kernel (was 3,1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal, iterations=2)  # More iterations (was 1)
    
    # Additional step: Try vertical erosion to separate vertically connected parts
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_boxes = []
    h_img, w_img = preprocessed_img.shape[:2]
    
    if debug:
        print(f"  DEBUG contour: Found {len(contours)} contours, img_size={w_img}x{h_img}")
    
    # First pass: collect all valid contours with their bounding boxes
    valid_contours = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size and aspect ratio
        area = w * h
        img_area = h_img * w_img
        min_area = img_area * 0.002  # 0.2% of image
        max_area = img_area * 0.15    # Reduced from 0.3 to 0.15 (15% max) - exclude large blocks
        
        aspect_ratio = h / w if w > 0 else 0
        min_height = h_img * 0.15  # Reduced to 15% (allow smaller characters)
        max_height = h_img * 0.85   # Increased to 85% (allow taller characters, but still filter huge blocks)
        
        if debug and i < 15:  # Debug first 15 contours
            print(f"    Contour {i}: area={area:.0f} ({area/img_area*100:.2f}%), "
                  f"aspect={aspect_ratio:.2f}, h={h} ({h/h_img*100:.1f}%), "
                  f"w={w}, h_min={min_height:.0f}, h_max={max_height:.0f}")
        
        # Better filtering: exclude very large blocks and very small noise
        if (min_area < area < max_area and 
            0.2 < aspect_ratio < 5.0 and
            min_height < h < max_height and  # Height must be in reasonable range
            w > 3 and h > 3):  # Minimum size
            valid_contours.append((x, y, w, h, area))
    
    # Second pass: remove nested contours (smaller contours inside larger ones)
    # Sort by area (largest first) to check nesting
    valid_contours.sort(key=lambda c: c[4], reverse=True)
    char_boxes = []
    
    for i, (x1, y1, w1, h1, area1) in enumerate(valid_contours):
        is_nested = False
        
        # Check if this contour is nested inside a larger one
        for j, (x2, y2, w2, h2, area2) in enumerate(valid_contours):
            if i == j or area2 <= area1:
                continue
            
            # Check if contour 1 is inside contour 2
            # Allow some margin (10%) for slight misalignment
            margin = 0.1
            if (x2 - margin*w2 <= x1 <= x2 + w2 + margin*w2 and
                y2 - margin*h2 <= y1 <= y2 + h2 + margin*h2 and
                x2 - margin*w2 <= x1 + w1 <= x2 + w2 + margin*w2 and
                y2 - margin*h2 <= y1 + h1 <= y2 + h2 + margin*h2):
                # Check if area overlap is significant (>50% of smaller contour)
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                if overlap_area > 0.5 * min(area1, area2):
                    is_nested = True
                    break
        
        if not is_nested:
            # Add padding
            padding = max(2, min(w1, h1) // 6)
            x = max(0, x1 - padding)
            y = max(0, y1 - padding)
            w = min(w_img - x, w1 + 2 * padding)
            h = min(h_img - y, h1 + 2 * padding)
            
            char_img = preprocessed_img[y:y+h, x:x+w]
            char_boxes.append((x, y, w, h, char_img))
    
    # Sort left to right
    char_boxes.sort(key=lambda box: box[0])
    
    if debug:
        print(f"  DEBUG contour: After filtering: {len(char_boxes)} characters detected")
    
    return char_boxes

# ----------------------------------------------------------
# Detect if plate is 2-line or 1-line based on aspect ratio
# ----------------------------------------------------------
def is_two_line_plate(crop):
    h, w = crop.shape[:2]
    ratio = w / h
    return ratio < 3.2

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
def deskew_plate(img, angle_threshold=2.0, debug=False):
    """
    Detect and correct rotation/skew in license plate image.
    Only applies correction if angle exceeds threshold (to avoid affecting horizontal plates).
    
    Args:
        img: Input image (BGR or grayscale)
        angle_threshold: Minimum angle (degrees) to trigger correction (default: 2.0)
        debug: If True, print detected angle (default: False)
    
    Returns:
        Corrected image (same format as input)
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
        return img
    
    # Method 1: Use minAreaRect on text contours
    # Create binary image to find text regions
    # Try both normal and inverted thresholds to handle different plate styles
    _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours for both
    contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use the binary with more/larger contours
    if len(contours1) > len(contours2) or (contours1 and not contours2):
        binary = binary1
        contours = contours1
    else:
        binary = binary2
        contours = contours2
    
    if not contours:
        return img  # No contours found, return original
    
    # Find the largest contour (should be the plate text region)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Normalize angle to [-45, 45] range
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    
    # Only correct if angle exceeds threshold
    if abs(angle) < angle_threshold:
        if debug:
            print(f"  Deskew: Angle {angle:.2f}° < threshold {angle_threshold}°, skipping correction")
        return img  # Plate is already horizontal enough
    
    if debug:
        print(f"  Deskew: Detected angle {angle:.2f}°, applying correction")
    
    # Method 2: Fallback to projection profile method if minAreaRect gives extreme angle
    if abs(angle) > 30:
        # Try projection profile method
        # Re-create binary for projection method
        _, binary_proj = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        angles = np.arange(-15, 15, 0.5)
        best_angle = 0
        best_score = 0
        
        for test_angle in angles:
            # Rotate image
            center = (gray.shape[1] // 2, gray.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, test_angle, 1.0)
            rotated = cv2.warpAffine(binary_proj, M, (gray.shape[1], gray.shape[0]), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Calculate horizontal projection variance (higher = better alignment)
            h_projection = np.sum(rotated, axis=1)
            score = np.var(h_projection)
            
            if score > best_score:
                best_score = score
                best_angle = test_angle
        
        if abs(best_angle) >= angle_threshold:
            angle = best_angle
    
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
    
    return corrected

# ----------------------------------------------------------
# Preprocess plate - Optimized for EasyOCR
# ----------------------------------------------------------
def preprocess_plate(img, variant="standard", apply_deskew=True):
    """
    Clean preprocessing pipeline optimized for EasyOCR.
    Focuses on clarity without destroying texture.
    
    Args:
        img: Input image (BGR or grayscale)
        variant: "standard", "high_contrast", "sharp", "clean"
        apply_deskew: Whether to apply rotation correction (default: True)
    """
    # 0. Deskew/rotation correction (applied first, before upscaling)
    if apply_deskew:
        img = deskew_plate(img, angle_threshold=2.0, debug=True)
    
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
    
    return final

# ----------------------------------------------------------
# Multi-pass OCR with different preprocessing variants
# ----------------------------------------------------------
def ocr_plate(img, use_multi_pass=True, return_all_attempts=False):
    """
    OCR plate with multi-pass attempts for better accuracy.
    Returns: (text, confidence, method_used)
    """
    if not use_multi_pass:
        # Single pass - simple and fast
        # Deskew first, then preprocess
        img_deskewed = deskew_plate(img, angle_threshold=2.0, debug=True)
        preprocessed = preprocess_plate(img_deskewed, variant="standard", apply_deskew=False)
        results = reader.readtext(
            preprocessed, 
            detail=1, 
            paragraph=False,
            width_ths=0.7,  # Lower threshold for better character detection
            height_ths=0.7
        )
        
        if not results:
            return "", 0.0, "single"
        
        # Combine all detected text
        text_parts = [r[1] for r in results]
        confidences = [r[2] for r in results]
        
        combined_text = "".join(text_parts)
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        return combined_text, avg_conf, "single"
    
    # Multi-pass OCR with different preprocessing variants
    attempts = []
    
    # EasyOCR parameters for better detection
    # Lower thresholds = more sensitive to characters (better for small/blurry text)
    ocr_params = {
        'detail': 1,
        'paragraph': False,
        'width_ths': 0.4,  # Lower = more sensitive to characters (was 0.6)
        'height_ths': 0.4,  # Lower = more sensitive to characters (was 0.6)
        'slope_ths': 0.1,  # Allow slight rotation
        'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # Only allow alphanumeric
    }
    
    # More sensitive parameters for detecting small/thin characters
    ocr_params_sensitive = {
        'detail': 1,
        'paragraph': False,
        'width_ths': 0.3,  # Very sensitive for thin characters like '1'
        'height_ths': 0.3,  # Very sensitive for small characters
        'slope_ths': 0.1,
        'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    }
    
    # Deskew ONCE before all preprocessing variants (to avoid redundant processing)
    # This ensures rotation correction is applied only once, not 12+ times
    img_deskewed = deskew_plate(img, angle_threshold=2.0, debug=True)
    
    # Pass 1: Standard preprocessing (deskew already applied, skip it in preprocess_plate)
    pre1 = preprocess_plate(img_deskewed, variant="standard", apply_deskew=False)
    results1 = reader.readtext(pre1, **ocr_params)
    if results1:
        text1 = "".join([r[1] for r in results1])
        conf1 = np.mean([r[2] for r in results1])
        attempts.append((text1, conf1, "standard"))
    
    # Pass 1b: Standard with sensitive parameters (for thin characters)
    results1b = reader.readtext(pre1, **ocr_params_sensitive)
    if results1b:
        text1b = "".join([r[1] for r in results1b])
        conf1b = np.mean([r[2] for r in results1b])
        attempts.append((text1b, conf1b, "standard_sensitive"))
    
    # Pass 2: High contrast variant (deskew already applied)
    pre2 = preprocess_plate(img_deskewed, variant="high_contrast", apply_deskew=False)
    results2 = reader.readtext(pre2, **ocr_params)
    if results2:
        text2 = "".join([r[1] for r in results2])
        conf2 = np.mean([r[2] for r in results2])
        attempts.append((text2, conf2, "high_contrast"))
    
    # Pass 3: Sharp variant (deskew already applied)
    pre3 = preprocess_plate(img_deskewed, variant="sharp", apply_deskew=False)
    results3 = reader.readtext(pre3, **ocr_params)
    if results3:
        text3 = "".join([r[1] for r in results3])
        conf3 = np.mean([r[2] for r in results3])
        attempts.append((text3, conf3, "sharp"))
    
    # Pass 4: Clean variant (more denoising, deskew already applied)
    pre4 = preprocess_plate(img_deskewed, variant="clean", apply_deskew=False)
    results4 = reader.readtext(pre4, **ocr_params)
    if results4:
        text4 = "".join([r[1] for r in results4])
        conf4 = np.mean([r[2] for r in results4])
        attempts.append((text4, conf4, "clean"))
    
    # Pass 5: Inverted (for dark text on light background)
    # This often works better for plates with light background
    pre5 = preprocess_plate(img_deskewed, variant="standard", apply_deskew=False)
    inverted = cv2.bitwise_not(pre5)
    results5 = reader.readtext(inverted, **ocr_params)
    if results5:
        text5 = "".join([r[1] for r in results5])
        conf5 = np.mean([r[2] for r in results5])
        attempts.append((text5, conf5, "inverted"))
    
    # Pass 5b: Inverted with high contrast
    pre5b = preprocess_plate(img_deskewed, variant="high_contrast", apply_deskew=False)
    inverted5b = cv2.bitwise_not(pre5b)
    results5b = reader.readtext(inverted5b, **ocr_params)
    if results5b:
        text5b = "".join([r[1] for r in results5b])
        conf5b = np.mean([r[2] for r in results5b])
        attempts.append((text5b, conf5b, "inverted_high_contrast"))
    
    # Pass 6: Very high scale for maximum clarity (especially for G vs 6)
    h, w = img_deskewed.shape[:2]
    scale = 6.0 if min(h, w) < 100 else 5.0  # Very high scale
    img6 = cv2.resize(img_deskewed, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    if len(img6.shape) == 3:
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
    else:
        gray6 = img6.copy()
    
    # Add padding
    padding = 30
    gray6 = cv2.copyMakeBorder(gray6, padding, padding, padding, padding, 
                               cv2.BORDER_CONSTANT, value=255)
    
    # Strong CLAHE for maximum contrast
    clahe6 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced6 = clahe6.apply(gray6)
    
    # Strong sharpening to emphasize character details
    kernel6 = np.array([
        [0, -1, 0],
        [-1, 6, -1],
        [0, -1, 0]
    ])
    sharp6 = cv2.filter2D(enhanced6, -1, kernel6)
    sharp6 = np.clip(sharp6, 0, 255).astype(np.uint8)
    
    results6 = reader.readtext(sharp6, **ocr_params)
    if results6:
        text6 = "".join([r[1] for r in results6])
        conf6 = np.mean([r[2] for r in results6])
        attempts.append((text6, conf6, "ultra_high_scale"))
    
    # Pass 7: Adaptive threshold variant
    h, w = img_deskewed.shape[:2]
    scale7 = 4.0 if min(h, w) < 100 else 3.5
    img7 = cv2.resize(img_deskewed, None, fx=scale7, fy=scale7, interpolation=cv2.INTER_CUBIC)
    if len(img7.shape) == 3:
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
    else:
        gray7 = img7.copy()
    
    # Add padding
    padding = 20
    gray7 = cv2.copyMakeBorder(gray7, padding, padding, padding, padding, 
                               cv2.BORDER_CONSTANT, value=255)
    
    # Adaptive threshold with better parameters
    # Use larger block size and constant to avoid over-thresholding
    adaptive = cv2.adaptiveThreshold(
        gray7, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 5  # Larger block size (15 vs 11), more constant (5 vs 2)
    )
    
    results7 = reader.readtext(adaptive, **ocr_params)
    if results7:
        text7 = "".join([r[1] for r in results7])
        conf7 = np.mean([r[2] for r in results7])
        attempts.append((text7, conf7, "adaptive_thresh"))
    
    # Pick best result - not just by confidence, but also by pattern validation
    if not attempts:
        if return_all_attempts:
            return "", 0.0, "none", []
        return "", 0.0, "none"
    
    # Score each attempt: confidence + pattern validation
    scored_attempts = []
    for text, conf, method in attempts:
        pattern_score = validate_vn_plate_pattern(text)
        # Combined score: 70% confidence + 30% pattern validation
        combined_score = conf * 0.7 + pattern_score * 0.3
        scored_attempts.append((text, conf, method, combined_score, pattern_score))
        
        # DEBUG: Print pattern validation details
        if return_all_attempts:
            clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())
            pos3_char = clean_text[2] if len(clean_text) > 2 else "N/A"
            pos3_type = "LETTER" if pos3_char.isalpha() else "NUMBER"
            print(f"  DEBUG: {method}: text='{text}' -> clean='{clean_text}' -> pos3='{pos3_char}' ({pos3_type}) -> pattern={pattern_score:.2f}")
        
        # DEBUG: Print pattern validation details
        if return_all_attempts:
            clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())
            pos3_char = clean_text[2] if len(clean_text) > 2 else "N/A"
            print(f"  DEBUG: {method}: text='{text}' -> clean='{clean_text}' -> pos3='{pos3_char}' -> pattern={pattern_score:.2f}")
    
    # Sort by combined score (descending)
    scored_attempts.sort(key=lambda x: x[3], reverse=True)
    
    # Get best result
    best = scored_attempts[0]
    best_text, best_conf, best_method = best[0], best[1], best[2]
    
    if return_all_attempts:
        # Return all attempts with scores for debugging
        all_attempts_with_scores = [(t, c, m, cs, ps) for t, c, m, cs, ps in scored_attempts]
        return best_text, best_conf, best_method, [(t, c, m) for t, c, m, _, _ in scored_attempts]
    return best_text, best_conf, best_method

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
    
    # Apply replacements (but preserve position 3 if it's a letter)
    result = []
    for i, char in enumerate(text):
        if i == 2 and char.isalpha():
            # Position 3: ALWAYS keep as letter (don't replace)
            result.append(char)
        elif char in replacements:
            # Other positions: apply replacement
            result.append(replacements[char])
        else:
            result.append(char)
    
    return ''.join(result)

# ----------------------------------------------------------
# Complete OCR pipeline for plate
# ----------------------------------------------------------
def ocr_plate_complete(img, use_multi_pass=True, return_all_attempts=False):
    """
    Complete OCR pipeline: preprocess -> OCR -> normalize
    Returns: (raw_text, normalized_text, confidence, method)
    If return_all_attempts=True, also returns list of all attempts
    """
    if return_all_attempts and use_multi_pass:
        # Get all attempts for visualization
        raw_text, confidence, method, all_attempts = ocr_plate(img, use_multi_pass=use_multi_pass, return_all_attempts=True)
        normalized = normalize_plate(raw_text)
        return raw_text, normalized, confidence, method, all_attempts
    else:
        raw_text, confidence, method = ocr_plate(img, use_multi_pass=use_multi_pass)
        normalized = normalize_plate(raw_text)
        return raw_text, normalized, confidence, method
