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
# Detect individual characters using contour detection (IMPROVED)
# ----------------------------------------------------------
def detect_individual_characters(preprocessed_img, debug=False):
    """
    Detect and separate individual characters using contour detection.
    Improved version with border removal, adaptive morphology, and better thresholding.
    
    Returns: 
        tuple: (char_boxes, binary_image)
        - char_boxes: list of (x, y, w, h, char_img) sorted left to right
        - binary_image: binary image used for detection (for visualization)
    """
    h_img, w_img = preprocessed_img.shape[:2]
    
    # Check if image is mostly dark (inverted) or mostly light (normal)
    mean_val = np.mean(preprocessed_img)
    is_inverted = mean_val < 127
    
    # IMPROVED THRESHOLDING: Use adaptive thresholding for more stable results
    # Adaptive thresholding is less sensitive to background brightness variations
    block_size = max(11, min(w_img, h_img) // 20)  # Adaptive block size based on image size
    if block_size % 2 == 0:
        block_size += 1  # Must be odd
    
    if is_inverted:
        # Inverted image: white text on black background
        # Try adaptive threshold first, fallback to Otsu
        try:
            binary = cv2.adaptiveThreshold(
                preprocessed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, 2
            )
        except:
            _, binary = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Normal image: black text on white background
        try:
            binary = cv2.adaptiveThreshold(
                preprocessed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, 2
            )
        except:
            _, binary = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # IMPROVED MORPHOLOGY: Adaptive kernel sizes based on image dimensions
    # Character width is typically 5-15% of image width
    char_width_estimate = max(3, int(w_img * 0.08))  # Estimate character width
    char_height_estimate = max(3, int(h_img * 0.4))  # Estimate character height
    
    # Horizontal kernel: slightly smaller than character width to create gaps
    kernel_h_size = max(3, char_width_estimate // 3)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_h_size, 1))
    
    # Vertical kernel: small to separate vertically connected parts
    kernel_v_size = max(1, char_height_estimate // 10)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_v_size))
    
    # Step 1: Close small gaps within characters (fill holes)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Step 2: Open horizontally to separate characters
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal, iterations=2)
    
    # Step 3: Open vertically to separate vertically connected parts
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical, iterations=1)
    
    if debug:
        print(f"  DEBUG contour: Adaptive morphology - kernel_h={kernel_h_size}x1, kernel_v=1x{kernel_v_size}")
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"  DEBUG contour: Found {len(contours)} contours before border removal, img_size={w_img}x{h_img}")
    
    # REMOVE BORDER CONTOURS FIRST
    contours = remove_border_contours(contours, (h_img, w_img), border_margin=0.05, debug=debug)
    
    # First pass: collect all valid contours with their bounding boxes
    valid_contours = []
    img_area = h_img * w_img
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # IMPROVED FILTERING: More robust thresholds
        min_area = img_area * 0.001  # 0.1% of image (lower threshold for small chars)
        max_area = img_area * 0.12   # 12% max (exclude large merged blocks)
        
        aspect_ratio = h / w if w > 0 else 0
        min_height = h_img * 0.12  # 12% min height
        max_height = h_img * 0.80   # 80% max height
        
        # Additional checks: exclude very wide or very tall contours (likely merged or noise)
        width_ratio = w / w_img
        height_ratio = h / h_img
        
        if debug and i < 20:  # Debug first 20 contours
            print(f"    Contour {i}: area={area:.0f} ({area/img_area*100:.2f}%), "
                  f"aspect={aspect_ratio:.2f}, h={h} ({h/h_img*100:.1f}%), "
                  f"w={w} ({w/w_img*100:.1f}%), bbox=({x},{y},{w},{h})")
        
        # Better filtering: exclude very large blocks, very small noise, and edge-touching contours
        touches_edge = (x < 5 or y < 5 or (x + w) > (w_img - 5) or (y + h) > (h_img - 5))
        
        if (min_area < area < max_area and 
            0.15 < aspect_ratio < 6.0 and  # Wider aspect ratio range
            min_height < h < max_height and
            width_ratio < 0.25 and  # Character shouldn't span >25% of width
            w > 2 and h > 2 and  # Minimum size
            not (touches_edge and area > img_area * 0.05)):  # Exclude large edge-touching contours
            valid_contours.append((x, y, w, h, area))
    
    # Second pass: remove nested contours (smaller contours inside larger ones)
    valid_contours.sort(key=lambda c: c[4], reverse=True)
    char_boxes = []
    
    for i, (x1, y1, w1, h1, area1) in enumerate(valid_contours):
        is_nested = False
        
        # Check if this contour is nested inside a larger one
        for j, (x2, y2, w2, h2, area2) in enumerate(valid_contours):
            if i == j or area2 <= area1:
                continue
            
            # Check if contour 1 is inside contour 2
            margin = 0.15  # Increased margin to 15% for better detection
            if (x2 - margin*w2 <= x1 <= x2 + w2 + margin*w2 and
                y2 - margin*h2 <= y1 <= y2 + h2 + margin*h2 and
                x2 - margin*w2 <= x1 + w1 <= x2 + w2 + margin*w2 and
                y2 - margin*h2 <= y1 + h1 <= y2 + h2 + margin*h2):
                # Check if area overlap is significant (>40% of smaller contour)
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                if overlap_area > 0.4 * min(area1, area2):
                    is_nested = True
                    break
        
        if not is_nested:
            # Add padding
            padding = max(2, min(w1, h1) // 8)  # Reduced padding to avoid merging
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
    
    return char_boxes, binary  # Return binary image for visualization

# ----------------------------------------------------------
# Detect characters using vertical projection (alternative method)
# ----------------------------------------------------------
def detect_characters_vertical_projection(preprocessed_img, debug=False):
    """
    Alternative method: Detect character boundaries using vertical projection.
    More stable for well-separated characters, less sensitive to morphology issues.
    
    Returns:
        tuple: (char_boxes, binary_image)
        - char_boxes: list of (x, y, w, h, char_img) sorted left to right
        - binary_image: binary image used for detection (for visualization)
    """
    h_img, w_img = preprocessed_img.shape[:2]
    
    # Create binary image
    mean_val = np.mean(preprocessed_img)
    is_inverted = mean_val < 127
    
    if is_inverted:
        _, binary = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate vertical projection (sum of white pixels in each column)
    projection = np.sum(binary, axis=0)  # Shape: (width,)
    
    # Find character boundaries
    # Threshold: columns with projection > threshold contain characters
    threshold = np.max(projection) * 0.1  # 10% of max projection
    
    # Find start and end of each character region
    char_regions = []
    in_char = False
    start_x = 0
    
    for x in range(w_img):
        if projection[x] > threshold:
            if not in_char:
                start_x = x
                in_char = True
        else:
            if in_char:
                # End of character region
                end_x = x
                char_width = end_x - start_x
                if char_width > 3:  # Minimum character width
                    char_regions.append((start_x, end_x))
                in_char = False
    
    # Handle case where last character extends to edge
    if in_char:
        char_regions.append((start_x, w_img))
    
    # Convert regions to bounding boxes
    char_boxes = []
    for start_x, end_x in char_regions:
        # Extract vertical range (find top and bottom of character)
        char_slice = binary[:, start_x:end_x]
        row_projection = np.sum(char_slice, axis=1)
        
        # Find top and bottom
        row_threshold = np.max(row_projection) * 0.1
        top_y = 0
        bottom_y = h_img
        
        for y in range(h_img):
            if row_projection[y] > row_threshold:
                top_y = y
                break
        
        for y in range(h_img - 1, -1, -1):
            if row_projection[y] > row_threshold:
                bottom_y = y + 1
                break
        
        w = end_x - start_x
        h = bottom_y - top_y
        
        # Add padding
        padding = max(2, min(w, h) // 8)
        x = max(0, start_x - padding)
        y = max(0, top_y - padding)
        w = min(w_img - x, w + 2 * padding)
        h = min(h_img - y, h + 2 * padding)
        
        char_img = preprocessed_img[y:y+h, x:x+w]
        char_boxes.append((x, y, w, h, char_img))
    
    if debug:
        print(f"  DEBUG vertical_projection: Found {len(char_boxes)} characters")
    
    return char_boxes, binary

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
