"""
Geometry Module: Plate Geometry Operations

Handles deskewing, cropping, splitting, and border removal operations.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .debug_logger import DebugImageLogger


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


def remove_plate_border(img: np.ndarray, border_ratio: float = 0.08, logger: Optional['DebugImageLogger'] = None) -> np.ndarray:
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
    
    # Instrumentation: log border removal
    if logger:
        logger.save("border_removed", cropped)
    
    return cropped


def is_two_line_plate(crop: np.ndarray) -> bool:
    """
    Classify plate as two-line (motorcycle) or single-line (car).
    
    Uses:
    1. Aspect ratio (w/h < 3.2 suggests two-line)
    2. Horizontal projection profile to detect gap between lines
    
    Args:
        crop: Cropped plate image
    
    Returns:
        bool: True if two-line plate, False if single-line
    """
    h, w = crop.shape[:2]
    ratio = w / h
    
    # Primary check: aspect ratio
    # Motorcycle plates: typically ratio < 3.2 (shorter and wider)
    # Car plates: typically ratio > 3.5 (longer and narrower)
    if ratio >= 3.5:
        return False  # Definitely single-line
    if ratio < 2.5:
        return True   # Definitely two-line
    
    # Ambiguous range (2.5 <= ratio < 3.5): use gap detection
    # Convert to grayscale if needed
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Horizontal projection: sum pixels in each row
    h_projection = np.sum(binary == 0, axis=1)  # Count black pixels (text)
    
    # Find gap: region with few text pixels (whitespace between lines)
    # Look for a horizontal gap that's at least 10% of height
    min_gap_height = max(3, int(h * 0.1))
    
    # Find rows with very few text pixels (potential gap)
    gap_threshold = np.max(h_projection) * 0.2  # Gap has <20% of max text density
    gap_rows = np.where(h_projection < gap_threshold)[0]
    
    if len(gap_rows) == 0:
        return ratio < 3.2  # No gap found, fallback to ratio
    
    # Check if there's a continuous gap region
    gap_regions = []
    start = gap_rows[0]
    for i in range(1, len(gap_rows)):
        if gap_rows[i] - gap_rows[i-1] > 1:
            # Gap broken, save previous region
            if gap_rows[i-1] - start >= min_gap_height:
                gap_regions.append((start, gap_rows[i-1]))
            start = gap_rows[i]
    # Check last region
    if gap_rows[-1] - start >= min_gap_height:
        gap_regions.append((start, gap_rows[-1]))
    
    # If found significant gap, likely two-line
    if len(gap_regions) > 0:
        # Check if gap is roughly in the middle (not at edges)
        for gap_start, gap_end in gap_regions:
            gap_center = (gap_start + gap_end) / 2
            if 0.3 * h < gap_center < 0.7 * h:  # Gap in middle 40% of image
                return True
    
    # No significant gap found, use ratio as fallback
    return ratio < 3.2


def crop_text_region(img: np.ndarray, margin_ratio: float = 0.05, logger: Optional['DebugImageLogger'] = None) -> np.ndarray:
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


def split_two_line_plate(crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split two-line motorcycle plate into top and bottom halves.
    Tries to detect actual gap between lines, falls back to fixed split with overlap.
    
    Args:
        crop: Two-line plate image
    
    Returns:
        (top, bottom): Top and bottom halves
    """
    h, w = crop.shape[:2]
    
    # Convert to grayscale if needed
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Horizontal projection: sum pixels in each row
    h_projection = np.sum(binary == 0, axis=1)  # Count black pixels (text)
    
    # Find gap: region with few text pixels (whitespace between lines)
    min_gap_height = max(3, int(h * 0.08))  # Gap should be at least 8% of height
    gap_threshold = np.max(h_projection) * 0.25  # Gap has <25% of max text density
    
    # Find rows with very few text pixels (potential gap)
    gap_rows = np.where(h_projection < gap_threshold)[0]
    
    split_point = h // 2  # Default: split in middle
    
    if len(gap_rows) > 0:
        # Find continuous gap regions
        gap_regions = []
        start = gap_rows[0]
        for i in range(1, len(gap_rows)):
            if gap_rows[i] - gap_rows[i-1] > 1:
                # Gap broken, save previous region
                if gap_rows[i-1] - start >= min_gap_height:
                    gap_regions.append((start, gap_rows[i-1]))
                start = gap_rows[i]
        # Check last region
        if gap_rows[-1] - start >= min_gap_height:
            gap_regions.append((start, gap_rows[-1]))
        
        # Use the gap region closest to middle
        if len(gap_regions) > 0:
            # Find gap closest to center
            center = h / 2
            best_gap = min(gap_regions, key=lambda g: abs((g[0] + g[1]) / 2 - center))
            split_point = (best_gap[0] + best_gap[1]) // 2
    
    # Use overlap to prevent losing characters at boundary
    overlap = max(5, h // 10)  # 10% overlap, minimum 5 pixels
    
    # Top: from start to split_point + overlap
    top_end = min(split_point + overlap, h)
    top = crop[0:top_end, :]
    
    # Bottom: from split_point - overlap to end
    bot_start = max(0, split_point - overlap)
    bottom = crop[bot_start:h, :]
    
    return top, bottom


def deskew_plate(
    img: np.ndarray,
    angle_threshold: float = 0.8,
    debug: bool = False,
    logger: Optional['DebugImageLogger'] = None,
    return_angle: bool = False
) -> np.ndarray:
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
        return (img, 0.0) if return_angle else img  # No contours found, return original
    
    # Filter contours to find character-like regions
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
    
    # Estimate angle using minAreaRect on character contours
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
    
    # Projection profile method (PRIMARY - most accurate for text)
    # Reuse binary image from contour detection
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
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=0)
        h_projection = np.sum(rotated, axis=1)
        score = np.var(h_projection)
        if score > best_score_coarse:
            best_score_coarse = score
            best_angle_coarse = test_angle
    
    # Step 2: Fine search around best coarse angle (±1.5° with 0.15° step) - 20 iterations
    angles_fine = np.arange(best_angle_coarse - 1.5, best_angle_coarse + 1.5, 0.15)
    best_angle = best_angle_coarse
    best_score = best_score_coarse
    
    for test_angle in angles_fine:
        M = cv2.getRotationMatrix2D(center, test_angle, 1.0)
        rotated = cv2.warpAffine(binary_proj, M, (w, h), 
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=0)
        h_projection = np.sum(rotated, axis=1)
        score = np.var(h_projection)
        if score > best_score:
            best_score = score
            best_angle = test_angle
    
    # Combine methods: Prefer projection profile (most accurate)
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
        return (img, 0.0) if return_angle else img  # Plate is already horizontal enough
    
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
    
    return (corrected, angle) if return_angle else corrected

