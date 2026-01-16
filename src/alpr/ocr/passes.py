"""
OCR Passes: Individual OCR Pass Implementations

Each pass applies different preprocessing strategies and OCR parameters
to handle different image quality scenarios.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .easyocr_engine import get_reader
from ..debug_logger import DebugImageLogger


def _ocr_pass_1_clean(img: np.ndarray, logger: Optional[DebugImageLogger] = None) -> Tuple[Optional[str], float]:
    """
    PASS 1 - CLEAN PASS
    Minimal preprocessing for clear, straight plates with good lighting.
    Goal: Fast, high-precision for easy cases.
    
    Args:
        img: Input cropped plate image (ALREADY DESKEWED by ocr_plate())
        logger: Optional DebugImageLogger for instrumentation (default: None)
    
    Returns:
        (text, confidence) or (None, 0.0) if no result
    
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
    reader = get_reader()
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


def _ocr_pass_2_robust(img: np.ndarray, logger: Optional[DebugImageLogger] = None) -> Tuple[Optional[str], float]:
    """
    PASS 2 - ROBUST PASS
    Stronger preprocessing for slight blur, rotation, or uneven lighting.
    Goal: High recall for moderately difficult cases.
    
    Args:
        img: Input cropped plate image (ALREADY DESKEWED by ocr_plate())
        logger: Optional DebugImageLogger for instrumentation (default: None)
    
    Returns:
        (text, confidence) or (None, 0.0) if no result
    
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
    reader = get_reader()
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


def _ocr_pass_3_fallback(img: np.ndarray, logger: Optional[DebugImageLogger] = None) -> Tuple[Optional[str], float]:
    """
    PASS 3 - FALLBACK PASS
    Aggressive preprocessing for very hard cases.
    Tries both normal and inverted images, chooses the better result.
    May produce noisy results - MUST go through normalization + pattern correction.
    Goal: Salvage attempt for difficult images.
    
    Args:
        img: Input cropped plate image (ALREADY DESKEWED by ocr_plate())
        logger: Optional DebugImageLogger for instrumentation (default: None)
    
    Returns:
        (text, confidence) or (None, 0.0) if no result
    
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
    reader = get_reader()
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

