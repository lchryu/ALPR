"""
OCR Waterfall: Multi-Pass OCR Orchestration

Implements the 3-pass waterfall OCR pipeline with quality gates.
Uses the decision layer for centralized gating logic.
"""

import numpy as np
from typing import Tuple, Optional, List
from .passes import _ocr_pass_1_clean, _ocr_pass_2_robust, _ocr_pass_3_fallback
from ..decision import should_accept, get_thresholds
from ..validation import normalize_plate, validate_vn_plate_pattern
from ..geometry import deskew_plate, remove_plate_border
from ..debug_logger import DebugImageLogger


def ocr_plate(
    img: np.ndarray,
    use_multi_pass: bool = True,
    return_all_attempts: bool = False,
    logger: Optional[DebugImageLogger] = None,
    skip_deskew: bool = False,
    is_partial_line: bool = False
) -> Tuple[str, float, str] | Tuple[str, float, str, List[Tuple[str, float, str]]]:
    """
    3-PASS WATERFALL OCR PIPELINE
    
    Design Philosophy:
    - Fast path for easy cases (Pass 1)
    - Progressive fallback for difficult cases (Pass 2 → Pass 3)
    - Quality gates at every stage (never return garbage)
    - Prefer explicit failure over false positives
    
    Waterfall logic:
    1. Run Pass 1 (clean) → if valid → STOP
    2. Run Pass 2 (robust) → if valid → STOP  
    3. Run Pass 3 (fallback) → validate with lower thresholds → return if valid
    4. If all passes fail → return empty result (explicit failure)
    
    Quality Gates:
    - Pass 1: Strict (conf ≥ 0.6, pattern ≥ 0.8)
    - Pass 2: Moderate (conf ≥ 0.5, pattern ≥ 0.7)
    - Pass 3: Lenient but still gated (conf ≥ 0.3, pattern ≥ 0.5)
    - All passes: Must pass BOTH confidence AND pattern validation
    
    Args:
        img: Input cropped license plate image
        use_multi_pass: If False, only run Pass 1
        return_all_attempts: If True, return all attempts (for debugging)
        logger: Optional DebugImageLogger for instrumentation (default: None)
        skip_deskew: If True, skip deskew (image already deskewed at higher level)
        is_partial_line: If True, this is a partial line from a two-line plate (use relaxed validation)
    
    Returns:
        Success: (text, confidence, method) where method ∈ {"pass1_clean", "pass2_robust", "pass3_fallback"}
        Failure: ("", 0.0, "none") - explicit failure, not garbage result
        If return_all_attempts=True, also returns list of all attempts
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
            normalized = normalize_plate(text)
            min_conf, min_pattern = get_thresholds(1, is_partial_line)
            pattern_score = validate_vn_plate_pattern(normalized)
            if should_accept(normalized, confidence, pattern_score, min_conf, min_pattern, is_partial_line):
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
        min_conf, min_pattern = get_thresholds(1, is_partial_line)
        pattern_score = validate_vn_plate_pattern(normalized)
        if should_accept(normalized, confidence, pattern_score, min_conf, min_pattern, is_partial_line):
            if return_all_attempts:
                return text, confidence, "pass1_clean", all_attempts
            return text, confidence, "pass1_clean"
    
    # PASS 2: Robust pass (use image without border)
    text, confidence = _ocr_pass_2_robust(img_no_border, logger=logger)
    if text:
        normalized = normalize_plate(text)
        all_attempts.append((text, confidence, "pass2_robust"))
        
        # Validate using normalized text, but return raw text
        min_conf, min_pattern = get_thresholds(2, is_partial_line)
        pattern_score = validate_vn_plate_pattern(normalized)
        if should_accept(normalized, confidence, pattern_score, min_conf, min_pattern, is_partial_line):
            if return_all_attempts:
                return text, confidence, "pass2_robust", all_attempts
            return text, confidence, "pass2_robust"
    
    # PASS 3: Fallback pass (use image without border, validate with lower thresholds)
    text, confidence = _ocr_pass_3_fallback(img_no_border, logger=logger)
    if text:
        normalized = normalize_plate(text)
        all_attempts.append((text, confidence, "pass3_fallback"))
        
        # Validate with lower thresholds but still validate (don't return garbage)
        min_conf, min_pattern = get_thresholds(3, is_partial_line)
        pattern_score = validate_vn_plate_pattern(normalized)
        if should_accept(normalized, confidence, pattern_score, min_conf, min_pattern, is_partial_line):
            if return_all_attempts:
                return text, confidence, "pass3_fallback", all_attempts
            return text, confidence, "pass3_fallback"
        # If Pass 3 result is too bad, don't return it
        if logger:
            print(f"  Pass 3 result rejected: text='{text}', normalized='{normalized}', conf={confidence:.3f}")
    
    # All passes failed
    # DESIGN INTENT: Return explicit failure rather than garbage result
    # Empty result signals "no valid OCR found" vs returning low-confidence noise
    if return_all_attempts:
        return "", 0.0, "none", all_attempts
    return "", 0.0, "none"


def ocr_plate_complete(
    img: np.ndarray,
    use_multi_pass: bool = True,
    return_all_attempts: bool = False,
    logger: Optional[DebugImageLogger] = None,
    skip_deskew: bool = False,
    is_partial_line: bool = False
) -> Tuple[str, str, float, str] | Tuple[str, str, float, str, List[Tuple[str, float, str]]]:
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
        is_partial_line: If True, this is a partial line from a two-line plate (use relaxed validation)
    """
    if return_all_attempts and use_multi_pass:
        # Get all attempts for visualization
        raw_text, confidence, method, all_attempts = ocr_plate(
            img, use_multi_pass=use_multi_pass, return_all_attempts=True,
            logger=logger, skip_deskew=skip_deskew, is_partial_line=is_partial_line
        )
        normalized = normalize_plate(raw_text)
        return raw_text, normalized, confidence, method, all_attempts
    else:
        raw_text, confidence, method = ocr_plate(
            img, use_multi_pass=use_multi_pass,
            logger=logger, skip_deskew=skip_deskew, is_partial_line=is_partial_line
        )
        normalized = normalize_plate(raw_text)
        return raw_text, normalized, confidence, method

