"""
Decision Layer: Quality Gates and Result Acceptance Logic

This module centralizes all gating logic for OCR results.
It determines whether an OCR result should be accepted or rejected
based on confidence scores and pattern validation.
"""

import numpy as np
from typing import Tuple

# Quality gate thresholds for each OCR pass
# These define the minimum standards each pass must meet

# Pass 1: Clean pass - strict thresholds for high confidence
DEFAULT_MIN_CONF_PASS1 = 0.6
DEFAULT_MIN_PATTERN_PASS1 = 0.8

# Pass 2: Robust pass - moderate thresholds
DEFAULT_MIN_CONF_PASS2 = 0.5
DEFAULT_MIN_PATTERN_PASS2 = 0.7

# Pass 3: Fallback pass - lenient but still gated
DEFAULT_MIN_CONF_PASS3 = 0.3
DEFAULT_MIN_PATTERN_PASS3 = 0.5

# Partial line thresholds (for two-line plates)
# Individual lines don't form complete patterns, so pattern validation is relaxed
DEFAULT_MIN_CONF_PASS1_PARTIAL = 0.4
DEFAULT_MIN_PATTERN_PASS1_PARTIAL = 0.0  # No pattern validation for partial lines

DEFAULT_MIN_CONF_PASS2_PARTIAL = 0.3
DEFAULT_MIN_PATTERN_PASS2_PARTIAL = 0.0

DEFAULT_MIN_CONF_PASS3_PARTIAL = 0.2
DEFAULT_MIN_PATTERN_PASS3_PARTIAL = 0.0


def get_thresholds(pass_num: int, is_partial_line: bool = False) -> Tuple[float, float]:
    """
    Get quality gate thresholds for a specific OCR pass.
    
    Args:
        pass_num: OCR pass number (1, 2, or 3)
        is_partial_line: If True, use relaxed thresholds for partial lines
    
    Returns:
        Tuple of (min_confidence, min_pattern_score)
    """
    if is_partial_line:
        if pass_num == 1:
            return DEFAULT_MIN_CONF_PASS1_PARTIAL, DEFAULT_MIN_PATTERN_PASS1_PARTIAL
        elif pass_num == 2:
            return DEFAULT_MIN_CONF_PASS2_PARTIAL, DEFAULT_MIN_PATTERN_PASS2_PARTIAL
        elif pass_num == 3:
            return DEFAULT_MIN_CONF_PASS3_PARTIAL, DEFAULT_MIN_PATTERN_PASS3_PARTIAL
        else:
            raise ValueError(f"Invalid pass_num: {pass_num}")
    else:
        if pass_num == 1:
            return DEFAULT_MIN_CONF_PASS1, DEFAULT_MIN_PATTERN_PASS1
        elif pass_num == 2:
            return DEFAULT_MIN_CONF_PASS2, DEFAULT_MIN_PATTERN_PASS2
        elif pass_num == 3:
            return DEFAULT_MIN_CONF_PASS3, DEFAULT_MIN_PATTERN_PASS3
        else:
            raise ValueError(f"Invalid pass_num: {pass_num}")


def score_result(text: str, confidence: float, pattern_score: float) -> float:
    """
    Compute a combined quality score for an OCR result.
    
    This is a helper function for ranking results. The actual gating
    uses should_accept() which checks both thresholds independently.
    
    Args:
        text: OCR text result
        confidence: OCR confidence score (0.0-1.0)
        pattern_score: Pattern validation score (0.0-1.0)
    
    Returns:
        Combined score (0.0-1.0), weighted average of confidence and pattern
    """
    if not text or confidence <= 0.0 or np.isnan(confidence) or np.isinf(confidence):
        return 0.0
    
    # Weighted combination: 60% confidence, 40% pattern
    # Pattern is important but confidence is primary indicator
    combined = (0.6 * confidence) + (0.4 * pattern_score)
    return float(np.clip(combined, 0.0, 1.0))


def should_accept(
    text: str,
    confidence: float,
    pattern_score: float,
    min_confidence: float,
    min_pattern_score: float,
    is_partial_line: bool = False
) -> bool:
    """
    Determine if an OCR result should be accepted based on quality gates.
    
    This is the central gating function. Both confidence AND pattern
    validation must pass for a result to be accepted.
    
    Design Intent:
    - Hard gate: Both thresholds must be met
    - Confidence alone cannot override pattern validation
    - Pattern validation alone cannot override confidence threshold
    - Partial lines use relaxed pattern validation (full validation happens after combining)
    
    Args:
        text: OCR text result (normalized)
        confidence: OCR confidence score (0.0-1.0)
        pattern_score: Pattern validation score (0.0-1.0)
        min_confidence: Minimum confidence threshold
        min_pattern_score: Minimum pattern validation threshold
        is_partial_line: If True, this is a partial line from a two-line plate
    
    Returns:
        True if result meets quality standards, False otherwise
    """
    # Edge case: Check text validity
    if not text:
        return False
    
    # Edge case: Check confidence validity
    if confidence <= 0.0 or np.isnan(confidence) or np.isinf(confidence):
        return False
    
    # Gate 1: Confidence threshold (must pass)
    if confidence < min_confidence:
        return False
    
    # For partial lines, skip pattern validation (will validate after combining)
    if is_partial_line:
        # Still check basic length constraints
        clean_text = ''.join(c for c in text.upper() if c.isalnum())
        if len(clean_text) < 2 or len(clean_text) > 6:
            return False
        return True
    
    # Full validation for complete plates
    clean_text = ''.join(c for c in text.upper() if c.isalnum())
    
    # Edge case: Check text length (VN plates are typically 7-10 chars after normalization)
    if len(clean_text) < 5 or len(clean_text) > 15:
        return False
    
    # Edge case: Check if text has both letters and numbers (VN plates have both)
    has_letter = any(c.isalpha() for c in clean_text)
    has_number = any(c.isdigit() for c in clean_text)
    if not (has_letter and has_number):
        return False
    
    # Gate 2: Pattern validation threshold (must pass)
    if pattern_score < min_pattern_score:
        return False
    
    return True

