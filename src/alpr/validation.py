"""
Validation Module: Text Normalization and Pattern Matching

Handles Vietnamese license plate text normalization and pattern validation.
"""

import re
from typing import Optional


def validate_vn_plate_pattern(text: str) -> float:
    """
    Validate if text matches Vietnamese plate pattern: XXY-XXXXX
    
    Pattern rules:
    - Position 3 (index 2) should be a LETTER
    - Length should be reasonable (7-9 characters after removing special chars)
    
    Args:
        text: Text to validate
    
    Returns:
        Score from 0.0 to 1.0:
        - 1.0: Perfect pattern match (position 3 is letter)
        - 0.3: Position 3 is number (common mistake)
        - 0.0: Invalid length or format
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


def post_process_vn_plate(text: str) -> str:
    """
    Post-process OCR result using Vietnamese license plate patterns.
    
    VN plate format: XXY-XXXXX (e.g., 51G-316.91, 60A-359.81)
    - Position 3 (index 2) is typically a LETTER (A-Z)
    - Other positions are typically NUMBERS (0-9)
    
    Args:
        text: Raw OCR text
    
    Returns:
        Post-processed text with position 3 corrections
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


def normalize_plate(text: Optional[str]) -> str:
    """
    Normalize Vietnamese license plate text.
    
    Removes special characters and fixes common OCR mistakes.
    Preserves valid characters (G, D) that are valid in VN plates.
    
    Args:
        text: Raw OCR text
    
    Returns:
        Normalized plate text (uppercase, alphanumeric only, with corrections)
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

