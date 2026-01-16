"""
EasyOCR Engine: Singleton Reader Instance

Creates and manages a single EasyOCR reader instance for reuse across all OCR operations.
"""

import easyocr
from typing import Optional

# Global reader instance (singleton pattern)
_reader: Optional[easyocr.Reader] = None


def get_reader() -> easyocr.Reader:
    """
    Get or create the singleton EasyOCR reader instance.
    
    The reader is initialized once on first call and reused for all subsequent calls.
    This avoids the overhead of reloading the model multiple times.
    
    Returns:
        EasyOCR Reader instance
    """
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'])
    return _reader


def reset_reader():
    """
    Reset the global reader instance (for testing purposes).
    
    This clears the singleton, forcing a new reader to be created on next get_reader() call.
    """
    global _reader
    _reader = None

