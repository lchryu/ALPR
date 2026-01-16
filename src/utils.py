"""
Backwards-Compatible Re-exports

This module provides backwards-compatible imports for code that hasn't been
updated to use the new refactored modules.

DEPRECATED: New code should import directly from alpr.* modules:
  - from alpr.geometry import deskew_plate, is_two_line_plate
  - from alpr.ocr.waterfall import ocr_plate_complete
  - from alpr.validation import normalize_plate, validate_vn_plate_pattern
  - from alpr.debug_logger import DebugImageLogger

This file will be removed in a future version.
"""

import warnings

# Import from new modules
from alpr.geometry import (
    deskew_plate,
    is_two_line_plate,
    split_two_line_plate,
    crop_text_region,
    remove_plate_border,
    remove_border_contours
)

from alpr.ocr.waterfall import (
    ocr_plate,
    ocr_plate_complete
)

from alpr.validation import (
    normalize_plate,
    validate_vn_plate_pattern,
    post_process_vn_plate
)

from alpr.debug_logger import DebugImageLogger

# Import EasyOCR reader for backwards compatibility
from alpr.ocr.easyocr_engine import get_reader

# Create reader instance for backwards compatibility
# Note: Old code may have used `reader` directly
reader = get_reader()


def _deprecation_warning():
    """Show deprecation warning when utils module is imported."""
    warnings.warn(
        "Importing from utils is deprecated. "
        "Please update imports to use alpr.* modules directly. "
        "See module docstring for migration guide.",
        DeprecationWarning,
        stacklevel=3
    )


# Show warning on import
_deprecation_warning()

# Re-export all functions for backwards compatibility
__all__ = [
    # Geometry functions
    "deskew_plate",
    "is_two_line_plate",
    "split_two_line_plate",
    "crop_text_region",
    "remove_plate_border",
    "remove_border_contours",
    # OCR functions
    "ocr_plate",
    "ocr_plate_complete",
    # Validation functions
    "normalize_plate",
    "validate_vn_plate_pattern",
    "post_process_vn_plate",
    # Debug logger
    "DebugImageLogger",
    # EasyOCR reader (for backwards compatibility)
    "reader",
]
