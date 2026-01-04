# Debug Image Logger - Usage Guide

## Overview

The `DebugImageLogger` is **pure instrumentation** - it does NOT modify pipeline logic, images, return values, or control flow. When `enabled=False`, it has zero side effects.

## Files

- **`src/debug_logger.py`**: Debug logger module
- **Modified functions in `src/utils.py`**: Added optional `logger=None` parameter
- **Modified `api/main.py`**: Added `debug` query parameter

## Usage Examples

### Example 1: API with Debug Enabled

```python
# API call with debug enabled
# POST /alpr?debug=true

import requests

with open("test_image.jpg", "rb") as f:
    files = {"file": ("test.jpg", f, "image/jpeg")}
    response = requests.post(
        "http://localhost:8000/alpr?debug=true",
        files=files
    )
    print(response.json())
```

**Output**: Images saved to `runs/debug/debug_<timestamp>/`

### Example 2: API with Debug Disabled (Default)

```python
# API call without debug (default behavior - identical to before)
# POST /alpr

import requests

with open("test_image.jpg", "rb") as f:
    files = {"file": ("test.jpg", f, "image/jpeg")}
    response = requests.post(
        "http://localhost:8000/alpr",  # No debug parameter
        files=files
    )
    print(response.json())
```

**Output**: No images saved, behavior identical to original API

### Example 3: Direct Function Call with Logger

```python
import cv2
from src.debug_logger import DebugImageLogger
from src.utils import ocr_plate_complete

# Load image
img = cv2.imread("plate_crop.jpg")

# Create logger (enabled)
logger = DebugImageLogger(enabled=True)

# Run OCR with logging
raw, normalized, conf, method = ocr_plate_complete(img, logger=logger)

print(f"Result: {normalized} (confidence: {conf:.2f})")
# Images saved to runs/debug/debug_<timestamp>/
```

### Example 4: Direct Function Call without Logger

```python
import cv2
from src.utils import ocr_plate_complete

# Load image
img = cv2.imread("plate_crop.jpg")

# Run OCR without logging (default behavior)
raw, normalized, conf, method = ocr_plate_complete(img)  # logger=None by default

print(f"Result: {normalized} (confidence: {conf:.2f})")
# No images saved, behavior identical to original
```

## Logged Images

When debug is enabled, the following images are saved (in order):

1. `000_input.jpg` - Original input image (API only)
2. `001_crop.jpg` - Cropped plate region (API only)
3. `002_deskew.jpg` - Deskewed/corrected image
4. `003_gray.jpg` - Grayscale conversion
5. `004_contrast.jpg` - CLAHE contrast enhancement
6. `005_threshold.jpg` - Binary threshold
7. `006_preprocessed_final.jpg` - Final preprocessed image
8. `007_pass1_input.jpg` - OCR Pass 1 input
9. `008_pass2_input.jpg` - OCR Pass 2 input (if Pass 1 fails)
10. `009_pass3_input.jpg` - OCR Pass 3 input (if Pass 2 fails)
11. `010_pass3_input_inverted.jpg` - Inverted image for Pass 3 (if needed)

## Important Notes

- **Zero side effects when disabled**: When `enabled=False`, the logger does nothing
- **No logic changes**: All pipeline logic remains identical
- **Production safe**: Default behavior (`debug=False`) is identical to original code
- **Instrumentation only**: This is for debugging, not for modifying pipeline behavior

