"""
Single Image Detection Script

CLI tool for testing ALPR on a single image.
Uses the refactored pipeline modules.
"""

import sys
import os
from pathlib import Path

# Add src/ to Python path
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from alpr.pipeline import run_alpr_on_image
from alpr.debug_logger import DebugImageLogger

# Load YOLO
MODEL_PATH = BASE_DIR / "models" / "best.pt"
model = YOLO(str(MODEL_PATH))


def detect_plate(image_path: str, use_multi_pass: bool = True, debug: bool = False):
    """
    Detect license plates and perform OCR.
    
    Args:
        image_path: Path to input image
        use_multi_pass: Whether to use multi-pass OCR for better accuracy
        debug: Enable debug image logging
    
    Returns:
        img: Original image
        outputs: List of detected plates with OCR results
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Create debug logger if enabled
    logger = None
    if debug:
        debug_dir = BASE_DIR / "runs" / "debug"
        logger = DebugImageLogger(enabled=True, root_dir=str(debug_dir))
    
    # Run ALPR pipeline
    result = run_alpr_on_image(img, model, debug_logger=logger)
    
    outputs = []
    for r in result["results"]:
        outputs.append({
            "bbox": r["bbox"],
            "plate": r["plate"],
            "raw": r["raw"],
            "det_conf": r["det_conf"],
            "ocr_conf": r["ocr_conf"],
            "method": r["method"],
            "two_line": r["two_line"]
        })
    
    return img, outputs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_single.py <image_path> [--debug]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug = "--debug" in sys.argv
    
    try:
        img, outputs = detect_plate(image_path, debug=debug)
        
        print(f"\n{'='*60}")
        print(f"Detection Results for: {image_path}")
        print(f"{'='*60}")
        
        if not outputs:
            print("No plates detected.")
        else:
            for i, output in enumerate(outputs, 1):
                print(f"\nPlate {i}:")
                print(f"  BBox: {output['bbox']}")
                print(f"  Plate: {output['plate']}")
                print(f"  Raw: {output['raw']}")
                print(f"  Det Conf: {output['det_conf']:.3f}")
                print(f"  OCR Conf: {output['ocr_conf']:.3f}")
                print(f"  Method: {output['method']}")
                print(f"  Two-line: {output['two_line']}")
        
        # Visualize results
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"ALPR Results: {Path(image_path).name}")
        
        for output in outputs:
            x1, y1, x2, y2 = output['bbox']
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'g-', linewidth=2)
            plt.text(x1, y1 - 10, f"{output['plate']} ({output['ocr_conf']:.2f})",
                    color='green', fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

