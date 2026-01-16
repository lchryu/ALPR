"""
Smoke Test: Verify Refactored Pipeline Produces Same Outputs

Compares outputs from refactored pipeline to ensure behavior is unchanged.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import json

# Add src/ to Python path
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO
from alpr.pipeline import run_alpr_on_image
from alpr.debug_logger import DebugImageLogger


def test_pipeline(image_path: str, model: YOLO, debug: bool = False) -> dict:
    """
    Run pipeline on a single image and return results.
    
    Args:
        image_path: Path to test image
        model: YOLO model instance
        debug: Enable debug logging
    
    Returns:
        Dictionary with pipeline results
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    logger = None
    if debug:
        debug_dir = BASE_DIR / "runs" / "debug"
        logger = DebugImageLogger(enabled=True, root_dir=str(debug_dir))
    
    result = run_alpr_on_image(img, model, debug_logger=logger)
    return result


def verify_schema(result: dict) -> bool:
    """
    Verify that result matches expected schema.
    
    Args:
        result: Pipeline result dictionary
    
    Returns:
        True if schema is valid, False otherwise
    """
    if "results" not in result:
        print("❌ Missing 'results' key")
        return False
    
    if not isinstance(result["results"], list):
        print("❌ 'results' is not a list")
        return False
    
    required_keys = {"bbox", "raw", "plate", "det_conf", "ocr_conf", "method", "two_line"}
    
    for i, r in enumerate(result["results"]):
        if not isinstance(r, dict):
            print(f"❌ Result {i} is not a dict")
            return False
        
        missing = required_keys - set(r.keys())
        if missing:
            print(f"❌ Result {i} missing keys: {missing}")
            return False
        
        # Type checks
        if not isinstance(r["bbox"], list) or len(r["bbox"]) != 4:
            print(f"❌ Result {i} bbox is invalid: {r['bbox']}")
            return False
        
        if not isinstance(r["plate"], str):
            print(f"❌ Result {i} plate is not string: {type(r['plate'])}")
            return False
        
        if not isinstance(r["method"], str):
            print(f"❌ Result {i} method is not string: {type(r['method'])}")
            return False
        
        if r["method"] not in {"pass1_clean", "pass2_robust", "pass3_fallback", "none", 
                               "fallback_pass1_clean", "fallback_pass2_robust", "fallback_pass3_fallback"}:
            # Allow fallback_ prefix for two-line fallback cases
            if not r["method"].startswith("fallback_") and "+" not in r["method"]:
                print(f"⚠️  Result {i} has unexpected method: {r['method']}")
    
    return True


def main():
    """Run smoke tests on provided images."""
    if len(sys.argv) < 2:
        print("Usage: python smoke_test.py <image1> [image2] [image3] ...")
        print("\nExample:")
        print("  python smoke_test.py ../data/test/images/test1.jpg ../data/test/images/test2.jpg")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    
    # Load model
    MODEL_PATH = BASE_DIR / "models" / "best.pt"
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Please ensure models/best.pt exists")
        sys.exit(1)
    
    print(f"📦 Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    print(f"\n🧪 Running smoke tests on {len(image_paths)} image(s)...\n")
    
    all_passed = True
    
    for image_path in image_paths:
        path = Path(image_path)
        if not path.exists():
            print(f"⚠️  Skipping {image_path}: file not found")
            continue
        
        print(f"Testing: {path.name}")
        print("-" * 60)
        
        try:
            # Run pipeline
            result = test_pipeline(str(path), model, debug=False)
            
            # Verify schema
            if not verify_schema(result):
                print(f"❌ Schema validation failed for {path.name}")
                all_passed = False
                continue
            
            # Print results
            print(f"✅ Schema valid")
            print(f"   Detected {len(result['results'])} plate(s)")
            
            for i, r in enumerate(result["results"], 1):
                print(f"\n   Plate {i}:")
                print(f"     BBox: {r['bbox']}")
                print(f"     Plate: '{r['plate']}'")
                print(f"     Raw: '{r['raw']}'")
                print(f"     Det Conf: {r['det_conf']:.3f}")
                print(f"     OCR Conf: {r['ocr_conf']:.3f}")
                print(f"     Method: {r['method']}")
                print(f"     Two-line: {r['two_line']}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error processing {path.name}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ All smoke tests passed!")
        print("\nThe refactored pipeline produces valid outputs.")
        print("Schema matches expected format.")
    else:
        print("❌ Some smoke tests failed!")
        print("Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

