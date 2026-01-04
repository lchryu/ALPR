#!/usr/bin/env python3
"""
ALPR Pipeline Visualization Tool

Standalone visualization tool for debugging and observing each step of the ALPR pipeline.
Does NOT modify production code.

Usage:
    python visualize_pipeline.py --image path/to/image.jpg
    python visualize_pipeline.py --image path/to/image.jpg --out custom/output/dir
    python visualize_pipeline.py --image path/to/image.jpg --no-gui
"""

import argparse
import cv2
import json
import numpy as np
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Add src/ to path for imports
BASE_DIR = Path(__file__).parent.absolute()
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from ultralytics import YOLO
    from utils import (
        deskew_plate,
        normalize_plate,
        validate_vn_plate_pattern,
        is_two_line_plate,
        split_two_line_plate,
        reader,
    )
except ImportError as e:
    print(f"Error importing utilities: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)


@dataclass
class StepResult:
    """Store OCR pass results"""
    pass_num: int
    method: str
    raw_text: str
    confidence: float
    input_image: np.ndarray


@dataclass
class PipelineResult:
    """Store complete pipeline results"""
    det_conf: float
    pass_used: str
    raw_text: str
    normalized_plate: str
    pattern_ok: bool
    ocr_conf: float
    method: str
    two_line: bool
    bbox: Tuple[int, int, int, int]


class PipelineVisualizer:
    """Visualize ALPR pipeline step-by-step"""
    
    def __init__(self, output_dir: Path, show_gui: bool = True):
        self.output_dir = output_dir
        self.show_gui = show_gui
        self.step_counter = 0
        self.step_results = []
        self.preprocessing_saved = False  # Track if preprocessing steps already saved
        
        # Load YOLO model
        model_path = BASE_DIR / "models" / "best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(str(model_path))
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_step(self, name: str, image: np.ndarray, is_text: bool = False):
        """Save step with auto-numbering"""
        filename = f"{self.step_counter:02d}_{name}"
        filepath = self.output_dir / filename
        
        if is_text:
            filepath = filepath.with_suffix('.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(image)  # image is actually text content
        else:
            filepath = filepath.with_suffix('.jpg')
            cv2.imwrite(str(filepath), image)
            
            if self.show_gui:
                # Resize for display if too large
                h, w = image.shape[:2]
                if w > 1200 or h > 800:
                    scale = min(1200/w, 800/h)
                    display_img = cv2.resize(image, None, fx=scale, fy=scale)
                else:
                    display_img = image
                cv2.imshow(f"Step {self.step_counter}: {name}", display_img)
                cv2.waitKey(100)  # Brief pause
        
        self.step_counter += 1
        return filepath
    
    def visualize_detection(self, img: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """Step 1: YOLO detection"""
        # Save input
        self.save_step("input", img)
        
        # Run detection
        results = self.model(img)[0]
        
        if len(results.boxes) == 0:
            print("No plates detected!")
            return None
        
        # Get best detection (highest confidence)
        best_box = None
        best_conf = 0.0
        for box in results.boxes:
            conf = float(box.conf)
            if conf > best_conf and conf >= 0.4:
                best_conf = conf
                best_box = box
        
        if best_box is None:
            print("No valid detections (confidence < 0.4)")
            return None
        
        # Extract bbox
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
        bbox = (x1, y1, x2, y2)
        
        # Draw overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"Plate (conf: {best_conf:.2f})"
        cv2.putText(overlay, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self.save_step("detect_overlay", overlay)
        
        return img, bbox, best_conf
    
    def visualize_crop(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Step 2: Crop plate"""
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2].copy()
        self.save_step("crop", crop)
        return crop
    
    def visualize_deskew(self, crop: np.ndarray) -> np.ndarray:
        """Step 3: Deskew correction"""
        deskewed = deskew_plate(crop, angle_threshold=2.0, debug=False)
        self.save_step("deskew", deskewed)
        return deskewed
    
    def visualize_preprocessing_steps(self, img: np.ndarray, pass_num: int = 1) -> dict:
        """Visualize preprocessing steps for a specific pass"""
        steps = {}
        
        h, w = img.shape[:2]
        
        # Determine scale based on pass
        if pass_num == 1:
            scale = 3.5 if min(h, w) < 100 else 3.0
        elif pass_num == 2:
            scale = 4.5 if min(h, w) < 100 else 4.0
        else:  # pass 3
            scale = 6.0 if min(h, w) < 100 else 5.0
        
        # Resize
        img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        steps['resize'] = img_scaled
        
        # Convert to grayscale
        if len(img_scaled.shape) == 3:
            gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_scaled.copy()
        steps['gray'] = gray
        
        # Padding
        if pass_num == 1:
            padding = 15
        elif pass_num == 2:
            padding = 20
        else:
            padding = 30
        
        gray_padded = cv2.copyMakeBorder(gray, padding, padding, padding, padding,
                                        cv2.BORDER_CONSTANT, value=255)
        
        # Denoising
        if pass_num == 1:
            # No denoising for pass 1
            denoised = gray_padded
        elif pass_num == 2:
            if min(gray_padded.shape) > 100:
                denoised = cv2.fastNlMeansDenoising(gray_padded, h=5, templateWindowSize=7, searchWindowSize=21)
            else:
                denoised = cv2.bilateralFilter(gray_padded, 3, 40, 40)
        else:  # pass 3
            if min(gray_padded.shape) > 100:
                denoised = cv2.fastNlMeansDenoising(gray_padded, h=8, templateWindowSize=7, searchWindowSize=21)
            else:
                denoised = cv2.bilateralFilter(gray_padded, 5, 60, 60)
        steps['denoise'] = denoised
        
        # CLAHE (contrast)
        if pass_num == 1:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        elif pass_num == 2:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        else:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        steps['contrast'] = enhanced
        
        # Sharpening (only for pass 2 and 3)
        if pass_num == 1:
            sharp = enhanced
        elif pass_num == 2:
            kernel = np.array([
                [0, -0.3, 0],
                [-0.3, 3.2, -0.3],
                [0, -0.3, 0]
            ])
            sharp = cv2.filter2D(enhanced, -1, kernel)
            sharp = np.clip(sharp, 0, 255).astype(np.uint8)
        else:  # pass 3
            kernel = np.array([
                [0, -1, 0],
                [-1, 6, -1],
                [0, -1, 0]
            ])
            sharp = cv2.filter2D(enhanced, -1, kernel)
            sharp = np.clip(sharp, 0, 255).astype(np.uint8)
        
        # Threshold
        if pass_num == 2:
            # Adaptive threshold for pass 2
            threshold = cv2.adaptiveThreshold(
                sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 5
            )
        else:
            # Otsu threshold for pass 1 and 3
            _, threshold = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        steps['threshold'] = threshold
        
        # Morphology (optional - not used in current passes, but save if needed)
        # For pass 3, we might try inverted
        if pass_num == 3:
            inverted = cv2.bitwise_not(threshold)
            steps['invert_optional'] = inverted
        
        return steps
    
    def run_ocr_pass(self, img: np.ndarray, pass_num: int) -> Optional[StepResult]:
        """Run OCR pass and return result"""
        if pass_num == 1:
            # Pass 1: Clean
            img_deskewed = deskew_plate(img, angle_threshold=2.0, debug=False)
            h, w = img_deskewed.shape[:2]
            scale = 3.5 if min(h, w) < 100 else 3.0
            img_scaled = cv2.resize(img_deskewed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if len(img_scaled.shape) == 3:
                gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_scaled.copy()
            padding = 15
            gray = cv2.copyMakeBorder(gray, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_params = {
                'detail': 1, 'paragraph': False,
                'width_ths': 0.6, 'height_ths': 0.6, 'slope_ths': 0.1,
                'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            }
            method = "pass1_clean"
            
        elif pass_num == 2:
            # Pass 2: Robust
            img_deskewed = deskew_plate(img, angle_threshold=2.0, debug=False)
            h, w = img_deskewed.shape[:2]
            scale = 4.5 if min(h, w) < 100 else 4.0
            img_scaled = cv2.resize(img_deskewed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if len(img_scaled.shape) == 3:
                gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_scaled.copy()
            padding = 20
            gray = cv2.copyMakeBorder(gray, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
            if min(gray.shape) > 100:
                gray = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7, searchWindowSize=21)
            else:
                gray = cv2.bilateralFilter(gray, 3, 40, 40)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            kernel = np.array([[0, -0.3, 0], [-0.3, 3.2, -0.3], [0, -0.3, 0]])
            sharp = cv2.filter2D(enhanced, -1, kernel)
            sharp = np.clip(sharp, 0, 255).astype(np.uint8)
            binary = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
            ocr_params = {
                'detail': 1, 'paragraph': False,
                'width_ths': 0.4, 'height_ths': 0.4, 'slope_ths': 0.1,
                'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            }
            method = "pass2_robust"
            
        else:  # pass 3
            # Pass 3: Fallback
            img_deskewed = deskew_plate(img, angle_threshold=2.0, debug=False)
            h, w = img_deskewed.shape[:2]
            scale = 6.0 if min(h, w) < 100 else 5.0
            img_scaled = cv2.resize(img_deskewed, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            if len(img_scaled.shape) == 3:
                gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_scaled.copy()
            padding = 30
            gray = cv2.copyMakeBorder(gray, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
            if min(gray.shape) > 100:
                gray = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
            else:
                gray = cv2.bilateralFilter(gray, 5, 60, 60)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])
            sharp = cv2.filter2D(enhanced, -1, kernel)
            sharp = np.clip(sharp, 0, 255).astype(np.uint8)
            _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_params = {
                'detail': 1, 'paragraph': False,
                'width_ths': 0.3, 'height_ths': 0.3, 'slope_ths': 0.1,
                'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            }
            method = "pass3_fallback"
        
        # Run OCR
        results = reader.readtext(binary, **ocr_params)
        
        # Try inverted for pass 3 if no results
        if not results and pass_num == 3:
            inverted = cv2.bitwise_not(binary)
            results = reader.readtext(inverted, **ocr_params)
            if results:
                binary = inverted
        
        if not results:
            return None
        
        text = "".join([r[1] for r in results])
        confidence = np.mean([r[2] for r in results])
        
        return StepResult(
            pass_num=pass_num,
            method=method,
            raw_text=text,
            confidence=confidence,
            input_image=binary
        )
    
    def visualize_ocr_passes(self, crop: np.ndarray) -> PipelineResult:
        """Run 3-pass waterfall OCR and visualize each"""
        # Check if two-line plate
        is_two = is_two_line_plate(crop)
        
        if is_two:
            # Split into two lines
            top, bottom = split_two_line_plate(crop)
            # Process top line
            result_top = self._process_single_line(top)
            # Process bottom line  
            result_bottom = self._process_single_line(bottom)
            
            # Combine results
            if result_top and result_bottom:
                raw_text = result_top.raw_text + result_bottom.raw_text
                normalized = normalize_plate(raw_text)
                ocr_conf = (result_top.ocr_conf + result_bottom.ocr_conf) / 2
                method = f"{result_top.method}+{result_bottom.method}"
                pass_used = f"{result_top.pass_used}+{result_bottom.pass_used}"
            elif result_top:
                raw_text = result_top.raw_text
                normalized = normalize_plate(raw_text)
                ocr_conf = result_top.ocr_conf
                method = result_top.method
                pass_used = result_top.pass_used
            elif result_bottom:
                raw_text = result_bottom.raw_text
                normalized = normalize_plate(raw_text)
                ocr_conf = result_bottom.ocr_conf
                method = result_bottom.method
                pass_used = result_bottom.pass_used
            else:
                raw_text = ""
                normalized = ""
                ocr_conf = 0.0
                method = "none"
                pass_used = "none"
            
            pattern_score = validate_vn_plate_pattern(normalized)
            final_result = PipelineResult(
                det_conf=0.0,
                pass_used=pass_used,
                raw_text=raw_text,
                normalized_plate=normalized,
                pattern_ok=pattern_score >= 0.7,
                ocr_conf=ocr_conf,
                method=method,
                two_line=True,
                bbox=(0, 0, 0, 0)
            )
        else:
            # Single line - process normally
            final_result = self._process_single_line(crop)
        
        return final_result
    
    def _process_single_line(self, crop: np.ndarray) -> Optional[PipelineResult]:
        """Process a single line (for single-line or split two-line plates)"""
        # Visualize preprocessing steps (using pass 2 as representative for visualization)
        # Note: Each pass has different preprocessing, but we show pass 2 as example
        # Only save preprocessing steps once (for the first line in two-line case)
        if not self.preprocessing_saved:
            preproc_steps = self.visualize_preprocessing_steps(crop, pass_num=2)
            
            # Save preprocessing steps (steps 4-10)
            self.save_step("gray", preproc_steps['gray'])
            self.save_step("resize", preproc_steps['resize'])
            self.save_step("denoise", preproc_steps['denoise'])
            self.save_step("contrast", preproc_steps['contrast'])
            self.save_step("threshold", preproc_steps['threshold'])
            if 'invert_optional' in preproc_steps:
                self.save_step("invert_optional", preproc_steps['invert_optional'])
            
            self.preprocessing_saved = True
        
        # Run waterfall OCR passes
        final_result = None
        
        # Pass 1
        result1 = self.run_ocr_pass(crop, pass_num=1)
        if result1:
            self.save_step("pass1_input", result1.input_image)
            text_content = f"Raw: {result1.raw_text}\nConfidence: {result1.confidence:.3f}\nMethod: {result1.method}"
            self.save_step("pass1_ocr", text_content, is_text=True)
            
            normalized = normalize_plate(result1.raw_text)
            pattern_score = validate_vn_plate_pattern(normalized)
            if result1.confidence >= 0.6 and pattern_score >= 0.8:
                final_result = PipelineResult(
                    det_conf=0.0,  # Will be set later
                    pass_used="pass1_clean",
                    raw_text=result1.raw_text,
                    normalized_plate=normalized,
                    pattern_ok=pattern_score >= 0.7,
                    ocr_conf=result1.confidence,
                    method=result1.method,
                    two_line=False,
                    bbox=(0, 0, 0, 0)
                )
        
        # Pass 2 (if pass 1 didn't succeed)
        if final_result is None:
            result2 = self.run_ocr_pass(crop, pass_num=2)
            if result2:
                self.save_step("pass2_input", result2.input_image)
                text_content = f"Raw: {result2.raw_text}\nConfidence: {result2.confidence:.3f}\nMethod: {result2.method}"
                self.save_step("pass2_ocr", text_content, is_text=True)
                
                normalized = normalize_plate(result2.raw_text)
                pattern_score = validate_vn_plate_pattern(normalized)
                if result2.confidence >= 0.5 and pattern_score >= 0.7:
                    final_result = PipelineResult(
                        det_conf=0.0,
                        pass_used="pass2_robust",
                        raw_text=result2.raw_text,
                        normalized_plate=normalized,
                        pattern_ok=pattern_score >= 0.7,
                        ocr_conf=result2.confidence,
                        method=result2.method,
                        two_line=False,
                        bbox=(0, 0, 0, 0)
                    )
        
        # Pass 3 (fallback - always return if we have result)
        if final_result is None:
            result3 = self.run_ocr_pass(crop, pass_num=3)
            if result3:
                self.save_step("pass3_input", result3.input_image)
                text_content = f"Raw: {result3.raw_text}\nConfidence: {result3.confidence:.3f}\nMethod: {result3.method}"
                self.save_step("pass3_ocr", text_content, is_text=True)
                
                normalized = normalize_plate(result3.raw_text)
                pattern_score = validate_vn_plate_pattern(normalized)
                final_result = PipelineResult(
                    det_conf=0.0,
                    pass_used="pass3_fallback",
                    raw_text=result3.raw_text,
                    normalized_plate=normalized,
                    pattern_ok=pattern_score >= 0.7,
                    ocr_conf=result3.confidence,
                    method=result3.method,
                    two_line=False,
                    bbox=(0, 0, 0, 0)
                )
        
        if final_result is None:
            # All passes failed
            final_result = PipelineResult(
                det_conf=0.0,
                pass_used="none",
                raw_text="",
                normalized_plate="",
                pattern_ok=False,
                ocr_conf=0.0,
                method="none",
                two_line=False,
                bbox=(0, 0, 0, 0)
            )
        
        return final_result
    
    def visualize_final_overlay(self, img: np.ndarray, bbox: Tuple[int, int, int, int], 
                               result: PipelineResult) -> np.ndarray:
        """Draw final result on image"""
        overlay = img.copy()
        x1, y1, x2, y2 = bbox
        
        # Draw bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw text
        label = f"{result.normalized_plate} (conf: {result.ocr_conf:.2f})"
        cv2.putText(overlay, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self.save_step("final_overlay", overlay)
        return overlay
    
    def save_summary(self, result: PipelineResult):
        """Save summary.json"""
        summary = {
            "det_conf": result.det_conf,
            "pass_used": result.pass_used,
            "raw_text": result.raw_text,
            "normalized_plate": result.normalized_plate,
            "pattern_ok": result.pattern_ok,
            "ocr_conf": result.ocr_conf,
            "method": result.method,
            "two_line": result.two_line,
        }
        
        summary_path = self.output_dir / "18_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def print_summary(self, result: PipelineResult):
        """Print console summary table"""
        print("\n" + "="*80)
        print("ALPR PIPELINE SUMMARY")
        print("="*80)
        print(f"{'det_conf':<12} {'pass_used':<15} {'raw':<20} {'normalized':<15} {'pattern_ok':<10} {'ocr_conf':<10} {'method':<20}")
        print("-"*80)
        pattern_str = "✓" if result.pattern_ok else "✗"
        print(f"{result.det_conf:<12.3f} {result.pass_used:<15} {result.raw_text:<20} {result.normalized_plate:<15} {pattern_str:<10} {result.ocr_conf:<10.3f} {result.method:<20}")
        print("="*80)
        print(f"\nOutput saved to: {self.output_dir}")
        print("="*80 + "\n")
    
    def run(self, image_path: str):
        """Run complete visualization pipeline"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Reset step counter and flags
        self.step_counter = 0
        self.preprocessing_saved = False
        
        # Step 1: Detection
        detection_result = self.visualize_detection(img)
        if detection_result is None:
            print("No valid detection found. Exiting.")
            return
        
        img, bbox, det_conf = detection_result
        
        # Step 2: Crop
        crop = self.visualize_crop(img, bbox)
        
        # Step 3: Deskew
        deskewed = self.visualize_deskew(crop)
        
        # Step 4-10: Preprocessing + OCR passes
        result = self.visualize_ocr_passes(deskewed)
        
        # Update result with detection info
        result.det_conf = det_conf
        result.bbox = bbox
        result.two_line = is_two_line_plate(crop)
        
        # Step 17: Final overlay
        self.visualize_final_overlay(img, bbox, result)
        
        # Save summary
        self.save_summary(result)
        
        # Print summary
        self.print_summary(result)
        
        if self.show_gui:
            print("\nPress any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize ALPR pipeline step-by-step")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="runs", help="Output directory (default: runs/)")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI display (save images only)")
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out) / f"vis_{timestamp}"
    
    # Create visualizer
    visualizer = PipelineVisualizer(output_dir, show_gui=not args.no_gui)
    
    # Run pipeline
    try:
        visualizer.run(args.image)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

