import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from utils import (
    preprocess_plate,
    ocr_plate_complete,
    normalize_plate,
    is_two_line_plate,
    split_two_line_plate,
    validate_vn_plate_pattern
)

# Load YOLO
model = YOLO("../models/best.pt")


def detect_plate(image_path, use_multi_pass=True):
    """
    Detect license plates and perform OCR.
    
    Args:
        image_path: Path to input image
        use_multi_pass: Whether to use multi-pass OCR for better accuracy
    
    Returns:
        img: Original image
        outputs: List of detected plates with OCR results
    """
    img = cv2.imread(image_path)
    results = model(img)[0]

    outputs = []

    for box in results.boxes:
        conf = float(box.conf)
        if conf < 0.4:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop = img[y1:y2, x1:x2]

        # Check if 2-line plate
        is_two = is_two_line_plate(crop)

        if is_two:
            # Split into 2 lines
            top, bottom = split_two_line_plate(crop)

            # OCR each line separately
            top_raw, top_norm, top_conf, top_method = ocr_plate_complete(top, use_multi_pass=use_multi_pass)
            bottom_raw, bottom_norm, bottom_conf, bottom_method = ocr_plate_complete(bottom, use_multi_pass=use_multi_pass)

            # Combine results
            raw = top_raw + bottom_raw
            plate = normalize_plate(raw)  # Normalize combined text
            avg_conf = (top_conf + bottom_conf) / 2 if (top_conf > 0 and bottom_conf > 0) else max(top_conf, bottom_conf)
            method = f"{top_method}+{bottom_method}"

        else:
            # Single line plate
            raw, plate, conf, method = ocr_plate_complete(crop, use_multi_pass=use_multi_pass)
            avg_conf = conf

        outputs.append({
            "bbox": (x1, y1, x2, y2),
            "crop": crop,
            "raw": raw,
            "plate": plate,
            "conf": conf,  # YOLO detection confidence
            "ocr_conf": avg_conf,  # OCR confidence
            "ocr_method": method,
            "two_line": is_two
        })

    return img, outputs


if __name__ == "__main__":
    image_path = r"../data/test/images/test.jpg"

    # Use multi-pass OCR for better accuracy
    img, plates = detect_plate(image_path, use_multi_pass=True)

    # Print results
    for p in plates:
        raw = p['raw']
        plate = p['plate']
        # Show difference if any
        diff_indicator = "✓" if raw == plate else "→"
        print(
            f"Raw OCR: {raw} {diff_indicator} Normalized: {plate} | "
            f"YOLO Conf: {p['conf']:.2f} | OCR Conf: {p['ocr_conf']:.2f} | "
            f"Method: {p['ocr_method']} | TwoLine: {p['two_line']}"
        )

    # Visualize results with full pipeline steps
    for p in plates:
        crop = p["crop"]
        
        # Get all preprocessing steps
        import numpy as np
        h, w = crop.shape[:2]
        
        # Step 1: Original
        original = crop.copy()
        
        # Step 2: Upscaled
        scale = 4.0 if min(h, w) < 100 else 3.5
        upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Step 3: Grayscale
        if len(upscaled.shape) == 3:
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        else:
            gray = upscaled.copy()
        
        # Step 4: With padding
        padding_top, padding_bottom, padding_left, padding_right = 20, 20, 20, 40
        gray_padded = cv2.copyMakeBorder(gray, padding_top, padding_bottom, padding_left, padding_right, 
                                        cv2.BORDER_CONSTANT, value=255)
        
        # Step 5: Denoised
        if min(gray_padded.shape) > 100:
            denoised = cv2.fastNlMeansDenoising(gray_padded, h=5, templateWindowSize=7, searchWindowSize=21)
        else:
            denoised = cv2.bilateralFilter(gray_padded, 3, 40, 40)
        
        # Step 6: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        clahe_applied = clahe.apply(denoised)
        
        # Step 7: Sharpened
        kernel = np.array([
            [0, -0.3, 0],
            [-0.3, 3.2, -0.3],
            [0, -0.3, 0]
        ])
        sharpened = cv2.filter2D(clahe_applied, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Step 8: Final contrast enhancement (make text darker, background brighter)
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final = cv2.addWeighted(binary, 0.8, sharpened, 0.2, 0)
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        # Get all OCR attempts if available
        try:
            raw, plate, conf, method, all_attempts = ocr_plate_complete(crop, use_multi_pass=True, return_all_attempts=True)
        except:
            all_attempts = []
            raw = p['raw']
            plate = p['plate']
            conf = p['ocr_conf']
            method = p['ocr_method']
        
        # Get OCR results with bounding boxes for the best method
        from utils import reader
        # Use more sensitive parameters to detect individual characters
        # Lower width_ths and height_ths to detect smaller text blocks (individual chars)
        ocr_params_bbox = {
            'detail': 1,
            'paragraph': False,
            'width_ths': 0.1,  # Very low to detect individual characters
            'height_ths': 0.1,  # Very low to detect individual characters
            'slope_ths': 0.1,
            'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        }
        ocr_results = reader.readtext(final, **ocr_params_bbox)
        
        # Figure 1: Preprocessing steps
        fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
        axes1 = axes1.flatten()
        
        steps = [
            ("1. Original", cv2.cvtColor(original, cv2.COLOR_BGR2RGB), None),
            ("2. Upscaled", cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB), None),
            ("3. Grayscale", gray, "gray"),
            ("4. Padded", gray_padded, "gray"),
            ("5. Denoised", denoised, "gray"),
            ("6. CLAHE", clahe_applied, "gray"),
            ("7. Sharpened", sharpened, "gray"),
            ("8. Contrast Enhanced", final, "gray"),
        ]
        
        for i, (title, img, cmap) in enumerate(steps):
            if cmap:
                axes1[i].imshow(img, cmap=cmap)
            else:
                axes1[i].imshow(img)
            axes1[i].set_title(title, fontsize=10)
            axes1[i].axis("off")
        
        plt.suptitle("Preprocessing Pipeline Steps", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Figure 2: OCR Results with Bounding Boxes
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
        
        # Original crop
        axes2[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes2[0].set_title("Original Crop")
        axes2[0].axis("off")
        
        # Final preprocessed with bounding boxes
        final_with_boxes = final.copy()
        if len(final_with_boxes.shape) == 2:
            final_with_boxes = cv2.cvtColor(final_with_boxes, cv2.COLOR_GRAY2RGB)
        
        # Draw bounding boxes and text
        detected_chars = []
        total_chars_detected = 0
        if ocr_results:
            for i, (bbox, text, conf_score) in enumerate(ocr_results):
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                pts = np.array(bbox, dtype=np.int32)
                
                # Determine color based on confidence and text length
                if conf_score > 0.7:
                    color = (0, 255, 0)  # Green - high confidence
                elif conf_score > 0.5:
                    color = (255, 255, 0)  # Yellow - medium confidence
                else:
                    color = (255, 0, 0)  # Red - low confidence
                
                # Draw bounding box
                cv2.polylines(final_with_boxes, [pts], True, color, 2)
                
                # Get top-left corner for text label
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]
                x_min, y_min = min(x_coords), min(y_coords)
                
                # Count characters in this text block
                char_count = len(text)
                total_chars_detected += char_count
                
                # Draw text label with more info
                label = f"Block {i}: '{text}' ({char_count} chars, conf: {conf_score:.2f})"
                cv2.putText(final_with_boxes, label, (x_min, y_min - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                detected_chars.append((text, conf_score, bbox, char_count))
        
        # Add warning if only 1 box detected (might be reading as single block)
        title_text = f"Detected: {len(ocr_results)} text block(s), {total_chars_detected} total chars"
        if len(ocr_results) == 1 and total_chars_detected > 1:
            title_text += "\n⚠️ Single block detected (may need character segmentation)"
        
        axes2[1].imshow(final_with_boxes)
        axes2[1].set_title(f"EasyOCR Detection\n{title_text}\n(Method: {method})", fontsize=9)
        axes2[1].axis("off")
        
        # Results with all attempts
        result_text = (
            f"Raw OCR: {raw}\n"
            f"Normalized: {plate}\n"
            f"YOLO Conf: {p['conf']:.2f}\n"
            f"OCR Conf: {conf:.2f}\n"
            f"Method: {method}\n"
            f"Two-line: {p['two_line']}\n"
        )
        
        if all_attempts:
            result_text += f"\n--- All OCR Attempts (with Pattern Score) ---\n"
            for attempt_text, attempt_conf, attempt_method in all_attempts:
                marker = "✓" if attempt_method == method else " "
                pattern_score = validate_vn_plate_pattern(attempt_text)
                pattern_indicator = "✓" if pattern_score >= 0.7 else "✗" if pattern_score < 0.5 else "~"
                result_text += f"{marker} {attempt_method}: {attempt_text}\n"
                result_text += f"   conf: {attempt_conf:.2f} | pattern: {pattern_score:.2f} {pattern_indicator}\n"
        
        axes2[2].text(0.05, 0.5, result_text, fontsize=10, verticalalignment='center', 
                     family='monospace', transform=axes2[2].transAxes)
        axes2[2].axis("off")
        
        plt.tight_layout()
        plt.show()
