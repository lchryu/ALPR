import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from utils import (
    preprocess_plate,
    ocr_plate_complete,
    normalize_plate,
    is_two_line_plate,
    split_two_line_plate,
    reader
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
    # image_path = r"../data/test/images/car7.jpg"
    image_path = r"../data/test/images/test_xm4.jpg"

    # Use multi-pass OCR for better accuracy
    img, plates = detect_plate(image_path, use_multi_pass=True)

    # Print results
    print("\n" + "="*70)
    print(" " * 20 + "🎯 FINAL OCR RESULTS 🎯")
    print("="*70)
    for p in plates:
        raw = p['raw']
        plate = p['plate']
        # Show difference if any
        diff_indicator = "✓" if raw == plate else "→"
        print(
            f"\n📋 License Plate: {plate}"
            f"\n   Raw OCR: {raw} {diff_indicator} Normalized: {plate}"
            f"\n   Confidence: YOLO={p['conf']:.2f} | OCR={p['ocr_conf']:.2f}"
            f"\n   Method: {p['ocr_method']} | TwoLine: {p['two_line']}"
        )
    print("="*70 + "\n")

    # Visualize results - Clean, production-ready visualization with preprocessing steps
    for p in plates:
        crop = p["crop"]
        raw = p['raw']
        plate = p['plate']
        method = p['ocr_method']
        
        # Get preprocessing steps for visualization
        from utils import preprocess_plate, deskew_plate
        
        # Step 1: Original
        original = crop.copy()
        
        # Step 2: Deskewed (rotation correction)
        deskewed = deskew_plate(crop, angle_threshold=2.0, debug=False)
        
        h, w = deskewed.shape[:2]
        
        # Step 3: Upscaled
        scale = 4.0 if min(h, w) < 100 else 3.5
        upscaled = cv2.resize(deskewed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Step 4: Grayscale
        if len(upscaled.shape) == 3:
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        else:
            gray = upscaled.copy()
        
        # Step 5: With padding
        padding_top, padding_bottom, padding_left, padding_right = 20, 20, 20, 40
        gray_padded = cv2.copyMakeBorder(gray, padding_top, padding_bottom, padding_left, padding_right, 
                                        cv2.BORDER_CONSTANT, value=255)
        
        # Step 6: Denoised
        if min(gray_padded.shape) > 100:
            denoised = cv2.fastNlMeansDenoising(gray_padded, h=5, templateWindowSize=7, searchWindowSize=21)
        else:
            denoised = cv2.bilateralFilter(gray_padded, 3, 40, 40)
        
        # Step 7: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        clahe_applied = clahe.apply(denoised)
        
        # Step 8: Sharpened
        kernel = np.array([
            [0, -0.3, 0],
            [-0.3, 3.2, -0.3],
            [0, -0.3, 0]
        ])
        sharpened = cv2.filter2D(clahe_applied, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Step 9: Final contrast enhancement
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final = cv2.addWeighted(binary, 0.8, sharpened, 0.2, 0)
        final = np.clip(final, 0, 255).astype(np.uint8)
        
        # Get the preprocessed image for the BEST METHOD to visualize OCR blocks
        from utils import preprocess_plate
        
        # Handle different preprocessing methods
        if method == "inverted" or method == "inverted_high_contrast":
            if method == "inverted_high_contrast":
                best_preprocessed = preprocess_plate(crop, variant="high_contrast")
            else:
                best_preprocessed = preprocess_plate(crop, variant="standard")
            best_preprocessed = cv2.bitwise_not(best_preprocessed)
        elif method == "adaptive_thresh":
            h, w = crop.shape[:2]
            scale = 4.0 if min(h, w) < 100 else 3.5
            img_scale = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if len(img_scale.shape) == 3:
                gray_scale = cv2.cvtColor(img_scale, cv2.COLOR_BGR2GRAY)
            else:
                gray_scale = img_scale.copy()
            padding = 20
            gray_scale = cv2.copyMakeBorder(gray_scale, padding, padding, padding, padding, 
                                           cv2.BORDER_CONSTANT, value=255)
            best_preprocessed = cv2.adaptiveThreshold(
                gray_scale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, 5
            )
        elif method == "ultra_high_scale":
            h, w = crop.shape[:2]
            scale = 6.0 if min(h, w) < 100 else 5.0
            img_scale = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            if len(img_scale.shape) == 3:
                gray_scale = cv2.cvtColor(img_scale, cv2.COLOR_BGR2GRAY)
            else:
                gray_scale = img_scale.copy()
            padding = 30
            gray_scale = cv2.copyMakeBorder(gray_scale, padding, padding, padding, padding, 
                                           cv2.BORDER_CONSTANT, value=255)
            clahe_scale = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced_scale = clahe_scale.apply(gray_scale)
            kernel_scale = np.array([
                [0, -1, 0],
                [-1, 6, -1],
                [0, -1, 0]
            ])
            best_preprocessed = cv2.filter2D(enhanced_scale, -1, kernel_scale)
            best_preprocessed = np.clip(best_preprocessed, 0, 255).astype(np.uint8)
        else:
            # For other methods, use standard preprocessing with variant
            variant_map = {
                "standard": "standard",
                "standard_sensitive": "standard",
                "high_contrast": "high_contrast",
                "sharp": "sharp",
                "clean": "clean"
            }
            variant = variant_map.get(method, "standard")
            best_preprocessed = preprocess_plate(crop, variant=variant)
        
        # Get OCR results with bounding boxes using standard OCR parameters
        ocr_params = {
            'detail': 1,
            'paragraph': False,
            'width_ths': 0.4,  # Standard threshold (same as multi-pass)
            'height_ths': 0.4,
            'slope_ths': 0.1,
            'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        }
        ocr_results = reader.readtext(best_preprocessed, **ocr_params)
        
        # Figure 1: Full image with YOLO detection box
        fig1, axes1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Draw YOLO bounding box on original image
        img_with_yolo = img.copy()
        x1, y1, x2, y2 = p['bbox']
        cv2.rectangle(img_with_yolo, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add label
        label = f"Plate: {plate} (YOLO: {p['conf']:.2f})"
        cv2.putText(img_with_yolo, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        axes1.imshow(cv2.cvtColor(img_with_yolo, cv2.COLOR_BGR2RGB))
        axes1.set_title("YOLO Plate Detection", fontsize=14, fontweight='bold')
        axes1.axis("off")
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Preprocessing Pipeline Steps (9 steps)
        fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
        axes2 = axes2.flatten()
        
        steps = [
            ("1. Original", original, None),
            ("2. Deskewed", deskewed, None),
            ("3. Upscaled", upscaled, None),
            ("4. Grayscale", gray, "gray"),
            ("5. Padded", gray_padded, "gray"),
            ("6. Denoised", denoised, "gray"),
            ("7. CLAHE", clahe_applied, "gray"),
            ("8. Sharpened", sharpened, "gray"),
            ("9. Contrast Enhanced", final, "gray"),
        ]
        
        for i, (title, img_step, cmap) in enumerate(steps):
            try:
                # Convert to RGB if needed (for color images)
                if cmap is None and len(img_step.shape) == 3:
                    if img_step.shape[2] == 3:
                        # BGR to RGB
                        display_img = cv2.cvtColor(img_step, cv2.COLOR_BGR2RGB)
                    else:
                        display_img = img_step
                else:
                    display_img = img_step
                
                if cmap:
                    axes2[i].imshow(display_img, cmap=cmap)
                else:
                    axes2[i].imshow(display_img)
                axes2[i].set_title(title, fontsize=10, fontweight='bold')
                axes2[i].axis("off")
            except Exception as e:
                print(f"Error displaying {title}: {e}")
                axes2[i].text(0.5, 0.5, f"Error\n{title}", 
                             ha="center", va="center", fontsize=10)
                axes2[i].axis("off")
        
        plt.suptitle("Preprocessing Pipeline Steps", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Figure 3: OCR Results - Clean visualization with EasyOCR blocks only
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Original crop
        axes3[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes3[0].set_title("Original Plate Crop", fontsize=12, fontweight='bold')
        axes3[0].axis("off")
        
        # Right: Preprocessed image with EasyOCR bounding boxes
        result_img = best_preprocessed.copy()
        if len(result_img.shape) == 2:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
        
        # Draw EasyOCR bounding boxes (text blocks only)
        if ocr_results:
            for i, (bbox, text, conf_score) in enumerate(ocr_results):
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                pts = np.array(bbox, dtype=np.int32)
                
                # Color based on confidence
                if conf_score > 0.7:
                    color = (0, 255, 0)  # Green - high confidence
                elif conf_score > 0.5:
                    color = (255, 255, 0)  # Yellow - medium confidence
                else:
                    color = (255, 0, 0)  # Red - low confidence
                
                # Draw bounding box
                cv2.polylines(result_img, [pts], True, color, 3)
                
                # Get top-left corner for text label
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]
                x_min, y_min = min(x_coords), min(y_coords)
                
                # Draw text label
                label = f"'{text}' ({conf_score:.2f})"
                cv2.putText(result_img, label, (x_min, y_min - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        axes3[1].imshow(result_img)
        
        # Title with summary
        title_text = f"EasyOCR Detection: {len(ocr_results)} text block(s)"
        if ocr_results:
            total_chars = sum(len(r[1]) for r in ocr_results)
            title_text += f", {total_chars} characters"
        title_text += f"\nRecognized: {plate} | Method: {method}"
        
        axes3[1].set_title(title_text, fontsize=11, fontweight='bold')
        axes3[1].axis("off")
        
        plt.suptitle("OCR Detection Results", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
