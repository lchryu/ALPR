import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from utils import (
    preprocess_plate,
    ocr_plate_complete,
    normalize_plate,
    is_two_line_plate,
    split_two_line_plate,
    validate_vn_plate_pattern,
    detect_individual_characters
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
        
        # IMPORTANT: Get the preprocessed image for the BEST METHOD (not the visualization final)
        # This ensures BBox detection uses the same image that gave the best OCR result
        from utils import preprocess_plate
        if method == "inverted" or method == "inverted_high_contrast":
            # For inverted methods, we need to invert the preprocessed image
            if method == "inverted_high_contrast":
                best_preprocessed = preprocess_plate(crop, variant="high_contrast")
            else:
                best_preprocessed = preprocess_plate(crop, variant="standard")
            best_preprocessed = cv2.bitwise_not(best_preprocessed)
        elif method == "adaptive_thresh":
            # For adaptive threshold, use special preprocessing
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
            # For ultra high scale
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
        
        # Get OCR results with bounding boxes using the SAME preprocessed image as best method
        from utils import reader
        ocr_params_bbox = {
            'detail': 1,
            'paragraph': False,
            'width_ths': 0.1,  # Very low to detect individual characters
            'height_ths': 0.1,  # Very low to detect individual characters
            'slope_ths': 0.1,
            'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        }
        ocr_results = reader.readtext(best_preprocessed, **ocr_params_bbox)
        
        # DEBUG: Also try OCR with the same method that was selected
        # to see if we get different results
        print(f"\n=== DEBUG INFO ===")
        print(f"Best method: {method}")
        print(f"Raw OCR result: {raw}")
        print(f"Normalized result: {plate}")
        print(f"BBox detection (params: width_ths=0.1): {len(ocr_results)} blocks")
        if ocr_results:
            bbox_texts = [r[1] for r in ocr_results]
            print(f"BBox texts: {bbox_texts}")
            print(f"BBox combined: {''.join(bbox_texts)}")
        
        # Try OCR with same params as used in multi-pass
        ocr_params_main = {
            'detail': 1,
            'paragraph': False,
            'width_ths': 0.4,  # Same as in multi-pass
            'height_ths': 0.4,
            'slope_ths': 0.1,
            'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        }
        ocr_results_main = reader.readtext(final, **ocr_params_main)
        if ocr_results_main:
            main_texts = [r[1] for r in ocr_results_main]
            print(f"Main OCR (params: width_ths=0.4): {len(ocr_results_main)} blocks")
            print(f"Main texts: {main_texts}")
            print(f"Main combined: {''.join(main_texts)}")
        
        # Check pattern validation
        from utils import validate_vn_plate_pattern, normalize_plate
        print(f"\nPattern validation for '{raw}': {validate_vn_plate_pattern(raw)}")
        print(f"After normalize: {normalize_plate(raw)}")
        print(f"Expected: 51G31691")
        print("=" * 50)
        
        # Figure 1: Preprocessing steps
        fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
        axes1 = axes1.flatten()
        
        steps = [
            ("1. Original", original, None),
            ("2. Upscaled", upscaled, None),
            ("3. Grayscale", gray, "gray"),
            ("4. Padded", gray_padded, "gray"),
            ("5. Denoised", denoised, "gray"),
            ("6. CLAHE", clahe_applied, "gray"),
            ("7. Sharpened", sharpened, "gray"),
            ("8. Contrast Enhanced", final, "gray"),
        ]
        
        for i, (title, img, cmap) in enumerate(steps):
            try:
                # Convert to RGB if needed (for color images)
                if cmap is None and len(img.shape) == 3:
                    if img.shape[2] == 3:
                        # BGR to RGB
                        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        display_img = img
                else:
                    display_img = img
                
                if cmap:
                    axes1[i].imshow(display_img, cmap=cmap)
                else:
                    axes1[i].imshow(display_img)
                axes1[i].set_title(title, fontsize=10)
                axes1[i].axis("off")
            except Exception as e:
                print(f"Error displaying {title}: {e}")
                axes1[i].text(0.5, 0.5, f"Error\n{title}", 
                             ha="center", va="center", fontsize=10)
                axes1[i].axis("off")
        
        plt.suptitle("Preprocessing Pipeline Steps", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Detect individual characters using contour detection FIRST
        individual_chars = detect_individual_characters(best_preprocessed, debug=True)
        print(f"Contour detection: Found {len(individual_chars)} individual characters")
        
        # Figure 3: Contour detection steps (for debugging) - Show BEFORE OCR results
        fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
        
        # Show binary image used for contour detection
        mean_val = np.mean(best_preprocessed)
        is_inverted = mean_val < 127
        if is_inverted:
            _, binary_vis = cv2.threshold(best_preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary_vis = cv2.threshold(best_preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        binary_sep = cv2.morphologyEx(binary_vis, cv2.MORPH_OPEN, kernel_h, iterations=1)
        
        axes3[0].imshow(best_preprocessed, cmap="gray")
        axes3[0].set_title("Preprocessed Image")
        axes3[0].axis("off")
        
        axes3[1].imshow(binary_vis, cmap="gray")
        axes3[1].set_title("Binary (for contours)")
        axes3[1].axis("off")
        
        axes3[2].imshow(binary_sep, cmap="gray")
        axes3[2].set_title("After Morph Open\n(to separate chars)")
        axes3[2].axis("off")
        
        plt.suptitle("Contour Detection Steps", fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Figure 2: OCR Results with Bounding Boxes
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        
        # Original crop
        axes2[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes2[0].set_title("Original Crop")
        axes2[0].axis("off")
        
        # Final preprocessed with bounding boxes (use best method's preprocessed image)
        final_with_boxes = best_preprocessed.copy()
        if len(final_with_boxes.shape) == 2:
            final_with_boxes = cv2.cvtColor(final_with_boxes, cv2.COLOR_GRAY2RGB)
        
        # Draw EasyOCR bounding boxes (text blocks) - GREEN boxes
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
                
                # Draw bounding box (thicker for text blocks)
                cv2.polylines(final_with_boxes, [pts], True, color, 3)
                
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
        
        # Draw individual character bounding boxes (from contour detection) - MAGENTA boxes
        for i, (x, y, w, h, char_img) in enumerate(individual_chars):
            # Draw thicker boxes for individual characters (magenta) to make them visible
            cv2.rectangle(final_with_boxes, (x, y), (x+w, y+h), (255, 0, 255), 2)  # Magenta, thicker
            cv2.putText(final_with_boxes, f"C{i}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Add warning if only 1 box detected (might be reading as single block)
        title_text = f"EasyOCR: {len(ocr_results)} block(s), {total_chars_detected} chars"
        title_text += f" | Contour: {len(individual_chars)} chars"
        if len(ocr_results) == 1 and total_chars_detected > 1:
            title_text += "\n⚠️ Single block (EasyOCR grouped chars)"
        
        axes2[1].imshow(final_with_boxes)
        axes2[1].set_title(f"Detection Results\n{title_text}\n(Method: {method})", fontsize=9)
        axes2[1].axis("off")
        
        plt.suptitle("OCR Detection Results", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Figure 4: Individual characters detected by contour
        if individual_chars:
            n_chars = len(individual_chars)
            cols = min(8, n_chars)
            rows = (n_chars + cols - 1) // cols
            
            fig3, axes3 = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
            
            # Ensure axes3 is always a flat list/array
            if rows == 1 and cols == 1:
                axes3 = [axes3]
            elif rows == 1:
                axes3 = axes3.flatten() if hasattr(axes3, 'flatten') else list(axes3)
            elif cols == 1:
                axes3 = axes3.flatten() if hasattr(axes3, 'flatten') else list(axes3)
            else:
                axes3 = axes3.flatten()
            
            for i, (x, y, w, h, char_img) in enumerate(individual_chars):
                if i < len(axes3):
                    axes3[i].imshow(char_img, cmap="gray")
                    axes3[i].set_title(f"Char {i}\n({w}x{h})", fontsize=8)
                    axes3[i].axis("off")
            
            # Hide unused subplots
            for i in range(n_chars, len(axes3)):
                axes3[i].axis("off")
            
            plt.suptitle(f"Individual Characters Detected by Contour ({n_chars} chars)", fontsize=12)
            plt.tight_layout()
            plt.show()
            
            print(f"\n=== Individual Characters (Contour Detection) ===")
            print(f"Detected {len(individual_chars)} characters:")
            for i, (x, y, w, h, _) in enumerate(individual_chars):
                print(f"  Char {i}: BBox=({x},{y},{w},{h})")
        else:
            print(f"\n⚠️  Contour detection found 0 characters!")
            print(f"   This might be because:")
            print(f"   1. Characters are too connected (morphological operations didn't separate them)")
            print(f"   2. Filtering thresholds are too strict")
            print(f"   3. Preprocessing made characters merge together")
        
        # Results summary (printed to console, not displayed in figure)
        # All OCR attempts are already printed in debug output above
