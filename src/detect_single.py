import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from utils import (
    preprocess_plate,
    ocr_plate_complete,
    normalize_plate,
    is_two_line_plate,
    split_two_line_plate
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
    image_path = r"../data/test/images/test2.jpg"

    # Use multi-pass OCR for better accuracy
    img, plates = detect_plate(image_path, use_multi_pass=True)

    # Print results
    for p in plates:
        print(
            f"Plate: {p['plate']} | Raw: {p['raw']} | "
            f"YOLO Conf: {p['conf']:.2f} | OCR Conf: {p['ocr_conf']:.2f} | "
            f"Method: {p['ocr_method']} | TwoLine: {p['two_line']}"
        )

    # Visualize results
    for p in plates:
        crop = p["crop"]
        preprocessed = preprocess_plate(crop)

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        # Original crop
        ax[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Original Crop")
        ax[0].axis("off")

        # Preprocessed
        ax[1].imshow(preprocessed, cmap="gray")
        ax[1].set_title("Preprocessed (EasyOCR input)")
        ax[1].axis("off")

        # Results
        text_info = (
            f"Raw OCR: {p['raw']}\n"
            f"Normalized: {p['plate']}\n"
            f"YOLO Conf: {p['conf']:.2f}\n"
            f"OCR Conf: {p['ocr_conf']:.2f}\n"
            f"Method: {p['ocr_method']}\n"
            f"Two-line: {p['two_line']}"
        )
        ax[2].text(0.05, 0.5, text_info, fontsize=12, verticalalignment='center', family='monospace')
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()
