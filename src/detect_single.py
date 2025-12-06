import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from utils import (
    preprocess,
    ocr_text,
    normalize,
    is_two_line_plate,
    split_two_line_plate
)

# Load YOLO
model = YOLO("../models/best.pt")


def detect_plate(image_path):
    img = cv2.imread(image_path)
    results = model(img)[0]

    outputs = []

    for box in results.boxes:
        conf = float(box.conf)
        if conf < 0.4:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop = img[y1:y2, x1:x2]

        # --- check 1-line hay 2-line ---
        is_two = is_two_line_plate(crop)

        if is_two:
            # tách 2 dòng
            top, bottom = split_two_line_plate(crop)

            pre_top = preprocess(top)
            pre_bottom = preprocess(bottom)

            top_raw = ocr_text(pre_top)
            bottom_raw = ocr_text(pre_bottom)

            raw = top_raw + bottom_raw

        else:
            pre = preprocess(crop)
            raw = ocr_text(pre)

        plate = normalize(raw)

        outputs.append({
            "bbox": (x1, y1, x2, y2),
            "crop": crop,
            "raw": raw,
            "plate": plate,
            "conf": conf,
            "two_line": is_two
        })

    return img, outputs



if __name__ == "__main__":
    image_path = r"../data/test/images/test.jpg"

    img, plates = detect_plate(image_path)

    for p in plates:
        print(
            f"Plate: {p['plate']} | Raw: {p['raw']} "
            f"| Conf: {p['conf']:.2f} | TwoLine={p['two_line']}"
        )

    # --- visualize từng plate ---
    for p in plates:
        crop = p["crop"]
        pre = preprocess(crop)

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        # crop
        ax[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Crop")
        ax[0].axis("off")

        # preprocess
        ax[1].imshow(pre, cmap="gray")
        ax[1].set_title("Preprocessed (EasyOCR input)")
        ax[1].axis("off")

        # info
        text_info = (
            f"Raw OCR: {p['raw']}\n"
            f"Normalized: {p['plate']}\n"
            f"Conf: {p['conf']:.2f}\n"
            f"Two-line: {p['two_line']}"
        )
        ax[2].text(0.05, 0.5, text_info, fontsize=13)
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()
