import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from utils import (
    preprocess,
    ocr_by_characters,
    normalize,
    is_two_line_plate,
    split_two_line_plate,
    detect_characters
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

            # Detect và OCR từng ký tự cho mỗi dòng
            top_char_boxes = detect_characters(pre_top)
            bottom_char_boxes = detect_characters(pre_bottom)
            
            top_raw, top_norm, top_chars = ocr_by_characters(pre_top)
            bottom_raw, bottom_norm, bottom_chars = ocr_by_characters(pre_bottom)

            # Gom lại: dòng trên + dòng dưới
            raw = top_raw + bottom_raw
            plate = normalize(raw)  # Normalize lại toàn bộ sau khi gom
            char_details = top_chars + bottom_chars
            char_boxes = top_char_boxes + bottom_char_boxes
            preprocessed_img = pre_top  # Dùng pre_top để visualize (có thể cải thiện sau)

        else:
            pre = preprocess(crop)
            char_boxes = detect_characters(pre)
            raw, plate, char_details = ocr_by_characters(pre)
            preprocessed_img = pre

        outputs.append({
            "bbox": (x1, y1, x2, y2),
            "crop": crop,
            "preprocessed": preprocessed_img,  # Ảnh đã preprocess để visualize
            "char_boxes": char_boxes,  # Các ký tự đã detect (chưa OCR)
            "raw": raw,
            "plate": plate,
            "conf": conf,
            "two_line": is_two,
            "char_details": char_details  # Chi tiết từng ký tự (đã OCR)
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

    # --- visualize từng plate và từng ký tự ---
    for p in plates:
        crop = p["crop"]
        pre = p.get("preprocessed", preprocess(crop))  # Dùng preprocessed nếu có
        
        # Lấy các ký tự đã detect
        char_boxes = p.get("char_boxes", [])
        
        print(f"\n=== Detected {len(char_boxes)} characters ===")
        
        # Visualize 1: Overview của plate
        fig1, ax1 = plt.subplots(1, 3, figsize=(14, 4))
        
        # crop gốc
        ax1[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax1[0].set_title("Original Crop")
        ax1[0].axis("off")
        
        # preprocessed với bounding boxes
        pre_with_boxes = pre.copy()
        if len(pre_with_boxes.shape) == 2:
            pre_with_boxes = cv2.cvtColor(pre_with_boxes, cv2.COLOR_GRAY2RGB)
        
        for i, (x, y, w, h, _) in enumerate(char_boxes):
            cv2.rectangle(pre_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(pre_with_boxes, str(i), (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        ax1[1].imshow(pre_with_boxes)
        ax1[1].set_title(f"Preprocessed with {len(char_boxes)} detected chars")
        ax1[1].axis("off")
        
        # info
        text_info = (
            f"Raw OCR: {p['raw']}\n"
            f"Normalized: {p['plate']}\n"
            f"Conf: {p['conf']:.2f}\n"
            f"Two-line: {p['two_line']}\n"
            f"Chars detected: {len(char_boxes)}"
        )
        ax1[2].text(0.05, 0.5, text_info, fontsize=13, verticalalignment='center')
        ax1[2].axis("off")
        
        plt.tight_layout()
        plt.show()
        
        # Visualize 2: Từng ký tự đã crop
        if char_boxes:
            n_chars = len(char_boxes)
            cols = min(8, n_chars)  # Tối đa 8 cột
            rows = (n_chars + cols - 1) // cols
            
            fig2, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
            if n_chars == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if isinstance(axes, list) else [axes]
            else:
                axes = axes.flatten()
            
            for i, (x, y, w, h, char_img) in enumerate(char_boxes):
                if i < len(axes):
                    axes[i].imshow(char_img, cmap="gray")
                    axes[i].set_title(f"Char {i}\n({w}x{h})", fontsize=10)
                    axes[i].axis("off")
            
            # Ẩn các subplot không dùng
            for i in range(n_chars, len(axes)):
                axes[i].axis("off")
            
            plt.suptitle(f"Detected Characters ({n_chars} chars)", fontsize=14)
            plt.tight_layout()
            plt.show()
            
            # In thông tin từng ký tự
            print("\nCharacter details:")
            for i, (x, y, w, h, _) in enumerate(char_boxes):
                if i < len(p['char_details']):
                    char_info = p['char_details'][i]
                    print(f"  Char {i}: '{char_info['char']}' | "
                          f"Conf: {char_info['conf']:.2f} | "
                          f"BBox: ({x},{y},{w},{h})")
                else:
                    print(f"  Char {i}: BBox: ({x},{y},{w},{h})")
        else:
            print("⚠️  No characters detected!")
