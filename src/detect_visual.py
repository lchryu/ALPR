import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from utils import preprocess, ocr_text, normalize

model = YOLO(r"../models/best.pt")


def visualize(image_path):
    img = cv2.imread(image_path)
    results = model(img)[0]

    plates = []

    for box in results.boxes:
        conf = float(box.conf)
        if conf < 0.5:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        crop = img[y1:y2, x1:x2]
        pre = preprocess(crop)
        raw = ocr_text(pre)
        plate = normalize(raw)

        plates.append({
            "bbox": (x1, y1, x2, y2),
            "crop": crop,
            "pre": pre,
            "raw": raw,
            "plate": plate,
            "conf": conf
        })

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{plate} ({conf:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Plot results
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Plate(s)")
    plt.axis('off')
    plt.show()

    # Show crops + preprocess
    for i, p in enumerate(plates):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(p["crop"], cv2.COLOR_BGR2RGB))
        plt.title("Crop")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(p["pre"], cmap="gray")
        plt.title("Preprocessed")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        text = f"Raw OCR: {p['raw']}\nNormalized: {p['plate']}\nConf: {p['conf']:.2f}"
        plt.text(0.1, 0.5, text, fontsize=12)
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    # visualize(r"D:\ALPR\data\test\images\test.jpg")
    visualize(r"D:\ALPR\data\test\images\xemayBigPlate228_jpg.rf.9df9b828972ad59b0e88bff3d6f0b037.jpg")
    # visualize(r"D:\ALPR\data\test\images\CarLongPlate260_jpg.rf.a159ae75442e7381b9a0611f9b71a286.jpg")
