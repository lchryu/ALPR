import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from utils import preprocess, ocr_text, normalize

# model = YOLO("../models/best.pt")
model = YOLO("../models/best.pt")
def detect_plate(image_path):
    img = cv2.imread(image_path)
    results = model(img)[0]

    outputs = []
    for box in results.boxes:
        conf = float(box.conf)
        if conf < 0.5:
            continue
        
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop = img[y1:y2, x1:x2]

        pre = preprocess(crop)
        raw = ocr_text(pre)
        plate = normalize(raw)

        outputs.append((plate, conf))

    return img, outputs


if __name__ == "__main__":
    # img, plates = detect_plate("../data/test/images/test1.jpg")
    # img, plates = detect_plate("../data/test/images/CarLongPlate55_jpg.rf.f2c7b8e5345cddb4c181eda34e453bab.jpg")
    img, plates = detect_plate("../data/test/images/xemayBigPlate228_jpg.rf.9df9b828972ad59b0e88bff3d6f0b037.jpg")
    
    # img, plates = detect_plate(r"D:\ALPR\data\test\images\test.jpg")


    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    for p, conf in plates:
        print(p, conf)
