from os import system
system("cls")

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.load.*weights_only.*')

from ultralytics import YOLO
import cv2

# Load model YOLO (model detect biển số)
model = YOLO("./models/best.pt")  # đường dẫn model của m

# Load ảnh test
# img_path = "./data/test/images/car1.jpg"
# img_path = "./data/test/images/test_xm1.jpg"
# img_path = "./data/test/images/xemay295_jpg.rf.cce52bb8031707a3987d91ab1bc6b45d.jpg"
img_path = "./img_test/multi_box.png"
img = cv2.imread(img_path)

# Chạy detection
results = model(img)
box_coords = results[0].boxes.xyxy.cpu().numpy()
print(f'Box coordinates: {box_coords}')
print(f'Box confidence: {results[0].boxes.conf.cpu().numpy()}')
print(f'Box class: {results[0].boxes.cls.cpu().numpy()}')
print(f'Box class name: {results[0].boxes.cls.cpu().numpy().tolist()}')

# Vẽ bounding box
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        # in ra các thông tin trên của box
        print(f'Box coordinates: {x1, y1, x2, y2}')
        print(f'Box confidence: {conf}')
        print(f'Box class: {box.cls.cpu().numpy()}')
        print(f'Box class name: {box.cls.cpu().numpy().tolist()}')
        print("=" * 100)
        # ====================================================================
        
        # Tham số vẽ bounding box
        bbox_color = (0, 255, 0)  # màu xanh lá
        bbox_thickness = 2
        
        # Tham số vẽ text
        text_position = (x1, y1 - 5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (0, 255, 0)
        text_thickness = 1
        
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
        cv2.putText(
            img,
            f"plate {conf:.2f}",
            text_position,
            font,
            font_scale,
            text_color,
            text_thickness
        )

# # Lưu ảnh kết quả
cv2.imwrite("bbox_result.jpg", img)
