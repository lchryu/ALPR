import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.load.*weights_only.*')

from ultralytics import YOLO
import cv2

# Load model YOLO (model detect biển số)
model = YOLO("./models/best.pt")  # đường dẫn model của m

# Load ảnh test
# img_path = "./data/test/images/car1.jpg"
# img_path = "./data/test/images/test_xm1.jpg"
img_path = "./data/test/images/xemay295_jpg.rf.cce52bb8031707a3987d91ab1bc6b45d.jpg"
img = cv2.imread(img_path)

# Chạy detection
results = model(img)

# Vẽ bounding box
# for r in results:
#     for box in r.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])

#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(
#             img,
#             f"plate {conf:.2f}",
#             (x1, y1 - 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 255, 0),
#             1
#         )

# # Lưu ảnh kết quả
# cv2.imwrite("bbox_result.jpg", img)
