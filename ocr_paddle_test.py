from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(
    lang='en',
    use_textline_orientation=False,   # bản mới thay use_angle_cls
)

img_path = r"D:\ALPR\data\test\images\test.jpg"
img = cv2.imread(img_path)

result = ocr.ocr(img, cls=False)

print("==== OCR Result ====")
for line in result:
    for res in line:
        text, conf = res[1]
        print(text, "| conf:", conf)
