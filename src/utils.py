import cv2
import numpy as np
import re
import easyocr

# Create reader 1 lần (tối ưu)
reader = easyocr.Reader(['en'])

# ----------------------------------------------------------
# Detect if plate is 2-line or 1-line based on aspect ratio
# ----------------------------------------------------------
def is_two_line_plate(crop):
    h, w = crop.shape[:2]
    ratio = w / h
    return ratio < 3.2

# ----------------------------------------------------------
# Split 2-line motorcycle plate
# ----------------------------------------------------------
def split_two_line_plate(crop):
    h, w = crop.shape[:2]
    mid = h // 2
    return crop[0:mid, :], crop[mid:h, :]

# ----------------------------------------------------------
# Preprocess (same)
# ----------------------------------------------------------
def preprocess(crop):
    # Resize lớn hơn để OCR đọc dễ hơn
    scale = 2.0
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Convert gray
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Giảm noise nhẹ (không blur quá mức)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)

    # Tăng contrast
    gray = cv2.equalizeHist(gray)

    # Sharpen nhẹ để nét chữ rõ hơn
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    sharp = cv2.filter2D(gray, -1, kernel)

    return sharp

# ----------------------------------------------------------
# 🔥 OCR bằng EasyOCR (thay Tesseract)
# ----------------------------------------------------------
def ocr_text(img):
    results = reader.readtext(img, detail=0)
    if not results:
        return ""
    return "".join(results)

# ----------------------------------------------------------
# Normalize plate
# ----------------------------------------------------------
def normalize(t):
    t = t.upper()

    t = t.replace(" ", "")
    t = t.replace(".", "")
    t = t.replace("·", "")
    t = t.replace("•", "")
    t = t.replace("-", "")
    t = t.replace("_", "")

    t = t.replace("O", "0")
    t = t.replace("I", "1")
    t = t.replace("Z", "2")
    t = t.replace("S", "5")
    t = t.replace("B", "8")

    t = re.sub(r"[^A-Z0-9]", "", t)
    return t
