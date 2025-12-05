import cv2
import numpy as np
# import pytesseract
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import re

def preprocess(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h = 80
    ratio = h / th.shape[0]
    th = cv2.resize(th, (int(th.shape[1] * ratio), h))
    return th


def ocr_text(img):
    config = "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = pytesseract.image_to_string(img, config=config)
    return text.strip()


def normalize(t):
    t = t.upper()
    t = re.sub(r"[^A-Z0-9]", "", t)
    pattern = r"\d{2}[A-Z]\d{4,5}"
    m = re.search(pattern, t)
    return m.group(0) if m else t
