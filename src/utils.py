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
# Preprocess - Tối ưu cho EasyOCR
# ----------------------------------------------------------
def preprocess(crop):
    # Kiểm tra kích thước tối thiểu
    h, w = crop.shape[:2]
    
    # Upscale thông minh: nếu ảnh quá nhỏ thì scale lớn hơn
    if min(h, w) < 50:
        scale = 4.0
    elif min(h, w) < 100:
        scale = 3.0
    else:
        scale = 2.5
    
    # Resize với interpolation tốt
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    
    # Denoise với Non-local Means (tốt hơn bilateral cho text)
    # Nếu ảnh quá nhỏ thì dùng bilateral thay vì NLM (NLM chậm)
    if min(gray.shape) > 100:
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) - tốt hơn equalizeHist
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Adaptive threshold để tạo binary image (tùy chọn)
    # Thử adaptive threshold để tách foreground/background tốt hơn
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations để làm sạch
    # Đóng các lỗ nhỏ trong chữ
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_close)
    
    # Mở để loại bỏ noise nhỏ
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel_open)
    
    # Blend: 60% adaptive threshold + 40% CLAHE gray
    # Adaptive threshold giúp tách chữ rõ, CLAHE giữ texture
    final = cv2.addWeighted(adaptive_thresh, 0.6, gray, 0.4, 0)
    
    # Sharpen nhẹ để làm nét chữ
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    final = cv2.filter2D(final, -1, kernel_sharpen)
    
    # Normalize về [0, 255]
    final = np.clip(final, 0, 255).astype(np.uint8)
    
    return final

# ----------------------------------------------------------
# Detect và tách từng ký tự trong biển số
# ----------------------------------------------------------
def detect_characters(preprocessed_img):
    """
    Tách từng ký tự từ ảnh đã preprocess bằng contour detection
    Returns: list of (x, y, w, h, char_img) - sorted từ trái sang phải
    """
    # Tạo binary image để tìm contours
    # Thử nhiều cách để có kết quả tốt nhất
    _, binary1 = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary2 = cv2.adaptiveThreshold(
        preprocessed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Dùng binary tốt hơn (thường Otsu tốt hơn)
    binary = binary1
    
    # Morphological operations để nối các phần của ký tự bị tách
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_boxes = []
    h_img, w_img = preprocessed_img.shape[:2]
    
    # Lọc và lưu các bounding box của ký tự
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Lọc noise: bỏ các box quá nhỏ hoặc quá lớn
        area = w * h
        img_area = h_img * w_img
        min_area = img_area * 0.005  # 0.5% diện tích ảnh (giảm để bắt được ký tự nhỏ)
        max_area = img_area * 0.25   # 25% diện tích ảnh
        
        # Lọc theo aspect ratio (ký tự thường có ratio hợp lý)
        aspect_ratio = h / w if w > 0 else 0
        
        # Chiều cao tối thiểu (giảm xuống để bắt được ký tự nhỏ hơn)
        min_height = h_img * 0.25  # 25% chiều cao ảnh
        
        if (min_area < area < max_area and 
            0.3 < aspect_ratio < 4.0 and  # Mở rộng range cho aspect ratio
            h > min_height and
            w > 3 and h > 3):  # Kích thước tối thiểu tuyệt đối
            # Thêm padding
            padding = max(3, min(w, h) // 5)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w_img - x, w + 2 * padding)
            h = min(h_img - y, h + 2 * padding)
            
            char_img = preprocessed_img[y:y+h, x:x+w]
            char_boxes.append((x, y, w, h, char_img))
    
    # Sắp xếp từ trái sang phải (theo x)
    char_boxes.sort(key=lambda box: box[0])
    
    return char_boxes

# ----------------------------------------------------------
# OCR một ký tự đơn lẻ
# ----------------------------------------------------------
def ocr_single_char(char_img):
    """
    OCR một ký tự đơn lẻ với confidence cao hơn
    """
    # Thêm padding trắng xung quanh để EasyOCR đọc tốt hơn
    h, w = char_img.shape[:2]
    padding = max(10, min(h, w) // 4)
    padded = cv2.copyMakeBorder(
        char_img, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=255
    )
    
    # OCR với detail để lấy confidence
    results = reader.readtext(padded, detail=1, paragraph=False)
    
    if not results:
        return "", 0.0
    
    # Lấy kết quả có confidence cao nhất
    best_result = max(results, key=lambda x: x[2])
    char = best_result[1].strip()
    conf = best_result[2]
    
    return char, conf

# ----------------------------------------------------------
# OCR toàn bộ biển bằng cách OCR từng ký tự
# ----------------------------------------------------------
def ocr_by_characters(preprocessed_img):
    """
    OCR biển số bằng cách detect và OCR từng ký tự riêng lẻ
    Returns: (raw_text, normalized_text, char_details)
    """
    char_boxes = detect_characters(preprocessed_img)
    
    if not char_boxes:
        # Nếu không detect được ký tự, fallback về OCR toàn bộ
        raw = ocr_text(preprocessed_img)
        return raw, normalize(raw), []
    
    chars = []
    char_details = []
    
    for x, y, w, h, char_img in char_boxes:
        char, conf = ocr_single_char(char_img)
        if char:
            chars.append(char)
            char_details.append({
                'char': char,
                'bbox': (x, y, w, h),
                'conf': conf
            })
    
    raw_text = "".join(chars)
    normalized_text = normalize(raw_text)
    
    return raw_text, normalized_text, char_details

# ----------------------------------------------------------
# 🔥 OCR bằng EasyOCR (thay Tesseract) - Fallback method
# ----------------------------------------------------------
def ocr_text(img):
    results = reader.readtext(img, detail=0)
    if not results:
        return ""
    return "".join(results)

# ----------------------------------------------------------
# Normalize plate - Dùng regex để loại bỏ ký tự đặc biệt
# ----------------------------------------------------------
def normalize(t):
    if not t:
        return ""
    
    # Chuyển sang uppercase
    t = t.upper()
    
    # Dùng regex để loại bỏ tất cả ký tự đặc biệt (. - _ space và các ký tự khác)
    # Chỉ giữ lại chữ cái và số
    t = re.sub(r"[^A-Z0-9]", "", t)
    
    # Thay thế các ký tự dễ nhầm lẫn
    replacements = {
        "O": "0",  # O -> 0
        "I": "1",  # I -> 1
        "Z": "2",  # Z -> 2
        "S": "5",  # S -> 5
        "B": "8",  # B -> 8
    }
    
    for old, new in replacements.items():
        t = t.replace(old, new)
    
    return t
