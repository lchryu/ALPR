# 📁 Cấu trúc Project ALPR

## 🎯 **FILE QUAN TRỌNG - CORE**

### **API Backend** 
📂 `api/main.py` 
- FastAPI endpoint `/alpr` 
- Nhận ảnh upload, trả về JSON với kết quả nhận dạng biển số
- **Đây là API chính để frontend gọi**

### **Frontend (Web UI)**
📂 `index.html`
- Giao diện web upload ảnh và hiển thị kết quả
- Gọi API `/alpr` để xử lý
- **Đây là UI chính cho user**

### **Core Pipeline - Xử lý OCR**
📂 `src/utils.py` ⭐ **QUAN TRỌNG NHẤT**
- Tất cả logic OCR: preprocessing, multi-pass OCR, deskew, normalization
- Hàm chính: `ocr_plate_complete()`, `preprocess_plate()`, `normalize_plate()`
- **Đây là brain của project**

📂 `src/detect_single.py`
- Script để test pipeline trên 1 ảnh
- Có visualization đầy đủ các bước preprocessing
- **Dùng để debug/test pipeline**

---

## 🧪 **FILE TEST/NOISE - Có thể xóa**

📂 `api/test_api.py`
- Script test API bằng requests
- **Chỉ để test, không cần thiết cho production**

📂 `ocr_paddle_test.py`
- Test PaddleOCR (thư viện khác, không dùng trong project)
- **File cũ, có thể xóa**

📂 `test_pytorch_version.py`
- Check version PyTorch/CUDA
- **File test đơn giản, có thể xóa**

📂 `src/detect_visual.py`
- Script visualization cũ (có vẻ không dùng nữa)
- **Có thể xóa nếu không dùng**

📂 `src/train.py`
- Script train YOLO model
- **Chỉ cần khi train model mới**

---

## 📦 **FOLDER KHÁC**

📂 `models/best.pt` - Model YOLO đã train
📂 `data/` - Dataset train/test
📂 `src/runs/` - Output từ YOLO training
📂 `api/requirements.txt` - Dependencies cho API
📂 `requirements.txt` - Dependencies chính

---

## 🚀 **Cách chạy**

1. **Chạy API:**
   ```bash
   cd api
   uvicorn main:app --reload
   ```

2. **Mở Frontend:**
   - Mở file `index.html` bằng browser
   - Hoặc serve bằng web server

3. **Test Pipeline:**
   ```bash
   cd src
   python detect_single.py
   ```

