# Ghi chú: API đã chạy thành công! 🎉

## Vấn đề đã gặp

### 1. Lỗi import easyocr
- **Lỗi:** `ModuleNotFoundError: No module named 'easyocr'`
- **Nguyên nhân:** Uvicorn với `--reload` dùng multiprocessing spawn trên Windows, không kế thừa environment đúng cách
- **Giải pháp:** Tạo script `start_server.py` để kiểm tra imports trước khi chạy

### 2. Lỗi numpy/opencv compatibility
- **Lỗi:** `AttributeError: _ARRAY_API not found` và `ImportError: numpy.core.multiarray failed to import`
- **Nguyên nhân:** 
  - `numpy 2.0.1` không tương thích với `opencv-python-headless 4.7.0.72` (được build với numpy 1.x)
  - OpenCV được compile với NumPy 1.x nhưng môi trường có NumPy 2.x
- **Giải pháp:** Downgrade numpy về 1.x hoặc upgrade opencv-python-headless

## Cách chạy API thành công

### Bước 1: Sửa lỗi numpy/opencv (nếu chưa sửa)

```bash
# Activate environment
conda activate alpr

# Uninstall opencv cũ
pip uninstall -y opencv-python-headless

# Cài numpy 1.26.4
pip install numpy==1.26.4

# Cài lại opencv
pip install opencv-python-headless==4.7.0.72

# Kiểm tra
python -c "import numpy; import cv2; print('OK')"
```

Hoặc chạy script tự động:
```bash
fix_numpy_opencv.bat
```

### Bước 2: Chạy API server

**Cách 1: Dùng script Python (Khuyến nghị)**
```bash
conda activate alpr
python start_server.py
```

**Cách 2: Dùng script batch**
```bash
run_api_simple.bat
```

**Cách 3: Chạy trực tiếp với uvicorn**
```bash
conda activate alpr
uvicorn app.main:app --reload
```

## API Endpoints

Sau khi server chạy thành công, API sẽ có tại:

- **Base URL:** `http://127.0.0.1:8000`
- **Health check:** `GET http://127.0.0.1:8000/`
- **API docs (Swagger):** `http://127.0.0.1:8000/docs`
- **ALPR endpoint:** `POST http://127.0.0.1:8000/alpr`
- **Debug status:** `GET http://127.0.0.1:8000/debug/status`

## Test API

### Test với curl:
```bash
curl -X POST "http://127.0.0.1:8000/alpr" -F "file=@path/to/image.jpg"
```

### Test với debug mode:
```bash
curl -X POST "http://127.0.0.1:8000/alpr?debug=true" -F "file=@path/to/image.jpg"
```

### Test trong browser:
Mở `index.html` trong browser và upload ảnh qua UI

## Lưu ý quan trọng

1. **Environment:** Luôn đảm bảo activate environment `alpr` trước khi chạy
2. **Python version:** Sử dụng Python từ conda environment `alpr`
3. **Dependencies:** 
   - `numpy < 2.0` (khuyến nghị: 1.26.4)
   - `opencv-python-headless == 4.7.0.72`
   - `easyocr` phải được cài trong environment đúng
4. **Model file:** Đảm bảo file `models/best.pt` tồn tại
5. **Port:** Mặc định chạy trên port 8000, nếu bị chiếm có thể đổi trong `start_server.py`

## Cấu trúc file quan trọng

```
ALPR/
├── start_server.py          # Script khởi động API (kiểm tra imports)
├── app/main.py              # FastAPI application
├── fix_numpy_opencv.bat     # Script sửa lỗi numpy/opencv
├── run_api_simple.bat       # Script chạy API đơn giản
├── requirements.txt         # Dependencies (đã pin numpy<2.0)
└── models/best.pt          # YOLO model (cần có)
```

## Troubleshooting

### Nếu vẫn gặp lỗi import:
1. Kiểm tra environment: `conda env list`
2. Activate đúng environment: `conda activate alpr`
3. Kiểm tra Python path: `python -c "import sys; print(sys.executable)"`
4. Kiểm tra packages: `pip list | grep -E "numpy|opencv|easyocr"`

### Nếu port 8000 bị chiếm:
Sửa trong `start_server.py`:
```python
uvicorn.run(
    "app.main:app",
    host="127.0.0.1",
    port=8001,  # Đổi port
    reload=True,
    reload_dirs=[str(BASE_DIR)]
)
```

## Kết luận

✅ API đã chạy thành công sau khi:
1. Sửa lỗi numpy/opencv compatibility
2. Đảm bảo tất cả dependencies được cài đúng trong environment `alpr`
3. Sử dụng script `start_server.py` để kiểm tra trước khi chạy

**Ngày ghi chú:** 2026-01-24
