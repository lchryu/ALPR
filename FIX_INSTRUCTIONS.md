# Hướng dẫn sửa lỗi numpy/opencv

## Vấn đề
- `numpy 2.0.1` không tương thích với `opencv-python-headless 4.7.0.72`
- Lỗi: `AttributeError: _ARRAY_API not found`

## Giải pháp

### Cách 1: Dùng script tự động (Khuyến nghị)
```bash
fix_numpy_opencv.bat
```

### Cách 2: Chạy thủ công trong terminal

**Bước 1: Activate environment**
```bash
conda activate alpr
```

**Bước 2: Uninstall opencv cũ**
```bash
pip uninstall -y opencv-python-headless
```

**Bước 3: Cài numpy 1.x**
```bash
pip install numpy==1.26.4
```

**Bước 4: Cài lại opencv**
```bash
pip install opencv-python-headless==4.7.0.72
```

**Bước 5: Kiểm tra**
```bash
python -c "import numpy; import cv2; print('OK')"
```

### Cách 3: Dùng conda (nếu pip không hoạt động)
```bash
conda activate alpr
conda install numpy=1.26.4 -y
pip install opencv-python-headless==4.7.0.72
```

## Sau khi sửa xong
Chạy API server:
```bash
python start_server.py
```
