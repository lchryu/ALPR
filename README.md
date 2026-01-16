# ALPR - Automatic License Plate Recognition

Hệ thống nhận dạng biển số xe tự động sử dụng YOLOv8 và EasyOCR.

## 🚀 Cài đặt

```bash
pip install -r requirements.txt
```

## 📁 Cấu trúc Project

```
ALPR/
├── app/
│   └── main.py              # FastAPI server (API endpoint)
├── scripts/
│   ├── detect_single.py     # Test trên 1 ảnh
│   ├── train.py             # Train YOLO model
│   └── smoke_test.py        # Kiểm tra pipeline
├── src/
│   └── alpr/                # Core pipeline code
│       ├── pipeline.py      # Main processing
│       ├── decision.py      # Quality gates
│       ├── geometry.py      # Deskew, crop, split
│       ├── validation.py    # Text normalization
│       └── ocr/             # OCR passes
├── config.py                # Config (API keys, paths)
├── index.html               # Frontend UI
└── models/
    └── best.pt              # YOLO model
```

## 🎯 Chạy API Server

```bash
uvicorn app.main:app --reload
```

API sẽ chạy tại: `http://127.0.0.1:8000`

- **API Endpoint:** `POST /alpr`
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **Frontend:** Mở `index.html` trong browser

## 📝 Config

Tạo file `config.py` từ `config.example.py` và điền API key:

```python
ROBOFLOW_API_KEY = "your-api-key-here"
```

## 🧪 Test

```bash
# Test pipeline
python scripts/smoke_test.py path/to/image.jpg

# Test single image
python scripts/detect_single.py path/to/image.jpg
```

## 📚 API Usage

```bash
curl -X POST "http://127.0.0.1:8000/alpr?debug=true" \
  -F "file=@image.jpg"
```

Response:
```json
{
  "results": [
    {
      "bbox": [x1, y1, x2, y2],
      "plate": "51G31691",
      "raw": "51G-316.91",
      "det_conf": 0.95,
      "ocr_conf": 0.87,
      "method": "pass1_clean",
      "two_line": false
    }
  ]
}
```
