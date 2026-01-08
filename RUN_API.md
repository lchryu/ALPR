# Hướng dẫn chạy ALPR API

## Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Chạy API Server

### Cách 1: Dùng uvicorn trực tiếp (khuyến nghị)

```bash
# Từ thư mục gốc của project (D:\ALPR)
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### Cách 2: Dùng Python script

Tạo file `run_api.py` ở thư mục gốc:

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
```

Sau đó chạy:
```bash
python run_api.py
```

### Cách 3: Dùng FastAPI CLI (nếu đã cài)

```bash
fastapi dev api/main.py
```

## Kiểm tra API đã chạy

1. Mở trình duyệt và vào: `http://127.0.0.1:8000/docs`
   - Đây là Swagger UI để test API trực tiếp

2. Hoặc test bằng curl:
```bash
curl http://127.0.0.1:8000/docs
```

## Sử dụng Frontend

1. Mở file `index.html` trong trình duyệt
2. Hoặc chạy local server:
```bash
# Python 3
python -m http.server 8080

# Sau đó mở: http://localhost:8080/index.html
```

## Lưu ý

- API chạy trên `http://127.0.0.1:8000`
- Frontend (`index.html`) gọi API tại `http://127.0.0.1:8000/alpr`
- Nếu gặp lỗi CORS, đã có CORS middleware trong `api/main.py`
- Model YOLO cần có file `models/best.pt`
- Debug images sẽ được lưu vào `runs/debug/`

## Troubleshooting

### Lỗi: ModuleNotFoundError
```bash
# Đảm bảo đã cài đủ dependencies
pip install -r requirements.txt
```

### Lỗi: Model not found
- Kiểm tra file `models/best.pt` có tồn tại không
- Nếu không có, cần train model hoặc download model

### Lỗi: Port đã được sử dụng
```bash
# Đổi port khác
uvicorn api.main:app --reload --host 127.0.0.1 --port 8001
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

