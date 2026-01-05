# 📋 Danh sách file nên xóa

## 🗑️ **Files nên xóa ngay (Test/Noise)**

### Test Files
- ✅ `ocr_paddle_test.py` - Test PaddleOCR (không dùng trong project)
- ✅ `test_pytorch_version.py` - Test PyTorch version (không cần thiết)
- ✅ `api/test_api.py` - Test script API (có thể tạo lại khi cần)

### Duplicate/Old Files
- ✅ `gpt_visualize.py` - File visualization cũ (đã có `visualize_pipeline.py` mới)
- ✅ `src/detect_visual.py` - Script visualization cũ (không dùng nữa)

### Temporary Files
- ✅ `COMMIT_MESSAGE.txt` - File tạm cho commit message (đã commit xong)
- ✅ `DEBUG_LOGGER_SUMMARY.md` - Documentation tạm (có thể merge vào README)
- ✅ `DEBUG_LOGGER_USAGE.md` - Documentation tạm (có thể merge vào README)
- ✅ `VISUALIZE_PIPELINE_README.md` - Documentation tạm (có thể merge vào README)
- ✅ `run_api_on_another_pc.md` - Note tạm (nếu không cần thì xóa)

### Empty/Unused
- ✅ `api/readme.md` - Kiểm tra xem có nội dung không, nếu empty thì xóa

---

## 🤔 **Files tùy chọn (Có thể giữ hoặc xóa)**

### Development Tools
- ⚠️ `src/detect_single.py` - Script debug/test pipeline
  - **Giữ nếu:** Cần debug pipeline thường xuyên
  - **Xóa nếu:** Chỉ dùng `visualize_pipeline.py` và debug mode trong API

- ⚠️ `src/train.py` - Script train YOLO model
  - **Giữ nếu:** Cần train model mới
  - **Xóa nếu:** Không train nữa, chỉ dùng model có sẵn

---

## ✅ **Files nên GIỮ (Core/Useful)**

### Core Files
- ✅ `api/main.py` - API chính
- ✅ `src/utils.py` - Core OCR logic
- ✅ `src/debug_logger.py` - Debug logger
- ✅ `index.html` - Frontend UI
- ✅ `visualize_pipeline.py` - Visualization tool

### Documentation
- ✅ `README.md` - Main documentation
- ✅ `PROJECT_STRUCTURE.md` - Project structure guide

### Config/Dependencies
- ✅ `requirements.txt` - Main dependencies
- ✅ `api/requirements.txt` - API dependencies
- ✅ `models/best.pt` - YOLO model
- ✅ `data/` - Dataset (giữ nếu cần)

---

## 🚀 **Lệnh xóa nhanh (PowerShell)**

```powershell
# Xóa test files
Remove-Item ocr_paddle_test.py
Remove-Item test_pytorch_version.py
Remove-Item api\test_api.py

# Xóa duplicate/old files
Remove-Item gpt_visualize.py
Remove-Item src\detect_visual.py

# Xóa temporary files
Remove-Item COMMIT_MESSAGE.txt
Remove-Item DEBUG_LOGGER_SUMMARY.md
Remove-Item DEBUG_LOGGER_USAGE.md
Remove-Item VISUALIZE_PIPELINE_README.md
Remove-Item run_api_on_another_pc.md

# Xóa empty readme (nếu empty)
# Remove-Item api\readme.md
```

---

## 📝 **Ghi chú**

- **Backup trước khi xóa:** Nếu không chắc, backup vào folder khác trước
- **Git:** Nếu đã commit, có thể xóa an toàn (có thể recover từ git)
- **Documentation:** Có thể merge nội dung vào README.md chính trước khi xóa


