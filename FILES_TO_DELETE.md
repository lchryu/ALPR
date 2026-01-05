# 📋 Danh sách file nên xóa

## ✅ **ĐÃ XÓA (Hoàn thành)**

### Test Files
- ✅ `ocr_paddle_test.py` - Test PaddleOCR (không dùng trong project) - **ĐÃ XÓA**
- ✅ `test_pytorch_version.py` - Test PyTorch version (không cần thiết) - **ĐÃ XÓA**
- ✅ `test_yolo_redetection.py` - Test script YOLO re-detection - **ĐÃ XÓA**
- ✅ `test_bbox_simple.py` - Test script bbox đơn giản - **ĐÃ XÓA**
- ✅ `api/test_api.py` - Test script API (có thể tạo lại khi cần) - **ĐÃ XÓA**

### Duplicate/Old Files
- ✅ `src/detect_visual.py` - Script visualization cũ (không dùng nữa) - **ĐÃ XÓA**

### Temporary Files
- ✅ `COMMIT_MESSAGE.txt` - File tạm cho commit message (đã commit xong) - **ĐÃ XÓA**
- ✅ `DEBUG_LOGGER_SUMMARY.md` - Documentation tạm - **ĐÃ XÓA**
- ✅ `DEBUG_LOGGER_USAGE.md` - Documentation tạm - **ĐÃ XÓA**
- ✅ `VISUALIZE_PIPELINE_README.md` - Documentation tạm - **ĐÃ XÓA**
- ✅ `TEST_REDETECTION_README.md` - Documentation tạm - **ĐÃ XÓA**
- ✅ `run_api_on_another_pc.md` - Note tạm - **ĐÃ XÓA**

### Empty/Unused
- ✅ `api/readme.md` - File empty - **ĐÃ XÓA**

### Code Cleanup
- ✅ Xóa 2 hàm không dùng trong `utils.py`: `detect_individual_characters()`, `detect_characters_vertical_projection()` (~260 dòng)
- ✅ Xóa dependency không dùng: `pytesseract` từ `requirements.txt`

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


