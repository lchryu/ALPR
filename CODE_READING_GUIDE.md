# 📖 ALPR Code Reading Guide

**Mục tiêu:** Hiểu và trace toàn bộ codebase từ đầu đến cuối, biết code flow và dependencies.

---

## 🎯 Chiến Lược Đọc Code

**Nguyên tắc:** Đọc từ **entry point** → **core logic** → **supporting modules**

**Thứ tự:**
1. **Entry Point** (API/Frontend) - Xem code được gọi như thế nào
2. **Pipeline Orchestration** - Xem luồng xử lý chính
3. **Core Modules** - Hiểu từng module làm gì
4. **Supporting Modules** - Hiểu các utilities

---

## 🚀 Phase 1: Entry Points (30 phút)

### 1.1 FastAPI Entry Point

**File:** `app/main.py`

**Đọc theo thứ tự:**

1. **Lines 1-40:** Setup và imports
   - Xem cách load model YOLO (line 39-40)
   - Xem cách import pipeline module (line 23-24)

2. **Lines 43-105:** API endpoint `/alpr`
   - **Line 87-90:** Xem cách đọc file upload
   - **Line 95-100:** Xem cách tạo debug logger
   - **Line 103:** **QUAN TRỌNG** - Gọi `run_alpr_on_image()` - đây là entry point vào pipeline

3. **Lines 108-147:** Các endpoints khác (health check, debug images)

**Key Insight:**
```
User uploads image → FastAPI reads file → Calls run_alpr_on_image() → Returns JSON
```

**Trace path:**
```
app/main.py:103 → alpr.pipeline.run_alpr_on_image()
```

---

### 1.2 Frontend Entry Point

**File:** `index.html`

**Đọc:**
- **Lines 505-720:** JavaScript code
- **Line 506:** API URL configuration
- **Lines 603-720:** `sendBtn` click handler
  - **Line 625:** Gọi API endpoint
  - **Line 648-712:** Xử lý response và hiển thị

**Key Insight:**
```
User clicks button → Upload image → POST to /alpr → Display results
```

---

## 🔄 Phase 2: Pipeline Flow (45 phút)

### 2.1 Main Pipeline Function

**File:** `src/alpr/pipeline.py`

**Đọc theo thứ tự:**

1. **Lines 21-47:** Function signature và docstring
   - Hiểu input/output
   - Xem dependencies (YOLO model, debug logger)

2. **Lines 48-72:** Stage 1 - YOLO Detection
   ```
   Input image → YOLO model → Detect plates → Filter by confidence (0.4)
   ```
   - **Line 55:** `model(img)[0]` - YOLO detection
   - **Line 60-62:** Filter low confidence detections
   - **Line 64-65:** Extract bounding box và crop

3. **Lines 74-90:** Stage 2 - Plate Classification
   ```
   Cropped plate → is_two_line_plate() → True/False
   ```
   - **Line 79:** `is_two_line_plate(crop_stage1)` - Classify before deskew
   - **Lý do:** Deskew thay đổi tỷ lệ w/h, làm classification sai

4. **Lines 92-100:** Stage 3 - Deskew
   ```
   Cropped plate → deskew_plate() → Rotated/corrected plate
   ```
   - **Line 95:** `deskew_plate()` - Rotation correction

5. **Lines 102-107:** Stage 4 - Crop Text Region
   ```
   Deskewed plate → crop_text_region() → Tight crop around text
   ```
   - **Line 107:** Remove padding/whitespace

6. **Lines 109-223:** Stage 5 - OCR Processing
   - **Lines 111-223:** Two-line plate handling
     - Split → OCR từng line → Combine → Validate → Fallback nếu cần
   - **Lines 225-233:** Single-line plate handling
     - Direct OCR → Return result

**Key Flow:**
```
Image → YOLO → Crop → Classify → Deskew → Crop Text → OCR → Result
```

**Trace path:**
```
pipeline.py:103 → ocr_plate_complete() → ocr.waterfall.ocr_plate_complete()
```

---

## 🧩 Phase 3: Core Modules (60 phút)

### 3.1 OCR Waterfall Module

**File:** `src/alpr/ocr/waterfall.py`

**Đọc theo thứ tự:**

1. **Lines 17-58:** `ocr_plate()` function signature và docstring
   - Hiểu waterfall logic: Pass 1 → Pass 2 → Pass 3

2. **Lines 59-68:** Global Preprocessing
   ```
   Input image → Deskew (if needed) → Remove border → img_no_border
   ```
   - **Line 62:** Skip deskew nếu đã deskew ở pipeline level
   - **Line 68:** Remove plate border (ảnh hưởng OCR)

3. **Lines 70-80:** Pass 1 - Clean Pass
   ```
   img_no_border → _ocr_pass_1_clean() → Validate → Accept/Reject
   ```
   - **Line 71:** Call Pass 1
   - **Line 75-78:** Validate với thresholds (0.6 conf, 0.8 pattern)
   - **Line 78:** Early exit nếu valid

4. **Lines 82-92:** Pass 2 - Robust Pass
   ```
   img_no_border → _ocr_pass_2_robust() → Validate → Accept/Reject
   ```
   - Chỉ chạy nếu Pass 1 fail
   - Thresholds: 0.5 conf, 0.7 pattern

5. **Lines 94-108:** Pass 3 - Fallback Pass
   ```
   img_no_border → _ocr_pass_3_fallback() → Validate → Accept/Reject
   ```
   - Chỉ chạy nếu Pass 1 & 2 fail
   - Thresholds: 0.3 conf, 0.5 pattern
   - **Line 108:** Vẫn validate, không accept garbage

6. **Lines 110-113:** All Passes Failed
   ```
   Return empty result ("", 0.0, "none")
   ```

**Key Flow:**
```
img_no_border → Pass 1 → Valid? → Return
              ↓ No
              → Pass 2 → Valid? → Return
              ↓ No
              → Pass 3 → Valid? → Return
              ↓ No
              → Return empty
```

**Trace path:**
```
waterfall.py:71 → passes._ocr_pass_1_clean()
waterfall.py:75 → decision.should_accept()
waterfall.py:76 → validation.validate_vn_plate_pattern()
```

---

### 3.2 Decision Layer

**File:** `src/alpr/decision.py`

**Đọc theo thứ tự:**

1. **Lines 12-36:** Threshold Constants
   - Hiểu thresholds cho từng pass
   - Hiểu partial line thresholds (relaxed)

2. **Lines 39-70:** `get_thresholds()` function
   - Returns thresholds cho pass cụ thể
   - Used by waterfall để validate

3. **Lines 72-88:** `score_result()` function
   - Helper để rank results (optional)

4. **Lines 90-164:** `should_accept()` function - **QUAN TRỌNG NHẤT**
   - **Lines 100-102:** Check text validity
   - **Lines 104-107:** Check confidence validity
   - **Lines 109-111:** **Gate 1** - Confidence threshold
   - **Lines 113-118:** Partial line handling (relaxed)
   - **Lines 120-130:** Full plate validation
   - **Lines 132-135:** **Gate 2** - Pattern validation

**Key Logic:**
```
should_accept() = (confidence >= min_conf) AND (pattern_score >= min_pattern)
```

**Trace path:**
```
decision.should_accept() → validation.validate_vn_plate_pattern()
```

---

### 3.3 OCR Passes

**File:** `src/alpr/ocr/passes.py`

**Đọc theo thứ tự:**

1. **Lines 15-95:** `_ocr_pass_1_clean()`
   - **Lines 26-30:** Light preprocessing (3.0-3.5x scale)
   - **Lines 31-40:** Grayscale, padding, CLAHE
   - **Lines 42-43:** Mild threshold
   - **Lines 45-60:** EasyOCR với standard params
   - **Lines 62-66:** Sort và return

2. **Lines 97-193:** `_ocr_pass_2_robust()`
   - **Lines 107-111:** Stronger preprocessing (4.0-4.5x scale)
   - **Lines 112-125:** Denoising, stronger CLAHE
   - **Lines 127-131:** Light sharpening
   - **Lines 133-137:** Adaptive threshold
   - **Lines 139-153:** EasyOCR với sensitive params

3. **Lines 195-293:** `_ocr_pass_3_fallback()`
   - **Lines 205-209:** Aggressive preprocessing (5.0-6.0x scale)
   - **Lines 210-230:** Strong denoising, maximum CLAHE
   - **Lines 232-236:** Strong sharpening
   - **Lines 238-260:** Try both normal và inverted images
   - **Lines 262-273:** Choose better result

**Key Insight:**
```
Pass 1: Fast, minimal preprocessing
Pass 2: Moderate, stronger preprocessing
Pass 3: Slow, aggressive preprocessing + inverted
```

**Trace path:**
```
passes._ocr_pass_1_clean() → easyocr_engine.get_reader() → reader.readtext()
```

---

### 3.4 Geometry Module

**File:** `src/alpr/geometry.py`

**Đọc theo thứ tự:**

1. **Lines 72-100:** `remove_plate_border()`
   - Crop inner region (remove 8% border)
   - Used before OCR passes

2. **Lines 102-191:** `is_two_line_plate()`
   - **Lines 108-112:** Aspect ratio check
   - **Lines 114-191:** Gap detection (horizontal projection)
   - Used in pipeline để classify

3. **Lines 193-307:** `crop_text_region()`
   - Find text contours → Union bbox → Crop với margin
   - Used sau deskew để remove padding

4. **Lines 309-379:** `split_two_line_plate()`
   - Horizontal projection → Find gap → Split với overlap
   - Used cho two-line plates

5. **Lines 381-596:** `deskew_plate()` - **PHỨC TẠP NHẤT**
   - **Lines 404-439:** Contour detection
   - **Lines 441-485:** Filter character contours
   - **Lines 487-510:** minAreaRect method
   - **Lines 512-560:** Projection profile method (PRIMARY)
   - **Lines 562-580:** Combine methods
   - **Lines 582-595:** Apply rotation correction

**Key Functions:**
```
remove_plate_border() → Used before OCR
is_two_line_plate() → Used in pipeline classification
crop_text_region() → Used after deskew
split_two_line_plate() → Used for two-line OCR
deskew_plate() → Used in pipeline preprocessing
```

---

### 3.5 Validation Module

**File:** `src/alpr/validation.py`

**Đọc theo thứ tự:**

1. **Lines 13-38:** `validate_vn_plate_pattern()`
   - Check length (7-10 chars)
   - Check position 3 is letter (most important)
   - Returns score 0.0-1.0

2. **Lines 40-71:** `post_process_vn_plate()`
   - Fix position 3 (6→G, 4→A, etc.)
   - Used trong normalization

3. **Lines 73-145:** `normalize_plate()` - **QUAN TRỌNG**
   - **Lines 79-81:** Uppercase
   - **Lines 83-84:** Remove special chars
   - **Lines 86-87:** Post-process position 3
   - **Lines 89-143:** Fix OCR mistakes (O→0, I→1, etc.)
   - **Lines 131-141:** Context-aware replacement

**Key Flow:**
```
Raw OCR text → normalize_plate() → Clean text → validate_vn_plate_pattern() → Score
```

---

## 🔗 Phase 4: Dependencies & Data Flow (30 phút)

### 4.1 Complete Request Flow

**Trace một request từ đầu đến cuối:**

```
1. User uploads image via index.html
   ↓
2. POST /alpr → app/main.py:alpr_api()
   ↓
3. Read file → Create debug logger (optional)
   ↓
4. Call run_alpr_on_image(img, model, logger)
   ↓
5. pipeline.py:run_alpr_on_image()
   ├─> YOLO detection (model(img))
   ├─> For each detection:
   │   ├─> is_two_line_plate() → geometry.py
   │   ├─> deskew_plate() → geometry.py
   │   ├─> crop_text_region() → geometry.py
   │   └─> OCR processing:
   │       ├─> If two-line:
   │       │   ├─> split_two_line_plate() → geometry.py
   │       │   ├─> ocr_plate_complete(top) → ocr/waterfall.py
   │       │   ├─> ocr_plate_complete(bottom) → ocr/waterfall.py
   │       │   ├─> Combine results
   │       │   ├─> validate_vn_plate_pattern() → validation.py
   │       │   └─> Fallback to single-line if needed
   │       └─> If single-line:
   │           └─> ocr_plate_complete() → ocr/waterfall.py
   ↓
6. ocr_plate_complete() → ocr/waterfall.py
   ├─> ocr_plate() → ocr/waterfall.py
   │   ├─> deskew_plate() → geometry.py (if skip_deskew=False)
   │   ├─> remove_plate_border() → geometry.py
   │   ├─> Pass 1: _ocr_pass_1_clean() → ocr/passes.py
   │   │   ├─> Preprocessing
   │   │   ├─> EasyOCR → ocr/easyocr_engine.py
   │   │   └─> should_accept() → decision.py
   │   │       ├─> validate_vn_plate_pattern() → validation.py
   │   │       └─> Check confidence + pattern
   │   ├─> Pass 2: _ocr_pass_2_robust() (if Pass 1 fails)
   │   └─> Pass 3: _ocr_pass_3_fallback() (if Pass 1 & 2 fail)
   ├─> normalize_plate() → validation.py
   └─> Return (raw, normalized, confidence, method)
   ↓
7. Return JSON response → app/main.py
   ↓
8. Display in index.html
```

---

### 4.2 Module Dependencies Graph

```
app/main.py
  └─> alpr.pipeline.run_alpr_on_image()
      ├─> alpr.geometry.*
      ├─> alpr.ocr.waterfall.ocr_plate_complete()
      │   ├─> alpr.geometry.deskew_plate()
      │   ├─> alpr.geometry.remove_plate_border()
      │   ├─> alpr.ocr.passes.*
      │   │   └─> alpr.ocr.easyocr_engine.get_reader()
      │   ├─> alpr.decision.should_accept()
      │   │   └─> alpr.validation.validate_vn_plate_pattern()
      │   └─> alpr.validation.normalize_plate()
      └─> alpr.validation.normalize_plate()
          └─> alpr.validation.post_process_vn_plate()
```

---

## 📊 Phase 5: Key Data Structures (20 phút)

### 5.1 Function Signatures Quan Trọng

**Pipeline Entry:**
```python
run_alpr_on_image(
    img: np.ndarray,           # Input image (BGR)
    model: YOLO,               # YOLO model instance
    debug_logger: Optional[DebugImageLogger]
) -> Dict[str, Any]            # {"results": [...], "debug": {...}}
```

**OCR Waterfall:**
```python
ocr_plate_complete(
    img: np.ndarray,
    use_multi_pass: bool = True,
    logger: Optional[DebugImageLogger] = None,
    skip_deskew: bool = False,
    is_partial_line: bool = False
) -> Tuple[str, str, float, str]  # (raw, normalized, confidence, method)
```

**Decision Layer:**
```python
should_accept(
    text: str,
    confidence: float,
    pattern_score: float,
    min_confidence: float,
    min_pattern_score: float,
    is_partial_line: bool = False
) -> bool
```

---

### 5.2 Response Structure

**API Response:**
```python
{
    "results": [
        {
            "bbox": [x1, y1, x2, y2],      # Bounding box
            "raw": "51G-316.91",            # Raw OCR text
            "plate": "51G31691",            # Normalized plate
            "det_conf": 0.95,               # YOLO confidence
            "ocr_conf": 0.87,               # OCR confidence
            "method": "pass1_clean",         # OCR method used
            "two_line": False               # Classification
        }
    ],
    "debug": {                              # Only if debug=true
        "debug_folder": "...",
        "debug_folder_name": "...",
        "debug_images_count": 10,
        "debug_images": [...]
    }
}
```

---

## 🗺️ Phase 6: Code Flow Diagrams (15 phút)

### 6.1 Single-Line Plate Flow

```
Image
  ↓
YOLO Detection (conf >= 0.4)
  ↓
is_two_line_plate() → False
  ↓
deskew_plate()
  ↓
crop_text_region()
  ↓
ocr_plate_complete(skip_deskew=True)
  ↓
ocr_plate()
  ├─> remove_plate_border()
  ├─> Pass 1: _ocr_pass_1_clean()
  │   ├─> Preprocess (light)
  │   ├─> EasyOCR
  │   └─> should_accept(conf>=0.6, pattern>=0.8) → ✅ Return
  │
  ├─> Pass 2: _ocr_pass_2_robust() (if Pass 1 fails)
  │   └─> should_accept(conf>=0.5, pattern>=0.7) → ✅ Return
  │
  └─> Pass 3: _ocr_pass_3_fallback() (if Pass 1 & 2 fail)
      └─> should_accept(conf>=0.3, pattern>=0.5) → ✅ Return or ❌ Empty
  ↓
normalize_plate()
  ↓
Return (raw, normalized, conf, method)
```

---

### 6.2 Two-Line Plate Flow

```
Image
  ↓
YOLO Detection
  ↓
is_two_line_plate() → True
  ↓
deskew_plate()
  ↓
crop_text_region()
  ↓
split_two_line_plate() → (top, bottom)
  ↓
OCR Top Line:
  └─> ocr_plate_complete(top, is_partial_line=True)
      └─> Pass 1/2/3 với relaxed thresholds (pattern=0.0)
  ↓
OCR Bottom Line:
  └─> ocr_plate_complete(bottom, is_partial_line=True)
  ↓
Combine: top_raw + bottom_raw
  ↓
normalize_plate(combined)
  ↓
validate_vn_plate_pattern(combined) → pattern_score
  ↓
If pattern_score < 0.5 AND conf < 0.5:
  └─> Fallback: Try single-line OCR
      └─> If better → Use single-line result
  ↓
Return result
```

---

## 📝 Phase 7: Reading Checklist

### ✅ Entry Points
- [ ] Đọc `app/main.py` - Hiểu API endpoint
- [ ] Đọc `index.html` (lines 505-720) - Hiểu frontend flow

### ✅ Pipeline
- [ ] Đọc `pipeline.py` - Hiểu orchestration
- [ ] Trace một request từ đầu đến cuối

### ✅ Core Modules
- [ ] Đọc `ocr/waterfall.py` - Hiểu waterfall logic
- [ ] Đọc `decision.py` - Hiểu gating logic
- [ ] Đọc `ocr/passes.py` - Hiểu từng pass làm gì
- [ ] Đọc `geometry.py` - Hiểu geometric operations
- [ ] Đọc `validation.py` - Hiểu normalization

### ✅ Supporting
- [ ] Đọc `ocr/easyocr_engine.py` - Hiểu reader singleton
- [ ] Đọc `debug_logger.py` - Hiểu instrumentation

### ✅ Dependencies
- [ ] Vẽ dependency graph
- [ ] Trace data flow cho 1 request
- [ ] Hiểu khi nào gọi function nào

---

## 🎯 Recommended Reading Order

### Day 1: Entry Points & Pipeline (1.5 giờ)
1. `app/main.py` (30 phút)
2. `index.html` JavaScript (20 phút)
3. `pipeline.py` (40 phút)

### Day 2: OCR & Decision (2 giờ)
1. `ocr/waterfall.py` (45 phút)
2. `decision.py` (30 phút)
3. `ocr/passes.py` (45 phút)

### Day 3: Geometry & Validation (1.5 giờ)
1. `geometry.py` (60 phút)
2. `validation.py` (30 phút)

### Day 4: Supporting & Trace (1 giờ)
1. `ocr/easyocr_engine.py` (15 phút)
2. `debug_logger.py` (15 phút)
3. Trace complete flow với 1 example (30 phút)

---

## 💡 Tips Đọc Code Hiệu Quả

1. **Bắt đầu từ entry point:** Luôn bắt đầu từ `app/main.py` hoặc `scripts/`
2. **Follow the data:** Trace một ảnh từ đầu đến cuối
3. **Đọc docstrings trước:** Hiểu function làm gì trước khi đọc implementation
4. **Vẽ flow diagram:** Tự vẽ diagram để hiểu flow
5. **Test while reading:** Chạy code với debug mode để xem intermediate results
6. **Đọc theo layers:** Entry → Pipeline → Modules → Utilities

---

## 🔍 Trace Example: Single Request

**Scenario:** User uploads image với 1 single-line plate

**Step-by-step trace:**

1. **`app/main.py:87`** - `np.frombuffer()` - Convert uploaded file to numpy array
2. **`app/main.py:90`** - `cv2.imdecode()` - Decode image
3. **`app/main.py:103`** - `run_alpr_on_image(img, model, logger)` - **Entry vào pipeline**

4. **`pipeline.py:55`** - `model(img)[0]` - YOLO detection
5. **`pipeline.py:79`** - `is_two_line_plate(crop_stage1)` - Classification
6. **`geometry.py:102-191`** - Check aspect ratio và gap → Returns `False`
7. **`pipeline.py:95`** - `deskew_plate()` - Rotation correction
8. **`geometry.py:381-596`** - Detect angle → Apply rotation
9. **`pipeline.py:107`** - `crop_text_region()` - Remove padding
10. **`pipeline.py:227`** - `ocr_plate_complete(crop_final)` - OCR

11. **`waterfall.py:68`** - `deskew_plate()` - Skip (already deskewed)
12. **`waterfall.py:71`** - `remove_plate_border()` - Remove border
13. **`waterfall.py:75`** - `_ocr_pass_1_clean()` - Pass 1
14. **`passes.py:15-95`** - Preprocess → EasyOCR → Returns text, conf
15. **`waterfall.py:78`** - `normalize_plate()` - Normalize text
16. **`validation.py:73-145`** - Remove special chars → Fix mistakes
17. **`waterfall.py:79`** - `validate_vn_plate_pattern()` - Get pattern score
18. **`validation.py:13-38`** - Check length → Check position 3 → Returns 1.0
19. **`waterfall.py:80`** - `should_accept()` - Check gates
20. **`decision.py:90-164`** - Check confidence >= 0.6 → Check pattern >= 0.8 → Returns `True`
21. **`waterfall.py:81`** - Early return với Pass 1 result

22. **`pipeline.py:227`** - Receive (raw, plate, conf, method)
23. **`pipeline.py:235`** - Append to results
24. **`pipeline.py:246`** - Return response dict

25. **`app/main.py:105`** - Return JSON response
26. **`index.html:648`** - Parse response → Display

---

## 📚 Key Files Summary

| File | Purpose | Key Functions | Dependencies |
|------|---------|---------------|--------------|
| `app/main.py` | API entry point | `alpr_api()` | pipeline, debug_logger |
| `pipeline.py` | Main orchestration | `run_alpr_on_image()` | geometry, ocr, validation |
| `ocr/waterfall.py` | OCR orchestration | `ocr_plate()`, `ocr_plate_complete()` | passes, decision, validation, geometry |
| `decision.py` | Quality gates | `should_accept()`, `get_thresholds()` | validation |
| `ocr/passes.py` | OCR implementations | `_ocr_pass_1/2/3_*()` | easyocr_engine |
| `geometry.py` | Geometric ops | `deskew_plate()`, `is_two_line_plate()`, etc. | debug_logger |
| `validation.py` | Text processing | `normalize_plate()`, `validate_vn_plate_pattern()` | - |
| `ocr/easyocr_engine.py` | EasyOCR singleton | `get_reader()` | easyocr |

---

## 🎓 Learning Path

### Beginner (Hiểu cơ bản)
1. Đọc `app/main.py` - Xem API làm gì
2. Đọc `pipeline.py` - Xem flow chính
3. Trace 1 request với debug mode

### Intermediate (Hiểu chi tiết)
1. Đọc tất cả modules
2. Hiểu waterfall logic
3. Hiểu decision layer
4. Trace với nhiều scenarios

### Advanced (Hiểu sâu)
1. Hiểu từng preprocessing step
2. Hiểu tại sao thresholds như vậy
3. Hiểu trade-offs giữa các passes
4. Có thể modify/extend code

---

**Happy reading! 📖**

