# ALPR Pipeline Visualization Tool

## 📁 File Location
**`visualize_pipeline.py`** (at project root)

## 🚀 Usage Instructions

### Basic Usage
```bash
python visualize_pipeline.py --image path/to/image.jpg
```

### With Custom Output Directory
```bash
python visualize_pipeline.py --image path/to/image.jpg --out custom/output/dir
```

### Without GUI (Save Images Only)
```bash
python visualize_pipeline.py --image path/to/image.jpg --no-gui
```

## 📊 Output Structure

All outputs are saved to: `runs/vis_<timestamp>/`

### File Naming (Strict Order)
```
00_input.jpg                    # Original input image
01_detect_overlay.jpg           # Original with YOLO detection bbox
02_crop.jpg                      # Cropped plate region
03_deskew.jpg                   # Deskewed/corrected plate
04_gray.jpg                     # Grayscale conversion
05_resize.jpg                   # Upscaled image
06_denoise.jpg                  # Denoised image
07_contrast.jpg                 # CLAHE contrast enhancement
08_threshold.jpg                # Binary threshold
09_morph_close.jpg              # (Optional) Morphology operations
10_invert_optional.jpg          # (Optional) Inverted for pass 3
11_pass1_input.jpg              # OCR Pass 1 input (binary image)
12_pass1_ocr.txt                # OCR Pass 1 results
13_pass2_input.jpg              # OCR Pass 2 input (binary image)
14_pass2_ocr.txt                # OCR Pass 2 results
15_pass3_input.jpg              # OCR Pass 3 input (binary image)
16_pass3_ocr.txt                # OCR Pass 3 results
17_final_overlay.jpg            # Final result overlay on original image
18_summary.json                 # Complete pipeline summary
```

### Example Output Directory Tree
```
runs/
└── vis_20241201_143022/
    ├── 00_input.jpg
    ├── 01_detect_overlay.jpg
    ├── 02_crop.jpg
    ├── 03_deskew.jpg
    ├── 04_gray.jpg
    ├── 05_resize.jpg
    ├── 06_denoise.jpg
    ├── 07_contrast.jpg
    ├── 08_threshold.jpg
    ├── 10_invert_optional.jpg
    ├── 11_pass1_input.jpg
    ├── 12_pass1_ocr.txt
    ├── 13_pass2_input.jpg
    ├── 14_pass2_ocr.txt
    ├── 15_pass3_input.jpg
    ├── 16_pass3_ocr.txt
    ├── 17_final_overlay.jpg
    └── 18_summary.json
```

### Summary JSON Format
```json
{
  "det_conf": 0.95,
  "pass_used": "pass1_clean",
  "raw_text": "51G-31691",
  "normalized_plate": "51G31691",
  "pattern_ok": true,
  "ocr_conf": 0.87,
  "method": "pass1_clean",
  "two_line": false
}
```

### Console Output
```
================================================================================
ALPR PIPELINE SUMMARY
================================================================================
det_conf     pass_used        raw                  normalized      pattern_ok  ocr_conf    method
--------------------------------------------------------------------------------
0.950        pass1_clean      51G-31691            51G31691        ✓           0.870       pass1_clean
================================================================================

Output saved to: runs/vis_20241201_143022
================================================================================
```

## 🔍 What It Does

1. **Detection**: Runs YOLO to detect license plate
2. **Crop**: Extracts plate region
3. **Deskew**: Corrects rotation/skew
4. **Preprocessing**: Shows all preprocessing steps (gray, resize, denoise, contrast, threshold)
5. **OCR Passes**: Runs 3-pass waterfall OCR:
   - Pass 1 (Clean): Fast, minimal preprocessing
   - Pass 2 (Robust): Stronger preprocessing if Pass 1 fails
   - Pass 3 (Fallback): Aggressive preprocessing as last resort
6. **Final Overlay**: Draws result on original image
7. **Summary**: Saves JSON with all metadata

## ⚠️ Important Notes

- **Does NOT modify production code** - completely standalone
- **Reuses existing utilities** from `src/utils.py` via import
- **Handles two-line plates** automatically (splits and processes separately)
- **Early exit** - stops at first valid pass (waterfall logic)
- **All intermediate steps saved** for debugging

## 🛠️ Requirements

- Python 3.7+
- OpenCV
- NumPy
- Ultralytics YOLO
- EasyOCR
- All dependencies from `requirements.txt`


