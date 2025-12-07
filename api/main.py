from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO
import sys
import os

# Add src/ to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from utils import (
    ocr_plate_complete,
    is_two_line_plate,
    split_two_line_plate,
    normalize_plate
)

app = FastAPI()

# Load YOLO model
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
model = YOLO(MODEL_PATH)


@app.post("/alpr")
async def alpr_api(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # YOLO detect
    results = model(img)[0]

    output = []

    for box in results.boxes:
        conf_det = float(box.conf)
        if conf_det < 0.4:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop = img[y1:y2, x1:x2]

        is_two = is_two_line_plate(crop)

        if is_two:
            # Split into 2
            top, bottom = split_two_line_plate(crop)

            top_raw, top_norm, top_conf, top_method = ocr_plate_complete(top)
            bot_raw, bot_norm, bot_conf, bot_method = ocr_plate_complete(bottom)

            raw = top_raw + bot_raw
            plate = normalize_plate(raw)

            ocr_conf = (top_conf + bot_conf) / 2 if top_conf and bot_conf else max(top_conf, bot_conf)
            method = f"{top_method}+{bot_method}"

        else:
            raw, plate, ocr_conf, method = ocr_plate_complete(crop)

        output.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "raw": raw,
            "plate": plate,
            "det_conf": conf_det,
            "ocr_conf": ocr_conf,
            "method": method,
            "two_line": is_two
        })

    return {"results": output}


@app.get("/")
def root():
    return {"message": "ALPR FastAPI is running!"}
