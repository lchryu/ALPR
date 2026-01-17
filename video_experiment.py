"""
Multi-Track ALPR Video Demo
YOLOv8 + ByteTrack + per-track OCR voting
"""

import sys
import time
from pathlib import Path
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

# ----------------- PATH SETUP -----------------
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from alpr.pipeline import run_alpr_on_image


# ----------------- CONFIG -----------------
VIDEO_PATH = "test.mp4"
MODEL_PATH = BASE_DIR / "models" / "best.pt"

OCR_INTERVAL_FRAMES = 15        # mỗi track OCR lại sau N frame
MIN_OCR_CONF = 0.7
BLUR_THRESHOLD = 100

VOTE_WINDOW = 30                # số OCR gần nhất để vote


# ----------------- UTIL -----------------
def normalize_plate(s: str) -> str:
    if not s:
        return ""
    s = s.upper().replace(" ", "").replace(".", "")
    trans = str.maketrans({
        "O": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
        "Z": "2",
    })
    return s.translate(trans)


def compute_weight(ocr_conf, det_conf):
    return float(ocr_conf * det_conf)


def is_blurry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD


def draw_bbox(frame, bbox, text, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if text:
        cv2.rectangle(frame, (x1, y1 - 28), (x2, y1), color, -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# ----------------- TRACK BUFFER -----------------
class PlateTrack:
    def __init__(self):
        self.last_ocr_frame = -999
        self.plates = deque(maxlen=VOTE_WINDOW)
        self.weights = deque(maxlen=VOTE_WINDOW)
        self.best_plate = ""
        self.best_conf = 0.0

    def add(self, plate, weight):
        self.plates.append(plate)
        self.weights.append(weight)
        self.vote()

    def vote(self):
        score = defaultdict(float)
        for p, w in zip(self.plates, self.weights):
            score[p] += w

        if not score:
            return

        total = sum(score.values())
        best = max(score, key=score.get)
        self.best_plate = best
        self.best_conf = score[best] / total if total > 0 else 0.0


# ----------------- MAIN -----------------
def main():
    if not Path(MODEL_PATH).exists():
        print("❌ Model not found")
        return

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_id = 0

    tracks = defaultdict(PlateTrack)

    window = "ALPR Multi-Track Demo"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print("▶ Running multi-track ALPR...")
    print("   SPACE: pause | Q: quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # ---------- YOLO TRACK ----------
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            iou=0.6,
            verbose=False
        )

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det_conf = float(box.conf.item())

                track = tracks[track_id]

                # ---------- OCR GATING ----------
                need_ocr = (
                    frame_id - track.last_ocr_frame >= OCR_INTERVAL_FRAMES
                )

                if need_ocr and not is_blurry(frame):
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    try:
                        res = run_alpr_on_image(crop, model=None, debug_logger=None)
                        if res and res.get("results"):
                            pr = res["results"][0]
                            plate_raw = pr.get("plate", "")
                            ocr_conf = pr.get("ocr_conf", 0)

                            if ocr_conf >= MIN_OCR_CONF:
                                plate_norm = normalize_plate(plate_raw)
                                w = compute_weight(ocr_conf, det_conf)
                                track.add(plate_norm, w)
                                track.last_ocr_frame = frame_id
                    except:
                        pass

                label = f"ID {track_id}"
                if track.best_plate:
                    label += f" | {track.best_plate} ({track.best_conf*100:.0f}%)"

                draw_bbox(display, (x1, y1, x2, y2), label)

        cv2.putText(display, f"Frame {frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(-1)

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n====== FINAL RESULTS ======")
    for tid, t in tracks.items():
        if t.best_plate:
            print(f"Track {tid}: {t.best_plate} ({t.best_conf*100:.1f}%)")


if __name__ == "__main__":
    main()
