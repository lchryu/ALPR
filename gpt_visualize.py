import os
import sys
import time
import json
import re

import cv2
import numpy as np
import easyocr


# ===============================
# EasyOCR (CPU by default)
# ===============================
# Tip: if you have GPU and want it: easyocr.Reader(['en'], gpu=True)
reader = easyocr.Reader(['en'], gpu=False)


# ===============================
# Output helpers
# ===============================
class StepSaver:
    def __init__(self, out_root="runs"):
        ts = time.strftime("vis_%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(out_root, ts)
        os.makedirs(self.out_dir, exist_ok=True)
        self.idx = 0

    def save(self, name, img):
        """Save image with auto step index."""
        path = os.path.join(self.out_dir, f"{self.idx:02d}_{name}.jpg")
        if img is None or (hasattr(img, "size") and img.size == 0):
            raise ValueError(f"Refusing to save empty image for step '{name}'")
        cv2.imwrite(path, img)
        self.idx += 1
        return path

    def save_text(self, name, text):
        path = os.path.join(self.out_dir, f"{self.idx:02d}_{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.idx += 1
        return path

    def save_json(self, name, obj):
        path = os.path.join(self.out_dir, f"{self.idx:02d}_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        self.idx += 1
        return path


# ===============================
# VN plate validate + normalize (from your logic)
# ===============================
def validate_vn_plate_pattern(text: str) -> float:
    """
    Validate if text matches Vietnamese plate pattern (soft score).
    Core rule: position 3 (index 2) should be a LETTER.
    Return score [0..1].
    """
    if not text or len(text) < 3:
        return 0.0

    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(clean) < 7 or len(clean) > 10:
        return 0.0

    if len(clean) > 2:
        return 1.0 if clean[2].isalpha() else 0.3

    return 0.5


def post_process_vn_plate(text: str) -> str:
    """Fix position 3 (index 2) to a LETTER if OCR produced a digit there."""
    if not text or len(text) < 3:
        return text

    chars = list(text.upper())
    if len(chars) > 2 and chars[2].isdigit():
        fixes = {
            "6": "G",
            "4": "A",
            "0": "A",
            "1": "I",
            "5": "S",
            "8": "B",
        }
        if chars[2] in fixes:
            chars[2] = fixes[chars[2]]
    return "".join(chars)


def normalize_plate(text: str) -> str:
    """
    Normalize plate:
    - keep only A-Z0-9
    - enforce index 2 letter
    - replace common OCR mistakes BUT preserve index 2 if it's a letter
    """
    if not text:
        return ""

    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    text = post_process_vn_plate(text)

    replacements = {
        "O": "0",
        "I": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    }

    out = []
    for i, ch in enumerate(text):
        if i == 2 and ch.isalpha():
            out.append(ch)
        else:
            out.append(replacements.get(ch, ch))
    return "".join(out)


def is_valid_result(normalized_text: str, conf: float, min_conf: float, min_pattern: float) -> bool:
    if not normalized_text or conf < min_conf:
        return False
    score = validate_vn_plate_pattern(normalized_text)
    return score >= min_pattern


# ===============================
# Deskew (same idea as your code)
# ===============================
def deskew_plate(img, angle_threshold=2.0, debug=False):
    """Rotation correction using minAreaRect; fallback projection if extreme."""
    if img is None:
        return img

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    if min(h, w) < 30:
        if debug:
            print(f"Deskew: too small ({w}x{h}) -> skip")
        return img

    _, bin1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bin2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts1, _ = cv2.findContours(bin1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(bin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts1) > len(cnts2) or (cnts1 and not cnts2):
        cnts = cnts1
    else:
        cnts = cnts2

    if not cnts:
        return img

    largest = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[2]

    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    if abs(angle) < angle_threshold:
        if debug:
            print(f"Deskew: angle {angle:.2f} < {angle_threshold} -> skip")
        return img

    if debug:
        print(f"Deskew: angle {angle:.2f} -> rotate")

    # Projection fallback if crazy
    if abs(angle) > 30:
        _, bin_proj = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        angles = np.arange(-15, 15, 0.5)
        best_angle, best_score = 0.0, -1.0
        center = (gray.shape[1] // 2, gray.shape[0] // 2)

        for a in angles:
            M = cv2.getRotationMatrix2D(center, a, 1.0)
            rotated = cv2.warpAffine(
                bin_proj, M, (gray.shape[1], gray.shape[0]),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            proj = np.sum(rotated, axis=1)
            score = float(np.var(proj))
            if score > best_score:
                best_score = score
                best_angle = a

        if abs(best_angle) >= angle_threshold:
            angle = best_angle
            if debug:
                print(f"Deskew: fallback projection angle -> {angle:.2f}")

    # rotate with expanded bounds to reduce cropping
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    border_val = (255, 255, 255) if len(img.shape) == 3 else 255
    corrected = cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val
    )
    return corrected


# ===============================
# Preprocess (visualize-first, mirrors your preprocess_plate)
# ===============================
def preprocess_plate_visual(img_bgr, saver: StepSaver, variant="standard", apply_deskew=True, debug=False):
    # 0 Deskew
    if apply_deskew:
        img_bgr = deskew_plate(img_bgr, angle_threshold=2.0, debug=debug)
    saver.save("03_deskew", img_bgr)

    h, w = img_bgr.shape[:2]

    # 1 Upscale (same policy)
    if min(h, w) < 50:
        scale = 5.0
    elif min(h, w) < 100:
        scale = 4.5
    else:
        scale = 4.0

    up = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    saver.save("04_upscale", up)

    # 2 Gray
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY) if len(up.shape) == 3 else up.copy()
    saver.save("05_gray", gray)

    # 3 Padding (right extra)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 40, cv2.BORDER_CONSTANT, value=255)
    saver.save("06_padding", gray)

    # 4 Denoise (gentle vs clean)
    if variant == "clean":
        if min(gray.shape) > 100:
            dn = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
        else:
            dn = cv2.bilateralFilter(gray, 5, 60, 60)
    else:
        if min(gray.shape) > 100:
            dn = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7, searchWindowSize=21)
        else:
            dn = cv2.bilateralFilter(gray, 3, 40, 40)
    saver.save("07_denoise", dn)

    # 5 CLAHE
    clip = 3.5 if variant == "high_contrast" else 2.5
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(dn)
    saver.save("08_clahe", enhanced)

    # 6 Sharpen
    if variant == "sharp":
        kernel = np.array([[0, -0.5, 0],
                           [-0.5, 4.5, -0.5],
                           [0, -0.5, 0]])
    else:
        kernel = np.array([[0, -0.3, 0],
                           [-0.3, 3.2, -0.3],
                           [0, -0.3, 0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    saver.save("09_sharpen", sharp)

    # 7 Otsu binary
    _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    saver.save("10_binary_otsu", binary)

    # 8 separate characters slightly
    kernel_separate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    binary_sep = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_separate, iterations=1)
    saver.save("11_binary_separate", binary_sep)

    # 9 blend (same weights)
    binary_final = cv2.addWeighted(binary_sep, 0.7, binary, 0.3, 0)
    final = cv2.addWeighted(binary_final, 0.8, sharp, 0.2, 0)
    final = np.clip(final, 0, 255).astype(np.uint8)
    saver.save("12_final_blend", final)

    return final


# ===============================
# OCR passes (mirror your pass params)
# ===============================
def easyocr_read(img, width_ths, height_ths=None):
    if height_ths is None:
        height_ths = width_ths
    params = dict(
        detail=1,
        paragraph=False,
        width_ths=width_ths,
        height_ths=height_ths,
        slope_ths=0.1,
        allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )
    results = reader.readtext(img, **params)
    if not results:
        return None, 0.0, []
    text = "".join([r[1] for r in results])
    conf = float(np.mean([r[2] for r in results]))
    return text, conf, results


def ocr_pass_1_clean(crop_bgr, saver: StepSaver):
    # deskew + moderate upscale + CLAHE + Otsu (as your code)
    img = deskew_plate(crop_bgr, angle_threshold=2.0, debug=False)
    h, w = img.shape[:2]
    scale = 3.5 if min(h, w) < 100 else 3.0
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    gray = cv2.copyMakeBorder(gray, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    saver.save("13_pass1_input", binary)
    text, conf, _ = easyocr_read(binary, width_ths=0.6, height_ths=0.6)
    return text, conf


def ocr_pass_2_robust(crop_bgr, saver: StepSaver):
    img = deskew_plate(crop_bgr, angle_threshold=2.0, debug=False)
    h, w = img.shape[:2]
    scale = 4.5 if min(h, w) < 100 else 4.0
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    if min(gray.shape) > 100:
        gray = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7, searchWindowSize=21)
    else:
        gray = cv2.bilateralFilter(gray, 3, 40, 40)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    kernel = np.array([[0, -0.3, 0],
                       [-0.3, 3.2, -0.3],
                       [0, -0.3, 0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    adaptive = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )
    saver.save("14_pass2_input", adaptive)
    text, conf, _ = easyocr_read(adaptive, width_ths=0.4, height_ths=0.4)
    return text, conf


def ocr_pass_3_fallback(crop_bgr, saver: StepSaver):
    img = deskew_plate(crop_bgr, angle_threshold=2.0, debug=False)
    h, w = img.shape[:2]
    scale = 6.0 if min(h, w) < 100 else 5.0
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    gray = cv2.copyMakeBorder(gray, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)

    if min(gray.shape) > 100:
        gray = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
    else:
        gray = cv2.bilateralFilter(gray, 5, 60, 60)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    kernel = np.array([[0, -1, 0],
                       [-1,  6, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    saver.save("15_pass3_input_normal", binary)

    text, conf, _ = easyocr_read(binary, width_ths=0.3, height_ths=0.3)
    if text:
        return text, conf, "pass3_normal"

    inv = cv2.bitwise_not(binary)
    saver.save("16_pass3_input_inverted", inv)
    text, conf, _ = easyocr_read(inv, width_ths=0.3, height_ths=0.3)
    return text, conf, "pass3_inverted"


# ===============================
# Main pipeline (visualize)
# ===============================
def run(image_path: str, out_root="runs", crop_mode="none", crop_rect=None, debug=False):
    saver = StepSaver(out_root=out_root)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    saver.save("00_input", img)

    # NOTE:
    # This script does NOT run detection.
    # If your input is a full car image, OCR will likely fail.
    # Options:
    # - Provide a cropped plate image
    # - Or use manual crop via --crop x,y,w,h
    crop = img
    if crop_mode == "manual" and crop_rect is not None:
        x, y, w, h = crop_rect
        crop = img[y:y+h, x:x+w]
    saver.save("01_crop", crop)

    # Visualize preprocess like your preprocess_plate (final blend)
    _ = preprocess_plate_visual(crop, saver, variant="standard", apply_deskew=True, debug=debug)

    # Waterfall OCR (3 passes) + stop conditions identical spirit
    attempts = []

    t1, c1 = ocr_pass_1_clean(crop, saver)
    if t1:
        n1 = normalize_plate(t1)
        s1 = validate_vn_plate_pattern(n1)
        attempts.append({"pass": "pass1_clean", "raw": t1, "norm": n1, "conf": c1, "pattern_score": s1})
        if is_valid_result(n1, c1, min_conf=0.6, min_pattern=0.8):
            winner = attempts[-1]
            saver.save_text("17_pass1_ocr", f"raw={t1}\nconf={c1}\nnorm={n1}\npattern_score={s1}\nWIN=YES\n")
            summary = {"pass_used": "pass1_clean", **winner}
            saver.save_json("18_summary", summary)
            print(f"[WIN] pass1_clean | raw={t1} | norm={n1} | conf={c1:.3f} | pattern={s1:.2f}")
            print(f"Output folder: {saver.out_dir}")
            return

        saver.save_text("17_pass1_ocr", f"raw={t1}\nconf={c1}\nnorm={n1}\npattern_score={s1}\nWIN=NO\n")
    else:
        saver.save_text("17_pass1_ocr", "NO RESULT\n")

    t2, c2 = ocr_pass_2_robust(crop, saver)
    if t2:
        n2 = normalize_plate(t2)
        s2 = validate_vn_plate_pattern(n2)
        attempts.append({"pass": "pass2_robust", "raw": t2, "norm": n2, "conf": c2, "pattern_score": s2})
        if is_valid_result(n2, c2, min_conf=0.5, min_pattern=0.7):
            winner = attempts[-1]
            saver.save_text("18_pass2_ocr", f"raw={t2}\nconf={c2}\nnorm={n2}\npattern_score={s2}\nWIN=YES\n")
            summary = {"pass_used": "pass2_robust", **winner}
            saver.save_json("19_summary", summary)
            print(f"[WIN] pass2_robust | raw={t2} | norm={n2} | conf={c2:.3f} | pattern={s2:.2f}")
            print(f"Output folder: {saver.out_dir}")
            return

        saver.save_text("18_pass2_ocr", f"raw={t2}\nconf={c2}\nnorm={n2}\npattern_score={s2}\nWIN=NO\n")
    else:
        saver.save_text("18_pass2_ocr", "NO RESULT\n")

    t3, c3, mode3 = ocr_pass_3_fallback(crop, saver)
    if t3:
        n3 = normalize_plate(t3)
        s3 = validate_vn_plate_pattern(n3)
        winner = {"pass": f"pass3_fallback({mode3})", "raw": t3, "norm": n3, "conf": c3, "pattern_score": s3}
        attempts.append(winner)
        saver.save_text("19_pass3_ocr", f"raw={t3}\nconf={c3}\nnorm={n3}\npattern_score={s3}\nWIN=FALLBACK\n")
        summary = {"pass_used": "pass3_fallback", **winner, "attempts": attempts}
        saver.save_json("20_summary", summary)
        print(f"[FALLBACK] pass3 | raw={t3} | norm={n3} | conf={c3:.3f} | pattern={s3:.2f}")
    else:
        saver.save_text("19_pass3_ocr", "NO RESULT\n")
        summary = {"pass_used": "none", "attempts": attempts}
        saver.save_json("20_summary", summary)
        print("[FAIL] All passes produced no text.")

    print(f"Output folder: {saver.out_dir}")


def parse_crop_arg(s: str):
    # "x,y,w,h"
    parts = s.split(",")
    if len(parts) != 4:
        raise ValueError("crop must be x,y,w,h")
    return tuple(int(p.strip()) for p in parts)


if __name__ == "__main__":
    # Usage:
    # python gpt_visualize.py <image_path>
    # optional:
    #   --out runs
    #   --debug
    #   --crop x,y,w,h
    args = sys.argv[1:]
    if not args:
        print("Usage: python gpt_visualize.py <image_path> [--out runs] [--debug] [--crop x,y,w,h]")
        sys.exit(1)

    image_path = args[0]
    out_root = "runs"
    debug = False
    crop_mode = "none"
    crop_rect = None

    i = 1
    while i < len(args):
        if args[i] == "--out":
            out_root = args[i + 1]
            i += 2
        elif args[i] == "--debug":
            debug = True
            i += 1
        elif args[i] == "--crop":
            crop_mode = "manual"
            crop_rect = parse_crop_arg(args[i + 1])
            i += 2
        else:
            print(f"Unknown arg: {args[i]}")
            sys.exit(1)

    run(image_path, out_root=out_root, crop_mode=crop_mode, crop_rect=crop_rect, debug=debug)
