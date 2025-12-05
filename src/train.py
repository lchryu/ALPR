import os
from roboflow import Roboflow
from ultralytics import YOLO
import torch

# -------------------------------
# 0. GPU optimization
# -------------------------------
torch.backends.cudnn.benchmark = True

# -------------------------------
# 1. Download dataset from Roboflow
# -------------------------------
def download_dataset():
    print("📥 Downloading Roboflow dataset...")

    rf = Roboflow(api_key="793zVCXAYAbEyuPXlG5E")

    project = rf.workspace("tran-ngoc-xuan-tin-k15-hcm-dpuid").project(
        "vietnam-license-plate-h8t3n"
    )

    dataset = project.version(1).download("yolov8", location="../data")

    print(f"📂 Dataset saved at: {dataset.location}")
    return dataset.location


# -------------------------------
# 2. Train YOLOv8
# -------------------------------
def train_yolo(data_yaml):
    print("🚀 Starting YOLOv8 training...")

    model = YOLO("yolov8s.pt")  # load pretrained

    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=4,        # GTX 1060 → batch=4 là vừa
        device=0,       # GPU
        name="alpr_local",
        patience=10,

        # 🔥 STABILITY MODE (để không lỗi)
        mosaic=0,
        plots=False,
        verbose=False,
        show=False,
        val=True,
        workers=0,
    )

    # Move best.pt to models/
    src = "runs/detect/alpr_local/weights/best.pt"
    dst = "../models/best.pt"

    os.makedirs("../models", exist_ok=True)
    os.replace(src, dst)

    print("🎉 Training complete! best.pt saved to models/")
    return dst


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    dataset_path = download_dataset()
    data_yaml = os.path.join(dataset_path, "data.yaml")
    train_yolo(data_yaml)
