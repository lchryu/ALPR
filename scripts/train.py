"""
YOLOv8 Training Script

Downloads dataset from Roboflow and trains YOLOv8 model.
Uses ROBOFLOW_API_KEY environment variable for authentication.
"""

import os
import sys
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO
import torch

# Add project root to path để import config
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from config import ROBOFLOW_API_KEY
except ImportError:
    # Fallback: thử đọc từ env var nếu không có config.py
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    if not ROBOFLOW_API_KEY:
        print("⚠️  Warning: Không tìm thấy config.py và ROBOFLOW_API_KEY env var")
        print("   Tạo file config.py ở thư mục gốc với: ROBOFLOW_API_KEY = 'your-key'")

# -------------------------------
# 0. GPU optimization
# -------------------------------
torch.backends.cudnn.benchmark = True

# -------------------------------
# 1. Download dataset from Roboflow
# -------------------------------
def download_dataset():
    """Download dataset from Roboflow."""
    print("📥 Downloading Roboflow dataset...")
    
    if not ROBOFLOW_API_KEY:
        raise ValueError(
            "ROBOFLOW_API_KEY không được set!\n"
            "Cách 1: Tạo file config.py ở thư mục gốc với: ROBOFLOW_API_KEY = 'your-key'\n"
            "Cách 2: Set environment variable: export ROBOFLOW_API_KEY='your-key'"
        )
    
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
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
    """Train YOLOv8 model on the dataset."""
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
    )
    
    print("✅ Training completed!")
    return model


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    try:
        # Download dataset
        dataset_location = download_dataset()
        data_yaml = os.path.join(dataset_location, "data.yaml")
        
        # Train model
        model = train_yolo(data_yaml)
        
        print(f"\n✅ Model saved to: {model.trainer.best}")
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

