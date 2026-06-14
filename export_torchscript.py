"""
Export tất cả models sang TorchScript (.torchscript).
Chạy trên GPU Jetson Nano bằng PyTorch CUDA — không cần ONNX/TensorRT.

Chạy: python export_torchscript.py
"""

from pathlib import Path
from ultralytics import YOLO

IMGSZ = 640

MODELS = {
    "YOLOv5s":              "runs/detect/yolov5s/weights/best.pt",
    "YOLOv5s-SGC-P345":     "runs/detect/yolov5s_sgc_p345/weights/best.pt",
    "YOLOv5s-SGC-P45":      "runs/detect/yolov5s_sgc_p45/weights/best.pt",
    "YOLOv8s":              "runs/detect/yolov8s/weights/best.pt",
    "YOLOv8s-SGC-P345":     "runs/detect/yolov8s_sgc_p345/weights/best.pt",
    "YOLOv8s-SGC-P45":      "runs/detect/yolov8s_sgc_p45/weights/best.pt",
    "YOLOv9s":              "runs/detect/yolov9s/weights/best.pt",
    "YOLOv9s-SGC-P345":     "runs/detect/yolov9s_sgc_p345/weights/best.pt",
    "YOLOv9s-SGC-P45":      "runs/detect/yolov9s_sgc_p45/weights/best.pt",
    "YOLO11s":              "runs/detect/yolov11s/weights/best.pt",
    "YOLO11s-SGC-P345":     "runs/detect/yolov11s_sgc_p345/weights/best.pt",
    "YOLO11s-SGC-P45":      "runs/detect/yolov11s_sgc_p45/weights/best.pt",

    "YOLOv5s_prune":              "runs/detect/yolov5s_prune/weights/best.pt",
    "YOLOv5s-SGC-P345_prune":     "runs/detect/yolov5s_sgc_p345_prune/weights/best.pt",
    "YOLOv5s-SGC-P45_prune":      "runs/detect/yolov5s_sgc_p45_prune/weights/best.pt",
    "YOLOv8s_prune":              "runs/detect/yolov8s_prune/weights/best.pt",
    "YOLOv8s-SGC-P345_prune":     "runs/detect/yolov8s_sgc_p345_prune/weights/best.pt",
    "YOLOv8s-SGC-P45_prune":      "runs/detect/yolov8s_sgc_p45_prune/weights/best.pt",
    "YOLOv9s_prune":              "runs/detect/yolov9s_prune/weights/best.pt",
    "YOLOv9s-SGC-P345_prune":     "runs/detect/yolov9s_sgc_p345_prune/weights/best.pt",
    "YOLOv9s-SGC-P45_prune":      "runs/detect/yolov9s_sgc_p45_prune/weights/best.pt",
    "YOLO11s_prune":              "runs/detect/yolov11s_prune/weights/best.pt",
    "YOLO11s-SGC-P345_prune":     "runs/detect/yolov11s_sgc_p345_prune/weights/best.pt",
    "YOLO11s-SGC-P45_prune":      "runs/detect/yolov11s_sgc_p45_prune/weights/best.pt",

    # Ablation
    "YOLOv8s-SGC_P345_AMRF_FPN":    "runs/detect/yolov8s_sgc_p345_AMRF_FPN/weights/best.pt",
    "YOLOv8s-SGC-P345_EqualWeight": "runs/detect/yolov8s_sgc_p345_EqualWeight/weights/best.pt",
    "YOLOv8s-SGC-P345_minimal":     "runs/detect/yolov8s_sgc_p345_minimal/weights/best.pt",
    "YOLOv8s-SGC-P345_NoAMRF":      "runs/detect/yolov8s_sgc_p345_NoAMRF/weights/best.pt",
    "YOLOv8s-SGC-P345_NoScale":     "runs/detect/yolov8s_sgc_p345_NoScale/weights/best.pt",
}

success = []
failed = []

for name, pt_path in MODELS.items():
    if not Path(pt_path).exists():
        print(f"[SKIP] {name}: {pt_path} not found")
        failed.append((name, "not found"))
        continue

    print(f"\n[EXPORT] {name}")
    try:
        model = YOLO(pt_path)
        model.export(format="torchscript", imgsz=IMGSZ)
        ts_path = pt_path.replace(".pt", ".torchscript")
        success.append((name, ts_path))
        print(f"  OK → {ts_path}")
    except Exception as e:
        print(f"  FAIL: {e}")
        failed.append((name, str(e)))

print("\n" + "=" * 60)
print(f"DONE: {len(success)} exported, {len(failed)} failed/skipped")
for name, path in success:
    print(f"  OK  {name}: {path}")
for name, err in failed:
    print(f"  ERR {name}: {err}")
