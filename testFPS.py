"""
FPS Benchmark trên VisDrone images thật.
Chạy: python benchmark_fps.py

SỬA các biến ở đầu file cho phù hợp.
"""

import torch
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# SỬA CÁC BIẾN NÀY
# ============================================================
VISDRONE_VAL = "/home/manhhv/AI/datasets/VisDrone/VisDrone2019-DET-test-dev/images"  # thư mục ảnh val
IMGSZ = 640
N_WARMUP = 20       # số lần warmup (bỏ kết quả)
N_RUNS = 200        # số lần đo (càng nhiều càng chính xác)
DEVICE = 0          # GPU index

# Danh sách model cần benchmark
MODELS = {
    # Trước pruning
    "YOLOv5s":              "runs/detect/yolov5s_VisDrone/weights/best.pt",
    # "YOLOv5s-SGC-P345":     "runs/detect/yolov5s_sgc_p345/weights/best.pt",
    # "YOLOv5s-SGC-P45":      "runs/detect/yolov5s_sgc_p45/weights/best.pt",
    "YOLOv8s":              "runs/detect/yolov8s_VisDrone/weights/best.pt",
    "YOLOv8s-SGC-P345":     "runs/detect/yolov8s_sgc_p345_VisDrone/weights/best.pt",
    "YOLOv8s-SGC-P45":      "runs/detect/yolov8s_sgc_p45_VisDrone/weights/best.pt",
    "YOLOv9s":              "runs/detect/yolov9s_VisDrone/weights/best.pt",
    # "YOLOv9s-SGC-P345":     "runs/detect/yolov9s_sgc_p345/weights/best.pt",
    # "YOLOv9s-SGC-P45":      "runs/detect/yolov9s_sgc_p45/weights/best.pt",
    "YOLO11s":              "runs/detect/yolov11s_VisDrone/weights/best.pt",
    # "YOLO11s-SGC-P345":     "runs/detect/yolov11s_sgc_p345/weights/best.pt",
    # "YOLO11s-SGC-P45":      "runs/detect/yolov11s_sgc_p345/weights/best.pt",

    # # Sau pruning
    # "YOLOv5s_prune":              "runs/detect/yolov5s_prune/weights/best.pt",
    # "YOLOv5s-SGC-P345_prune":     "runs/detect/yolov5s_sgc_p345_prune/weights/best.pt",
    # "YOLOv5s-SGC-P45_prune":      "runs/detect/yolov5s_sgc_p45_prune/weights/best.pt",
    # "YOLOv8s_prune":              "runs/detect/yolov8s_prune/weights/best.pt",
    # "YOLOv8s-SGC-P345_prune":     "runs/detect/yolov8s_sgc_p345_prune/weights/best.pt",
    # "YOLOv8s-SGC-P45_prune":      "runs/detect/yolov8s_sgc_p45_prune/weights/best.pt",
    # "YOLOv9s_prune":              "runs/detect/yolov9s_prune/weights/best.pt",
    # "YOLOv9s-SGC-P345_prune":     "runs/detect/yolov9s_sgc_p345_prune/weights/best.pt",
    # "YOLOv9s-SGC-P45_prune":      "runs/detect/yolov9s_sgc_p45_prune/weights/best.pt",
    # "YOLO11s_prune":              "runs/detect/yolov11s_prune/weights/best.pt",
    # "YOLO11s-SGC-P345_prune":     "runs/detect/yolov11s_sgc_p345_prune/weights/best.pt",
    # "YOLO11s-SGC-P45_prune":      "runs/detect/yolov11s_sgc_p45_prune/weights/best.pt",

    # # Ablation
    # "YOLOv8s_sgc_p345_AMRF_FPN":    "runs/detect/yolov8s_sgc_p345_AMRF_FPN/weights/best.pt",
    # "YOLOv8s_sgc_p345_EqualWeight": "runs/detect/yolov8s_sgc_p345_EqualWeight/weights/best.pt",
    # "YOLOv8s_sgc_p345_minimal":     "runs/detect/yolov8s_sgc_p345_minimal/weights/best.pt",
    # "YOLOv8s_sgc_p345_NoAMRF":      "runs/detect/yolov8s_sgc_p345_NoAMRF/weights/best.pt",
    # "YOLOv8s_sgc_p345_NoScale":     "runs/detect/yolov8s_sgc_p345_NoScale/weights/best.pt",
}

# ============================================================
# BENCHMARK
# ============================================================
# Lấy danh sách ảnh val
img_dir = Path(VISDRONE_VAL)
img_list = sorted(list(img_dir.glob("*.jpg")))
if not img_list:
    img_list = sorted(list(img_dir.glob("*.png")))
if not img_list:
    print(f"ERROR: Không tìm thấy ảnh trong {VISDRONE_VAL}")
    exit(1)

print(f"Found {len(img_list)} images in {VISDRONE_VAL}")
print(f"Warmup: {N_WARMUP}, Runs: {N_RUNS}")
print(f"Image size: {IMGSZ}")
print(f"Device: CUDA:{DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
print("=" * 70)

results = []

for name, model_path in MODELS.items():
    if not Path(model_path).exists():
        print(f"[SKIP] {name}: {model_path} not found")
        continue

    print(f"\nBenchmarking: {name}")
    model = YOLO(model_path)

    # Lấy N_RUNS ảnh (lặp lại nếu ít hơn)
    test_imgs = []
    for i in range(N_WARMUP + N_RUNS):
        test_imgs.append(str(img_list[i % len(img_list)]))

    # === Warmup ===
    for i in range(N_WARMUP):
        model.predict(test_imgs[i], imgsz=IMGSZ, verbose=False, device=DEVICE)
    torch.cuda.synchronize()

    # === Đo từng ảnh ===
    latencies = []
    for i in range(N_RUNS):
        img_path = test_imgs[N_WARMUP + i]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.predict(img_path, imgsz=IMGSZ, verbose=False, device=DEVICE)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)

    # Bỏ 5% đầu và 5% cuối (outliers)
    latencies_trimmed = np.sort(latencies)[int(N_RUNS*0.05):int(N_RUNS*0.95)]

    avg_ms = latencies_trimmed.mean()
    std_ms = latencies_trimmed.std()
    fps = 1000.0 / avg_ms

    results.append({
        "name": name,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "fps": fps,
        "min_ms": latencies_trimmed.min(),
        "max_ms": latencies_trimmed.max(),
    })

    print(f"  {avg_ms:.1f} ± {std_ms:.1f} ms  |  FPS: {fps:.1f}  |  range: [{latencies_trimmed.min():.1f}, {latencies_trimmed.max():.1f}] ms")

    # Giải phóng GPU memory
    del model
    torch.cuda.empty_cache()

    # Chờ GPU nguội 3 giây giữa các model
    time.sleep(3)

# ============================================================
# TÓM TẮT
# ============================================================
print("\n" + "=" * 70)
print(f"FPS BENCHMARK RESULTS — {torch.cuda.get_device_name(DEVICE)}")
print(f"Image size: {IMGSZ}, Runs: {N_RUNS} (trimmed 5%-95%)")
print("=" * 70)
print(f"{'Model':<25s} {'Avg(ms)':>8s} {'±Std':>7s} {'FPS':>7s}")
print("-" * 50)
for r in results:
    print(f"{r['name']:<25s} {r['avg_ms']:>7.1f}  {r['std_ms']:>6.1f}  {r['fps']:>6.1f}")

# Lưu CSV
csv_path = "benchmark_fps_results.csv"
with open(csv_path, "w") as f:
    f.write("Model,Avg_ms,Std_ms,FPS,Min_ms,Max_ms\n")
    for r in results:
        f.write(f"{r['name']},{r['avg_ms']:.2f},{r['std_ms']:.2f},{r['fps']:.1f},{r['min_ms']:.2f},{r['max_ms']:.2f}\n")
print(f"\nResults saved to {csv_path}")