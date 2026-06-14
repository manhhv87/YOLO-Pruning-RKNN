"""
Benchmark FPS tren Jetson Nano bang PyTorch CUDA.
Dung TorchScript files - khong can ultralytics.

Chay: /usr/bin/python3 benchmark_pytorch.py
"""

import torch
import time
import numpy as np
from pathlib import Path

IMGSZ = 640
N_WARMUP = 20
N_RUNS = 100
DEVICE = "cuda:0"
MODEL_DIR = "/home/manhhv/models"
RESULT_FILE = "/home/manhhv/fps_jetson.csv"

print("PyTorch {}".format(torch.__version__))
print("CUDA: {}".format(torch.cuda.get_device_name(0)))
print("Warmup: {}, Runs: {}, ImgSize: {}".format(N_WARMUP, N_RUNS, IMGSZ))
print("=" * 60)

ts_files = sorted(Path(MODEL_DIR).glob("*.torchscript"))
print("Found {} models\n".format(len(ts_files)))

results = []

for ts_path in ts_files:
    name = ts_path.stem
    print("Benchmarking: {}...".format(name), end=" ")

    try:
        model = torch.jit.load(str(ts_path), map_location=DEVICE)
        model.eval()
        model = model.to(DEVICE)

        dummy = torch.randn(1, 3, IMGSZ, IMGSZ).to(DEVICE)

        # Warmup
        with torch.no_grad():
            for _ in range(N_WARMUP):
                model(dummy)
        torch.cuda.synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(N_RUNS):
                torch.cuda.synchronize()
                t0 = time.time()
                model(dummy)
                torch.cuda.synchronize()
                t1 = time.time()
                latencies.append((t1 - t0) * 1000)

        latencies = sorted(latencies)
        trim = int(N_RUNS * 0.05)
        if trim > 0:
            latencies = latencies[trim:-trim]

        avg_ms = sum(latencies) / len(latencies)
        fps = 1000.0 / avg_ms

        print("{:.1f} ms | FPS: {:.1f}".format(avg_ms, fps))
        results.append({"name": name, "avg_ms": avg_ms, "fps": fps})

    except Exception as e:
        print("FAIL: {}".format(e))
        results.append({"name": name, "avg_ms": 0, "fps": 0})

    # Giai phong GPU
    try:
        del model
    except:
        pass
    torch.cuda.empty_cache()
    time.sleep(3)

# Ket qua
print("\n" + "=" * 60)
print("FPS BENCHMARK - Jetson Nano GPU (PyTorch CUDA)")
print("Image: {}x{}, Runs: {}".format(IMGSZ, IMGSZ, N_RUNS))
print("=" * 60)
print("{:<35s} {:>8s} {:>7s}".format("Model", "Avg(ms)", "FPS"))
print("-" * 55)
for r in results:
    print("{:<35s} {:>7.1f}  {:>6.1f}".format(r["name"], r["avg_ms"], r["fps"]))

# Luu CSV
with open(RESULT_FILE, "w") as f:
    f.write("Model,Latency_ms,FPS\n")
    for r in results:
        f.write("{},{:.2f},{:.1f}\n".format(r["name"], r["avg_ms"], r["fps"]))
print("\nSaved: {}".format(RESULT_FILE))
