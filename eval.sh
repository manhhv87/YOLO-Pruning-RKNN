#!/bin/bash
#SBATCH --job-name=MT
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x.%j.out    # stdout
##SBATCH --error=logs/%x.%j.err     #/dev/null             # discard stderr (tqdm, warnings... if the cluster allows)

module load singularity
module load cuda

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Log GPU usage (keep it if you need; optional)
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv -l 10 > logs/gpu_memory_log.csv &
NSMI_PID=$!

IMG="$HOME/pytorch/2.0.0-cuda11.7-cudnn8"

# Reduce progress bar spam
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

echo "=== START TRAINING YOLO ==="

# 1) TRAIN
#singularity exec --nv "$IMG" \
#  python3 -u train.py \
#    --model yolov5s_sgc.yaml \
#    --data corn.yaml \
#    --epochs 200 \
#    --imgsz 640 \
#    --batch 32 \
#    --device 0 \
#    --name yolov5s_sgc

# 1) EVAL
singularity exec --nv "$IMG" \
  python eval.py \
    --model runs/detect/yolov5s_sgc_loss/weights/best.pt \
    --data corn.yaml \
    --split test \
    --imgsz 640 \
    --batch 32 \
    --device 0

echo "=== TRAINING DONE, SUMMARIZING EPOCH RESULTS ==="

# 2) USE PYTHON TO READ results.csv AND PRINT CLEAN OUTPUT
singularity exec --nv "$IMG" python3 - << 'EOF'
import csv, pathlib, sys

root = pathlib.Path("runs")
candidates = sorted(root.rglob("results.csv"))
if not candidates:
    print("Could not find results.csv", file=sys.stderr)
    sys.exit(0)

csv_path = candidates[-1]

print(f"Metrics file: {csv_path}")
print("epoch,train_box,train_cls,val_box,val_cls,mAP50,mAP50_95")

with csv_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        print("{epoch},{tbox},{tcls},{vbox},{vcls},{map50},{map5095}".format(
            epoch=row.get("epoch"),
            tbox=row.get("train/box_loss"),
            tcls=row.get("train/cls_loss"),
            vbox=row.get("val/box_loss"),
            vcls=row.get("val/cls_loss"),
            map50=row.get("metrics/mAP50(B)"),
            map5095=row.get("metrics/mAP50-95(B)"),
        ))
EOF

# 3) STOP GPU LOGGING
kill $NSMI_PID
