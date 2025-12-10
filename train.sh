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

# TRAIN
singularity exec --nv "$IMG" \
  python3 -u train.py \
    --model yolov5s_sgc.yaml \
    --iou_type ciou \
    --data corn.yaml \
    --epochs 200 \
    --imgsz 640 \
    --batch 32 \
    --device 0 \
    --name yolov5s_sgc 

# STOP GPU LOGGING
kill $NSMI_PID
