import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO model on a dataset split (test/val/train)"
    )

    # path to trained/pruned weights
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/yolov5/weights/best.pt",
        help="Path to weights file, e.g. runs/detect/exp/weights/best.pt",
    )

    # data.yaml: must define train/val/test splits
    parser.add_argument(
        "--data",
        type=str,
        default="c2a_yolo.yaml",
        help="Path to data config file (data.yaml)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )    

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: 'cpu', '0', '0,1', ...",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate: train / val / test (must be defined in data.yaml)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize model
    model = YOLO(args.model)

    # Run evaluation
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,  # important: 'test' here is the held-out test split
    )

    # print main metrics
    # metrics.box.map: mAP@50-95, metrics.box.map50: mAP@50
    print("====================================")
    print(f"Evaluation finished on split: {args.split}")
    try:
        print(f"mAP@50     : {metrics.box.map50:.4f}")
        print(f"mAP@50-95  : {metrics.box.map:.4f}")
    except Exception:
        # in case the Ultralytics version has a different metrics structure
        print(metrics)


if __name__ == "__main__":
    main()
