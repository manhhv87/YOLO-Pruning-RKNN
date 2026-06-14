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
        default="runs/detect/yolov5s_sgc_loss/weights/best.pt",
        help="Path to weights file, e.g. runs/detect/exp/weights/best.pt",
    )

    # data.yaml: must define train/val/test splits
    parser.add_argument(
        "--data",
        type=str,
        default="corn.yaml",
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

    parser.add_argument(
        "--dir",
        type=str,
        default="runs/detect",
        help="Root directory to save results (e.g., /content/my_runs or runs/detect)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # load model (already trained or pruned weights)
    model = YOLO(args.model)

    # run evaluation
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,  # 'test' here is the held-out test split
        workers=4,
        project=args.dir,
    )

    # print main metrics
    print("====================================")
    print(f"Evaluation finished on split: {args.split}")
    try:
        # mAP
        print(f"mAP@50     : {metrics.box.map50:.4f}")
        print(f"mAP@50-95  : {metrics.box.map:.4f}")

        # mean Precision / Recall over all classes
        mp = metrics.box.mp
        mr = metrics.box.mr
        f1_mean = 2 * mp * mr / (mp + mr + 1e-16)  # avoid division by zero

        print(f"Precision (mp): {mp:.4f}")
        print(f"Recall    (mr): {mr:.4f}")
        print(f"F1-score      : {f1_mean:.4f}")

        # per-class Precision / Recall / F1
        p = metrics.box.p  # per-class precision (array)
        r = metrics.box.r  # per-class recall (array)
        f1_per_class = 2 * p * r / (p + r + 1e-16)

        # class names from model
        names = getattr(model, "names", None)

        print("\nPer-class metrics:")
        for i, (pi, ri, fi) in enumerate(zip(p, r, f1_per_class)):
            if isinstance(names, dict) and i in names:
                cname = names[i]
            else:
                cname = str(i)
            print(f"  Class {i} ({cname}): P={pi:.4f}, R={ri:.4f}, F1={fi:.4f}")

    except Exception as e:
        # in case the metrics structure differs between Ultralytics versions
        print("Error while reading metrics:", e)
        print(metrics)


if __name__ == "__main__":
    main()
