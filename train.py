import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO model with Ultralytics using command-line arguments"
    )

    # Main arguments
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5.yaml",
        help="Path to model file (yaml or weights), e.g. yolov5.yaml or best.pt",
    )
    parser.add_argument(
        "--iou_type",
        type=str,
        default="ciou",
        choices=["iou", "giou", "diou", "ciou", "eiou", "siou", "alpha_iou"],
        help=(
            "IoU variant used in bbox regression loss. "
            "Options: iou, giou, diou, ciou, eiou, siou, alpha_iou"
        ),
    )
    parser.add_argument(
        "--alpha_iou",
        type=float,
        default=2.0,
        help="Exponent for alpha-IoU (used only when iou_type='alpha_iou')",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="c2a_yolo.yaml",
        help="Path to data config file (data.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
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
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: 'cpu', '0', '0,1', ...",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov5",
        help="Experiment name (folder name for saving weights)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Enable pruning (default is False, enable by adding --prune)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize model
    model = YOLO(args.model)

    # Pass IoU configuration to the model so loss.py can read it
    model.args.iou_type = args.iou_type
    model.args.alpha_iou = args.alpha_iou

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,  # '0' or '0,1' or 'cpu'
        name=args.name,
        prune=args.prune,
    )

    # Optional: print something after training
    print("Training finished.")
    print(f"Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
