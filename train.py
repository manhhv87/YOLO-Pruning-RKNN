import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO model with Ultralytics using command-line arguments"
    )

    # Các tham số chính
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5.yaml",
        help="Đường dẫn file model (yaml hoặc weights), vd: yolov5.yaml hoặc best.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="c2a_yolo.yaml",
        help="Đường dẫn file data config (data.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Số epoch train",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh input (img size)",
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
        help="Thiết bị: 'cpu', '0', '0,1'...",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov5",
        help="Tên experiment (tên folder lưu weights)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Bật prune (mặc định là False, chỉ bật khi thêm --prune)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Khởi tạo model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,  # '0' hoặc '0,1' hoặc 'cpu'
        name=args.name,
        prune=args.prune,
    )

    # Nếu muốn in gì đó sau khi train:
    print("Training finished.")
    print(f"Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
