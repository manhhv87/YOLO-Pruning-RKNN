from ultralytics import YOLO

model = YOLO("yolov5.yaml")
results = model.train(
    data="corn.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device=[0],
    name="yolov5",
    prune=False,
)
