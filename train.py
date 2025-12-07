from ultralytics import YOLO

model = YOLO("yolov5.yaml")
results = model.train(
    data="c2a_yolo.yaml",
    epochs=200,
    imgsz=640,
    batch=32,
    device=[0],
    name="yolov5",
    prune=False,
)
