from ultralytics import YOLO

# Load a trained model
model = YOLO("yolo11.pt")

# Export to RKNN format
model.export(format="rknn")
