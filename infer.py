from ultralytics import YOLO

# Load a model (original or pruned)
model = YOLO('yolo11.pt')  # or model = YOLO('pruned.pt')

# Run inference
model.predict(
    'ultralytics/assets/bus.jpg',  # Input image path
    save=True,                     # Save results
    device=[0],                    # Use GPU 0
    line_width=2                   # Detection box line width
)