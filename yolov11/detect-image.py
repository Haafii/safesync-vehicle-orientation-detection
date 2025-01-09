from ultralytics import YOLO

# Load a pretrained YOLO11n model
# model = YOLO("best.pt")
model = YOLO("best.onnx", task="detect")


# Run inference on the source
results = model(source = "car.webp", save=True)  # list of Results objects