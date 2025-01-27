from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
model.train(data="dataset_custom.yaml", epochs=200, batch=32, imgsz=640, device=0, plots=True, workers=0, save_json=True, save_hybrid=True, cache = True)