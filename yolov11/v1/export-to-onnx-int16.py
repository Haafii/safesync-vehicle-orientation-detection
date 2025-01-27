from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", half=True)  # Enable FP16 quantization