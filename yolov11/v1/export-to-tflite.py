from ultralytics import YOLO
# Load the model
model = YOLO("best.pt")  
# Export with INT8 quantization
model.export(format="tflite", int8=True)