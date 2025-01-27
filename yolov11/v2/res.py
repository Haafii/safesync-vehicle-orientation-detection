from ultralytics import YOLO

# Load a model
model = YOLO("C:/Users/EIR/Downloads/ML MODEL/Vehicle/runs/detect/train13/weights/last.pt")  # load a partially trained model

# Resume training
results = model.train(resume=True)