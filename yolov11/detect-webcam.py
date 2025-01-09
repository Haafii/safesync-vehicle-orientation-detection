# from ultralytics import YOLO

# # Load a pretrained YOLO11n model
# model = YOLO("best.pt")

# # Run inference on the source
# results = model(source=0)  # generator of Results objects


import cv2
import time
from ultralytics import YOLO

# Load the pretrained YOLO model
# model = YOLO("best.pt")
model = YOLO("best.onnx", task="detect")


# Initialize the video source (0 for webcam or path to a video file)
source = 0  # Change this to a video file path if needed
cap = cv2.VideoCapture(source)

# Check if the video capture source is opened
if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

# Initialize FPS calculation
fps = 0.0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from source.")
        break

    # Run YOLO inference on the frame
    results = model(frame)  # Predict on the current frame

    # Visualize the results on the frame
    annotated_frame = results[0].plot()  # Plot bounding boxes, labels, etc.

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Overlay FPS on the frame
    cv2.putText(
        annotated_frame, 
        f"FPS: {fps:.2f}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2, 
        cv2.LINE_AA
    )

    # Display the annotated frame
    cv2.imshow("YOLO Inference", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
