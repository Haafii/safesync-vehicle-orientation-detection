# from ultralytics import YOLO

# # Load a pretrained YOLO11n model
# model = YOLO("best.pt")


# # Run inference on the source
# results = model(source = "input_video.mp4", save=True)  # list of Results objects



from ultralytics import YOLO
import cv2
import time

# Load a pretrained YOLO model
# model = YOLO("best.pt")
# model = YOLO("best.onnx", task="detect")
model = YOLO("best_saved_model/best_full_integer_quant.tflite", task="detect")



# Open the video file
cap = cv2.VideoCapture("input_video.mp4")

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Variables for calculating FPS
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model inference on the current frame
    results = model.predict(frame, save=False, verbose=False)

    # Annotate frame with detected objects
    annotated_frame = results[0].plot()

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Display FPS on the frame
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

    # Show the frame
    cv2.imshow("YOLO Detection", annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
