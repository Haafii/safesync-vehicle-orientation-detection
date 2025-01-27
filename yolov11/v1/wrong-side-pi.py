import cv2
import time
import logging
from ultralytics import YOLO
from picamera2 import Picamera2
import numpy as np

# Configure logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Initialize the Pi camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the pretrained YOLO model
model = YOLO("best.pt")

# Initialize FPS calculation
fps = 0.0
prev_time = time.time()

# Initialize counters for left side
left_back_count = 0
left_side_count = 0
left_front_count = 0

# Initialize time tracking for 5-second intervals
start_time = time.time()

# Initialize the result string for "wrong side" or "not wrong side"
side_check_result = ""

# Initialize counter for frame processing
count = 0

while True:
    # Capture frame from Pi camera
    frame = picam2.capture_array()
    
    # Process every third frame to improve performance
    count += 1
    if count % 3 != 0:
        continue
        
    # Flip the frame if needed (adjust -1 to 0, 1, or remove if not needed)
    frame = cv2.flip(frame, -1)
    
    # Split the frame into left and right halves
    height, width, _ = frame.shape
    left_frame = frame[:, :width // 2]
    right_frame = frame[:, width // 2:]

    # Run YOLO inference on left half
    left_results = model(left_frame)

    # Count detections for left side
    for detection in left_results[0].boxes:
        class_id = int(detection.cls)
        class_name = model.names[class_id]
        if 'back' in class_name:
            left_back_count += 1
        elif 'side' in class_name:
            left_side_count += 1
        elif 'front' in class_name:
            left_front_count += 1

    # Visualize the results on the frames
    left_annotated_frame = left_results[0].plot()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Overlay FPS
    cv2.putText(
        left_annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Display counts for 5 seconds
    if current_time - start_time <= 5:
        cv2.putText(
            left_annotated_frame,
            f"Front: {left_front_count}",
            (10, height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            left_annotated_frame,
            f"Side: {left_side_count}",
            (10, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            left_annotated_frame,
            f"Back: {left_back_count}",
            (10, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
    else:
        # After 5 seconds, check counts and update result
        if left_front_count > left_back_count:
            side_check_result = "Wrong Side"
            print("Wrong Side")
        elif left_back_count > left_front_count:
            side_check_result = "Not Wrong Side"
            print("Not Wrong Side")
        
        # Reset counts and timer
        left_back_count = 0
        left_side_count = 0
        left_front_count = 0
        start_time = time.time()

    # Display side check result
    cv2.putText(
        left_annotated_frame,
        side_check_result,
        (width // 4, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if side_check_result == "Not Wrong Side" else (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    # Display current frame counts at top
    left_back_count_top = 0
    left_side_count_top = 0
    left_front_count_top = 0
    for detection in left_results[0].boxes:
        class_id = int(detection.cls)
        class_name = model.names[class_id]
        if 'back' in class_name:
            left_back_count_top += 1
        elif 'side' in class_name:
            left_side_count_top += 1
        elif 'front' in class_name:
            left_front_count_top += 1

    # Display current counts at top
    cv2.putText(
        left_annotated_frame,
        f"Back: {left_back_count_top}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        left_annotated_frame,
        f"Side: {left_side_count_top}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        left_annotated_frame,
        f"Front: {left_front_count_top}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Combine frames
    final_frame = cv2.hconcat([left_annotated_frame, right_frame])

    # Display the result
    cv2.imshow("YOLO Inference", final_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()