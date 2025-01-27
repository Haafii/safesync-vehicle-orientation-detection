import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import time
import logging
import os
from datetime import datetime

logging.getLogger("ultralytics").setLevel(logging.ERROR)

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load COCO class names
with open("class.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv8 model
model = YOLO("best.pt")

# Initialize variables for FPS calculation
prev_time = 0
count = 0

# Initialize variables for 5-second accumulation
left_accum_back, left_accum_side, left_accum_front = 0, 0, 0
accum_start_time = time.time()
result_message = ""  # To display "Wrong Side" or "Not Wrong Side"

# Create "videos" folder if it doesn't exist
if not os.path.exists('videos'):
    os.makedirs('videos')

# Get the current date and time for the video filename
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_filename = f"videos/{current_time}.mp4"

# Define the video writer
frame_width = 1280  # Width of the frame
frame_height = 720  # Height of the frame
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame_width, frame_height))

while True:
    frame = picam2.capture_array()
    
    count += 1
    if count % 1 != 0:
        continue
    # frame = cv2.flip(frame, 1)
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 360, 1)
    frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))


    # Get the current time for FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Calculate the width of each half
    frame_width = frame.shape[1]
    half_width = frame_width // 2

    # Initialize counters for back, side, and front objects in left and right halves (per frame)
    left_back_count, left_side_count, left_front_count = 0, 0, 0
    right_back_count, right_side_count, right_front_count = 0, 0, 0

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, imgsz=240)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            mid_x = (x1 + x2) // 2  # Calculate the mid-point of the box to determine which half it belongs to

            # Determine the half and update counts based on the class name
            if mid_x < half_width:  # Left half
                if "back" in c:
                    left_back_count += 1
                elif "side" in c:
                    left_side_count += 1
                elif "front" in c:
                    left_front_count += 1
            else:  # Right half
                if "back" in c:
                    right_back_count += 1
                elif "side" in c:
                    right_side_count += 1
                elif "front" in c:
                    right_front_count += 1

            # Draw bounding boxes and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

    # Accumulate counts for 5 seconds
    left_accum_back += left_back_count
    left_accum_side += left_side_count
    left_accum_front += left_front_count

    # Check if 5 seconds have passed
    elapsed_time = current_time - accum_start_time
    if elapsed_time > 5:
        # After 5 seconds, check if "front" > "back" and set the result message
        if left_accum_front > left_accum_back:
            result_message = "Wrong Side"
            print("Wrong Side")  # Print to terminal
        elif left_accum_back > left_accum_front:
            result_message = "Not Wrong Side"
            print("Not Wrong Side")  # Print to terminal
        else:
            result_message = "Equal Counts"

        # Reset accumulation counters and timer
        left_accum_back, left_accum_side, left_accum_front = 0, 0, 0
        accum_start_time = current_time

    # Display the FPS on the top-left corner
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display individual counts in the top-left (left half) and top-right (right half)
    cv2.putText(frame, f"Left Back: {left_back_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Left Side: {left_side_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Left Front: {left_front_count}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Right Back: {right_back_count}", (half_width + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Right Side: {right_side_count}", (half_width + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Right Front: {right_front_count}", (half_width + 10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display accumulated counts for the left half at the bottom-left corner
    cv2.putText(frame, f"Accum Back: {left_accum_back}", (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Accum Side: {left_accum_side}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Accum Front: {left_accum_front}", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result message (e.g., "Wrong Side" or "Not Wrong Side") at the bottom-middle
    cv2.putText(frame, result_message, (frame_width // 2 - 150, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw a vertical line to visually split the frame
    cv2.line(frame, (half_width, 0), (half_width, frame.shape[0]), (255, 0, 0), 2)

    # Write the frame to the video file
    video_writer.write(frame)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object, close the display window, and release the video writer
video_writer.release()
cv2.destroyAllWindows()
