import cv2
import time
import logging
from ultralytics import YOLO
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load the pretrained YOLO model
model = YOLO("best.pt")

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

# Initialize counters for left side
left_back_count = 0
left_side_count = 0
left_front_count = 0

# Initialize time tracking for 5-second intervals
start_time = time.time()

# Initialize the result string for "wrong side" or "not wrong side"
side_check_result = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from source.")
        break

    # Split the frame into left and right halves
    height, width, _ = frame.shape
    left_frame = frame[:, :width // 2]
    right_frame = frame[:, width // 2:]

    # Run YOLO inference on both halves
    left_results = model(left_frame)
    # right_results = model(right_frame)

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

    # Visualize the results on the frames (both left and right)
    left_annotated_frame = left_results[0].plot()
    # right_annotated_frame = right_results[0].plot()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Overlay FPS on the left frame
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

    # Overlay the current counts (front, side, back) on the left bottom for 5 seconds
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
        # After 5 seconds, check the accumulated counts and print the result
        if left_front_count > left_back_count:
            side_check_result = "Wrong Side"
            print("Wrong Side")
        elif left_back_count > left_front_count:
            side_check_result = "Not Wrong Side"
            print("Not Wrong Side")
        
        # Reset the counts after 5 seconds and restart the count accumulation
        left_back_count = 0
        left_side_count = 0
        left_front_count = 0
        start_time = time.time()  # Reset the start time for the next 5 seconds period

    # Display the side check result on the screen (bottom middle)
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

    # Overlay counts on the right side
    # right_back_count = 0
    # right_side_count = 0
    # right_front_count = 0
    # for detection in right_results[0].boxes:
    #     class_id = int(detection.cls)
    #     class_name = model.names[class_id]
    #     if 'back' in class_name:
    #         right_back_count += 1
    #     elif 'side' in class_name:
    #         right_side_count += 1
    #     elif 'front' in class_name:
    #         right_front_count += 1

    # cv2.putText(
    #     right_annotated_frame, 
    #     f"Back: {right_back_count}", 
    #     (10, 60), 
    #     cv2.FONT_HERSHEY_SIMPLEX, 
    #     1, 
    #     (0, 255, 255), 
    #     2, 
    #     cv2.LINE_AA
    # )

    # cv2.putText(
    #     right_annotated_frame, 
    #     f"Side: {right_side_count}", 
    #     (10, 90), 
    #     cv2.FONT_HERSHEY_SIMPLEX, 
    #     1, 
    #     (255, 255, 0), 
    #     2, 
    #     cv2.LINE_AA
    # )

    # cv2.putText(
    #     right_annotated_frame, 
    #     f"Front: {right_front_count}", 
    #     (10, 120), 
    #     cv2.FONT_HERSHEY_SIMPLEX, 
    #     1, 
    #     (255, 0, 0), 
    #     2, 
    #     cv2.LINE_AA
    # )

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

    # Stack the frames side by side
    final_frame = cv2.hconcat([left_annotated_frame, right_frame])

    # Display the annotated frame
    cv2.imshow("YOLO Inference", final_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
