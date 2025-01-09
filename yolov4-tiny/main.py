import cv2
import numpy as np
import time

def load_yolo_model(weights_path, config_path):
    """
    Load YOLOv4-Tiny model.
    """
    net = cv2.dnn.readNet(weights_path, config_path)
    return net

def run_yolo_inference(net, frame, class_names, conf_threshold=0.5, nms_threshold=0.4):
    """
    Run inference on a frame and return detections.
    """
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    h, w = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y, width, height = (
                    int(detection[0] * w), int(detection[1] * h),
                    int(detection[2] * w), int(detection[3] * h)
                )
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    detections = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detections.append((x, y, w, h, class_names[class_ids[i]], confidences[i]))

    return detections

def detect_objects(video_path, weights_path, config_path, class_names_path, output_path="output_video.mp4"):
    """
    Main function to process the video, detect objects, display FPS, and save output video.
    """
    # Load class names
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load YOLO model
    net = load_yolo_model(weights_path, config_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Video writer for saving output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Use the video's FPS or default to 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Press 'q' to exit.")

    # Start time for FPS calculation
    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or failed to capture.")
            break

        # Run YOLO inference
        detections = run_yolo_inference(net, frame, class_names)

        # Draw bounding boxes
        for x, y, w, h, label, conf in detections:
            color = (0, 255, 0)  # Green for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Calculate FPS
        current_time = time.time()
        fps = int(1 / (current_time - prev_frame_time)) if prev_frame_time > 0 else 0
        prev_frame_time = current_time

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow("Object Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Paths to model and input files
video_path = "input_video.mp4"  # Replace with your video file path
weights_path = "yolov4-tiny-custom_best.weights"  # Replace with your YOLOv4-Tiny weights file
config_path = "yolov4-tiny-custom.cfg"  # Replace with your YOLOv4-Tiny config file
class_names_path = "vehicle_classes.txt"  # Replace with your class names file

# Run detection and save output
output_path = "output_video.mp4"  # Path to save the output video
detect_objects(video_path, weights_path, config_path, class_names_path, output_path)
