import cv2
import numpy as np

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

def filter_detections_by_half(detections, image_width):
    """
    Split detections into left and right halves of the frame.
    """
    left_detections = [d for d in detections if d[0] + d[2] // 2 < image_width // 2]
    right_detections = [d for d in detections if d[0] + d[2] // 2 >= image_width // 2]
    return left_detections, right_detections

def count_orientations(detections):
    """
    Count the different orientations of vehicles (front, back, side).
    """
    front_count = 0
    back_count = 0
    side_count = 0

    for _, _, _, _, label, _ in detections:
        if "front" in label:
            front_count += 1
        elif "back" in label:
            back_count += 1
        elif "side" in label:
            side_count += 1

    return front_count, back_count, side_count

def detect_and_count_orientation(video_path, weights_path, config_path, class_names_path, output_video_path):
    """
    Main function to process the video, count the vehicle orientations, and save the output.
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

    # Get video properties for saving the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        image_height, image_width = frame.shape[:2]

        # Run YOLO inference
        detections = run_yolo_inference(net, frame, class_names)

        # Filter detections for left and right halves of the frame
        left_detections, right_detections = filter_detections_by_half(detections, image_width)

        # Count the different orientations in both halves
        left_front, left_back, left_side = count_orientations(left_detections)
        right_front, right_back, right_side = count_orientations(right_detections)

        # Prepare the text to display on the frame
        left_text = f"Left side: Front={left_front}, Back={left_back}, Side={left_side}"
        right_text = f"Right side: Front={right_front}, Back={right_back}, Side={right_side}"

        # Display the text in red color at the top-left corner
        cv2.putText(frame, left_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, right_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display the frame with bounding boxes
        for x, y, w, h, label, _ in detections:
            color = (0, 255, 0) if "back" in label or "side" in label else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow("Vehicle Orientation Detection", frame)

        # Exit condition (press 'q' to exit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Paths to model and input files
video_path = "input_video.mp4"  # Replace with your video file path
weights_path = "yolov4-tiny-custom_best.weights"  # Replace with your YOLOv4-Tiny weights file
config_path = "yolov4-tiny-custom.cfg"  # Replace with your YOLOv4-Tiny config file
class_names_path = "classes.txt"  # Replace with your class names file
output_video_path = "output_video.avi"  # Output video path

# Run detection, count orientations, and save video
detect_and_count_orientation(video_path, weights_path, config_path, class_names_path, output_video_path)






# import cv2
# import numpy as np

# def load_yolo_model(weights_path, config_path):
#     """
#     Load YOLOv4-Tiny model.
#     """
#     net = cv2.dnn.readNet(weights_path, config_path)
#     return net

# def run_yolo_inference(net, frame, class_names, conf_threshold=0.5, nms_threshold=0.4):
#     """
#     Run inference on a frame and return detections.
#     """
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)

#     output_layers = net.getUnconnectedOutLayersNames()
#     outputs = net.forward(output_layers)

#     h, w = frame.shape[:2]
#     boxes, confidences, class_ids = [], [], []

#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > conf_threshold:
#                 center_x, center_y, width, height = (
#                     int(detection[0] * w), int(detection[1] * h),
#                     int(detection[2] * w), int(detection[3] * h)
#                 )
#                 x = int(center_x - width / 2)
#                 y = int(center_y - height / 2)
#                 boxes.append([x, y, width, height])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
#     detections = []

#     if len(indices) > 0:
#         for i in indices.flatten():
#             x, y, w, h = boxes[i]
#             detections.append((x, y, w, h, class_names[class_ids[i]], confidences[i]))

#     return detections

# def filter_detections_by_half(detections, image_width):
#     """
#     Split detections into left and right halves of the frame.
#     """
#     left_detections = [d for d in detections if d[0] + d[2] // 2 < image_width // 2]
#     right_detections = [d for d in detections if d[0] + d[2] // 2 >= image_width // 2]
#     return left_detections, right_detections

# def count_orientations(detections):
#     """
#     Count the different orientations of vehicles (front, back, side).
#     """
#     front_count = 0
#     back_count = 0
#     side_count = 0

#     for _, _, _, _, label, _ in detections:
#         if "front" in label:
#             front_count += 1
#         elif "back" in label:
#             back_count += 1
#         elif "side" in label:
#             side_count += 1

#     return front_count, back_count, side_count

# def detect_and_count_orientation(video_path, weights_path, config_path, class_names_path):
#     """
#     Main function to process the video and count the vehicle orientations.
#     """
#     # Load class names
#     with open(class_names_path, 'r') as f:
#         class_names = [line.strip() for line in f.readlines()]

#     # Load YOLO model
#     net = load_yolo_model(weights_path, config_path)

#     # Open video
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to read frame.")
#             break

#         image_height, image_width = frame.shape[:2]

#         # Run YOLO inference
#         detections = run_yolo_inference(net, frame, class_names)

#         # Filter detections for left and right halves of the frame
#         left_detections, right_detections = filter_detections_by_half(detections, image_width)

#         # Count the different orientations in both halves
#         left_front, left_back, left_side = count_orientations(left_detections)
#         right_front, right_back, right_side = count_orientations(right_detections)

#         # Prepare the text to display on the frame
#         left_text = f"Left side: Front={left_front}, Back={left_back}, Side={left_side}"
#         right_text = f"Right side: Front={right_front}, Back={right_back}, Side={right_side}"

#         # Display the text in red color at the top-left corner
#         cv2.putText(frame, left_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#         cv2.putText(frame, right_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         # Display the frame with bounding boxes
#         for x, y, w, h, label, _ in detections:
#             color = (0, 255, 0) if "back" in label or "side" in label else (0, 0, 255)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, f"{label}", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         # Display the frame
#         cv2.imshow("Vehicle Orientation Detection", frame)

#         # Exit condition (press 'q' to exit)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Paths to model and input files
# video_path = "input_video.mp4"  # Replace with your video file path
# weights_path = "yolov4-tiny-custom_best.weights"  # Replace with your YOLOv4-Tiny weights file
# config_path = "yolov4-tiny-custom.cfg"  # Replace with your YOLOv4-Tiny config file
# class_names_path = "classes.txt"  # Replace with your class names file

# # Run detection and count orientations
# detect_and_count_orientation(video_path, weights_path, config_path, class_names_path)
