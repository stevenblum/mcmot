from ultralytics import YOLO
import supervision as sv
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path)
sys.path.insert(0, )
import util
import cv2

# Load a pre-trained YOLOv11 model (or your custom-trained model)
# Replace 'yolov11.pt' with the path to your model weights if using a custom model

MODEL_PATH = util.find_latest_model()[0]
#MODEL_PATH = "/home/scblum/Projects/testbed_cv/train_yolo/saved_models/part_detection_25_10_07_14_28/weights/best.pt"
TRACKER_PATH = "bytetrack.yaml"
model = YOLO(MODEL_PATH)

# Path to your input MP4 video file
video_path = '/home/scblum/Projects/testbed_cv/train_yolo/images/2025 06 01 testbed simulation/best_video_normal.mp4'
#video_path = '/home/scblum/Projects/testbed_cv/raw_images/2025 10 15 Hanoi4 Fixed/4 disks.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Perform object tracking on the frame
        # 'persist=True' ensures that tracks are maintained across frames
        # You can also specify a tracker config file (e.g., 'tracker="bytetrack.yaml"')

        results = model.track(frame, persist=True, conf=0.2, iou=0.3, agnostic_nms=True, tracker=TRACKER_PATH, verbose=False)

        detections = sv.Detections.from_ultralytics(results[0])

        bounding_box_annotator = sv.BoxAnnotator()
        #label_annotator = sv.LabelAnnotator()

        labels = [
            model.model.names[class_id]
            for class_id
            in detections.class_id
        ]

        annotated_frame = bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        #annotated_frame = label_annotator.annotate(
        #    scene=annotated_frame, detections=detections, labels=labels)

        # Visualize the results on the frame
        #annotated_frame = results[0].plot()

        # Annotate frame with utils.plot_detections
        #xyxy_conf_cls = util.yolo_to_xyxy_conf_cls(results[0].boxes, frame.shape[1], frame.shape[0])
        #annotated_frame = util.plot_detections(frame, xyxy_conf_cls,[i for i in range(6)], "point")

        # Display the annotated frame
        cv2.imshow("YOLOv11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()