'''

source /home/scblum/Projects/testbed_cv/.venv5/bin/activate

cvat-cli --server-host http://localhost:8080 \
  --auth scblum2:Basketball1! \
  task auto-annotate 32\
  --function-file /home/scblum/Projects/testbed_cv/cvat_aa/cvat_aa_yolo_simple.py  \
  --clear-existing      \
  --allow-unmatched-labels
  

cvat-cli function create-native "YOLOv11 Custom" \
    --function-module cvat_sdk.auto_annotation.functions.torchvision_detection \
    -p model_name=str:fasterrcnn_resnet50_fpn_v2

# Might Use These Later

-p model=str:/home/scblum/Projects/testbed_cv/yolo_training/part_detection_25_10_07_14_28/weights/best.pt \

'''


#MODEL_PATH = "/home/scblum/Projects/testbed_cv/train_yolo/saved_models/part_detection_25_10_13_11_20/weights/best.pt"   # or specify a path
#MODEL_PATH = "/home/scblum/Projects/testbed_cv/train_yolo/saved_models/part_detection_25_10_07_14_28/weights/best.pt"   # or specify a path
MODEL_PATH = "LATEST"

CONFIDENCE_THRESHOLD = 0.1


import PIL.Image
from ultralytics import YOLO
import supervision as sv
import cvat_sdk.auto_annotation as cvataa
import cvat_sdk.models as models
from cvat_sdk.models import ShapeType
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util



if MODEL_PATH == "LATEST":
    model_weights_path = util.find_latest_model()[0]
else:
    model_weights_path = MODEL_PATH
model = YOLO(model_weights_path)
model.info()
model.conf = CONFIDENCE_THRESHOLD  # Confidence threshold
model.classes = [0, 1, 2, 3, 4, 5]
tracker = sv.ByteTrack(frame_rate=5)
smoother = sv.DetectionsSmoother()

# Build the label specification for CVAT
spec = cvataa.DetectionFunctionSpec(
    labels=[cvataa.label_spec(name, id) for id, name in model.names.items()],
)

def _yolo_to_cvat_detections(detections):
    # For "Detections" from Supervision
    for box, cls_id in zip(detections.xyxy, detections.class_id):
        yield cvataa.rectangle(int(cls_id.item()), [float(p.item()) for p in box])

# --- Main detection function for CVAT ---
def detect(context, image):

    result = model(image, conf=CONFIDENCE_THRESHOLD , iou=.7, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections_tracker = tracker.update_with_detections(detections)
    detections_smoothed = smoother.update_with_detections(detections_tracker)

    # Return detections as rectangles (CVAT will handle track creation)
    return list(_yolo_to_cvat_detections(detections_tracker))
