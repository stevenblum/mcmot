'''
source /home/scblum/Projects/testbed_cv/.venv5/bin/activate

cvat-agent create-native \
    --function-file /home/scblum/Projects/testbed_cv/cvat_aa/cvat_aa_yolo_tracking.py 
    --name "YOLO_Tracker"

'''


#MODEL_PATH = "/home/scblum/Projects/testbed_cv/train_yolo/saved_models/part_detection_25_10_13_11_20/weights/best.pt"   # or specify a path
#MODEL_PATH = "/home/scblum/Projects/testbed_cv/train_yolo/saved_models/part_detection_25_10_07_14_28/weights/best.pt"   # or specify a path
MODEL_PATH = "LATEST"

CONFIDENCE_THRESHOLD = 0.2


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
import numpy as np
import logging

# Set up logging FIRST, before any other logging calls
log_file_path = '/home/scblum/Projects/testbed_cv/cvat_aa/cvat_tracking.log'
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Also enable print statements
DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        message = f"[CVAT_TRACKING] {' '.join(map(str, args))}"
        print(message, **kwargs)
        # Also write to log file immediately
        logger.info(message)

# Test logging immediately
debug_print(f"Logging initialized. Log file: {log_file_path}")
logger.info("Logger initialized successfully")

# Write a test message to verify file creation
try:
    with open(log_file_path, 'a') as f:
        f.write(f"=== CVAT Tracking Log Started ===\n")
    print(f"✓ Log file created/verified at: {log_file_path}")
except Exception as e:
    print(f"✗ Failed to create log file: {e}")

if MODEL_PATH == "LATEST":
    model_weights_path = util.find_latest_model()[0]
else:
    model_weights_path = MODEL_PATH
model = YOLO(model_weights_path)
model.info()
model.conf = CONFIDENCE_THRESHOLD  # Confidence threshold
model.classes = [0, 1, 2, 3, 4, 5]
tracker = sv.ByteTrack(frame_rate=2)
smoother = sv.DetectionsSmoother()

# MUST BE TrackingFunctionSpec, DO NOT CHANGE
spec = cvataa.TrackingFunctionSpec(supported_shape_types=["rectangle"])

'''init_tracking_state must analyze the shape and create a state object containing any information 
that the AA function will need to predict its location on a subsequent image. It must then return this object.

For my implementation, the state needs to be an int, with the YOLO tracking number
'''
def init_tracking_state(context, pp_image, shape):
    return None

'''
preprocess_image, if implemented, must accept the following parameters:

context (TrackingFunctionContext). This is currently a dummy object and should be ignored. In future versions, this may contain additional information.

image (PIL.Image.Image). An image that will be used to either start or continue tracking.

preprocess_image must perform any analysis on the image that the function can perform independently of the shapes being tracked and return an object representing the results of that analysis. This object will be passed as pp_image to init_tracking_state and track.
'''

def preprocess_image(context, image):
    return image

'''
track must be a function accepting the following parameters:

context (TrackingFunctionShapeContext). An object with information about the shape being tracked. See details below.

pp_image (type varies). A preprocessed image. Consult the description of preprocess_image for more details. This image will have the same dimensions as those of the image used to create the state object.

state (type varies). The object returned by a previous call to init_tracking_state.

track must locate the shape that was used to create the state object on the new preprocessed image. If it is able to do that, it must return its prediction as a new TrackableShape object. This object must have the same value of type as the original shape.

If track is unable to locate the shape, it must return None.

track may modify state as needed to improve prediction accuracy on subsequent frames. It must not modify pp_image.
'''

def track(context, image, state):
    """Track function for continuing tracking"""
    debug_print(f"track called with state: {state}")
    logger.info(f"Tracking with state: {state}")
    
    if state is None:
        debug_print("State is None, cannot track")
        logger.warning("Cannot track - state is None")
        return None
    
    result = model(image, conf=CONFIDENCE_THRESHOLD, iou=.7, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections_tracker = tracker.update_with_detections(detections)
    
    debug_print(f"Found {len(detections)} detections, {len(detections_tracker)} tracked detections")
    
    target_track_id = state['track_id']
    debug_print(f"Looking for track_id: {target_track_id}")
    
    # Find the detection with our track ID
    if detections_tracker.tracker_id is not None:
        track_mask = detections_tracker.tracker_id == target_track_id
        debug_print(f"Track mask: {track_mask}")
        
        if np.any(track_mask):
            # Get the first matching detection
            idx = np.where(track_mask)[0][0]
            bbox = detections_tracker.xyxy[idx].cpu().numpy()
            
            debug_print(f"Found tracked object at bbox: {bbox}")
            
            # Update state
            state['last_bbox'] = bbox
            
            # Return tracked shape
            from cvat_sdk.auto_annotation import TrackableShape
            tracked_shape = TrackableShape(
                type="rectangle",
                points=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            )
            debug_print(f"Returning tracked shape: {tracked_shape.points}")
            logger.info(f"Successfully tracked object to: {bbox}")
            return tracked_shape
        else:
            debug_print(f"Track ID {target_track_id} not found in current detections")
            logger.warning(f"Lost track of object with ID {target_track_id}")
    else:
        debug_print("No tracker IDs available")
        logger.warning("No tracker IDs in current detections")
    
    debug_print("Tracking failed - returning None")
    return None

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0
