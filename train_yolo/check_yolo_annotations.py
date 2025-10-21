import os
import cv2
import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np

import util

# === CONFIG ===
PROJECT_PATH = "/home/scblum/Projects/testbed_cv/train_yolo"
MODEL_PATH="LATEST"
#MODEL_PATH = os.path.join(PROJECT_PATH, "saved_models/part_detection_25_10_13_15_24/weights/best.pt")
IMAGES_DIR = os.path.join(PROJECT_PATH, "datasets/images/train")
LABELS_DIR = os.path.join(PROJECT_PATH, "datasets/labels/train")
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7

# === FUNCTIONS ===
def load_annotations(label_path):
    """Load YOLO-format annotations (class, x_center, y_center, width, height)."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, x, y, w, h = map(float, parts)
            boxes.append((int(cls), x, y, w, h))
    return boxes

def yolo_to_xyxy(box, img_w, img_h):
    """Convert YOLO normalized box to pixel coordinates (x1, y1, x2, y2)."""
    cls, x, y, w, h = box
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return cls, np.array([x1, y1, x2, y2])

def iou(boxA, boxB):
    """Compute Intersection-over-Union."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)


# === MAIN ===

if MODEL_PATH == "LATEST":
    from util import find_latest_model
    MODEL_PATH = find_latest_model()[0]
model = YOLO(MODEL_PATH)

for img_name in sorted(os.listdir(IMAGES_DIR)):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(IMAGES_DIR, img_name)
    label_path = os.path.join(LABELS_DIR, os.path.splitext(img_name)[0] + ".txt")

    frame_bgr = cv2.imread(img_path)
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb_labeled = frame_rgb.copy()

    # Load/Plot Annotations
    annotations = [yolo_to_xyxy(b, w, h) for b in load_annotations(label_path)]
    if not annotations:
        print(f"Note: No annotations for {img_name}")
        continue 
    image = cv2.imread(img_path)
    results = model.predict(image, verbose=False, conf=CONFIDENCE_THRESHOLD)
    detections = sv.Detections.from_ultralytics(results[0])
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Annotate detections
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    #annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    sv.plot_image(annotated_image)

    '''
    for d_cls, d_box in det_boxes:
        matched = False
        for i, (a_cls, a_box) in enumerate(annotations):
            if i in matched_annotations:
                continue
            if iou(d_box, a_box) > IOU_THRESHOLD:
                matched_annotations.add(i)
                matched = True
                if d_cls != a_cls:
                    print(f"mismatch: {img_name}, ann_class={a_cls}, det_class={d_cls}, box={a_box.astype(int).tolist()}")
                break
        if not matched:
            print(f"missing: {img_name}, class={d_cls}, box={d_box.astype(int).tolist()}")

    for i, (a_cls, a_box) in enumerate(annotations):
        if i not in matched_annotations:
            print(f"false: {img_name}, class={a_cls}, box={a_box.astype(int).tolist()}")
    '''