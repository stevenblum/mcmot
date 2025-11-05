from . import MCMOTUtils
from ultralytics import YOLO
import supervision as sv


class ModelPlus:
    def __init__(self, model_path, confidence_threshold, tracking_yaml_path):
        if model_path is "LATEST":
            model_path = MCMOTUtils.find_latest_model()
        self.model = YOLO(model_path,verbose=False)
        self.tracker = sv.ByteTrack(frame_rate=25)
        #self.smoother = sv.DetectionsSmoother()
        colors = sv.ColorPalette.from_hex(["#ff0303",  "#ef16ae", "#0F8807", "#4af84a", "#003CFF", "#49d2f8"])
        self.annotator_tracks = sv.BoxAnnotator(color=colors, thickness=2)
        self.annotator_detections = sv.BoxAnnotator(color=colors, thickness=1)
        self.annotator_track_tails = sv.TraceAnnotator(color=colors, thickness=2)
        self.frame = None
        self.frame_annotated = None
        self.detections = None
        self.detections_tracked = None
        #self.detections_smoothed = None
        self.frame = None
        self.frame_annotated = None
        self.confidence_threshold = confidence_threshold

    def detect_and_track(self, frame):
        result = self.model(frame, conf=self.confidence_threshold, iou=.7, agnostic_nms=True)[0]
        self.detections = sv.Detections.from_ultralytics(result)
        self.detections_tracked = self.tracker.update_with_detections(self.detections)
        #detections_smoothed = self.smoother.update_with_detections(detections_tracked)
        self.frame = frame.copy()

    def annotate_frame(self):
        af = self.annotator_tracks.annotate(self.frame.copy(), self.detections_tracked)
        af = self.annotator_detections.annotate(af, self.detections_tracked)
        self.frame_annotated = self.annotator_track_tails.annotate(af, self.detections_tracked)