from .MCMOTUtils import MCMOTUtils
from ultralytics import YOLO
import supervision as sv


class ModelPlus:
    def __init__(self, model_path, confidence_threshold, tracking_yaml_path, yolo_model=None):
        if yolo_model is not None:
            self.model = yolo_model
        elif model_path is not None:
            if model_path == "LATEST":
                model_path = MCMOTUtils.find_latest_model()[0]
            self.model = YOLO(model_path, verbose=False)
        else:
            self.model = None
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

    def update_from_result(self, result, frame):
        self.detections = sv.Detections.from_ultralytics(result)
        self.detections_tracked = self.tracker.update_with_detections(self.detections)
        self.frame = frame.copy()

    def detect_and_track(self, frame):
        if self.model is None:
            raise RuntimeError("ModelPlus has no YOLO model attached for per-camera inference.")
        result = self.model(frame, conf=self.confidence_threshold, iou=.7, agnostic_nms=True)[0]
        self.update_from_result(result, frame)

    def annotate_frame(self):
        if self.frame is None:
            self.frame_annotated = None
            return None

        if self.detections_tracked is None:
            self.frame_annotated = self.frame.copy()
            return self.frame_annotated

        af = self.annotator_tracks.annotate(self.frame.copy(), self.detections_tracked)
        af = self.annotator_detections.annotate(af, self.detections_tracked)
        self.frame_annotated = self.annotator_track_tails.annotate(af, self.detections_tracked)
        return self.frame_annotated
