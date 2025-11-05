class MultiCameraTrack:
    def __init__(self, iou_threshold=0.3):
        self.last_camera_match = {}

    def set_match(self,camera_tracks):
        match = {}
        for ct in camera_tracks:
            match[ct.camera_id] = ct.track_id

        self.last_camera_match = match
