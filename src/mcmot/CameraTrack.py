class CameraTrack:
    def __init__(self, camera_number, track_id, bbox, score, class_id):
        self.camera_number = camera_number
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        self.score = score
        self.class_id = class_id

    def update(self, bbox, score):
        self.bbox = bbox
        self.score = score