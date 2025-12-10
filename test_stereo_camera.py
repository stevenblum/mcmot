
from mcmot.StereoCamera import StereoCamera

stereo = StereoCamera(
    primary_camera_number=0,
    primary_device_id=0,
    primary_camera_name="logitech_1",
    secondary_camera_number=1,
    secondary_device_id=1,
    secondary_camera_name="logitech_1",
    model_path=None,
    confidence_threshold=None,
    tracker_yaml_path=None,
    aruco_positions=None,
    display=True,
)

while True:
    stereo.capture_frames()
    stereo.display_frames()

