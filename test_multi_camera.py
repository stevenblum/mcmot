#  python3 test_multi_camera.py
import os
import sys
import builtins
import time

_log = builtins.print

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
VENV_ROOT = os.path.join(PROJECT_ROOT, ".venv")

# Force this script to run with the project venv interpreter.
if os.path.exists(VENV_PYTHON):
    current_prefix = os.path.realpath(sys.prefix)
    expected_prefix = os.path.realpath(VENV_ROOT)
    if current_prefix != expected_prefix:
        _log(f"Re-launching with project interpreter: {VENV_PYTHON}")
        os.execv(VENV_PYTHON, [VENV_PYTHON, __file__, *sys.argv[1:]])

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mcmot.MCMOTracker import MCMOTracker
from mcmot.MCMOTUtils import MCMOTUtils
import cv2

def _camera_stream_ok(device_id, probe_reads=3):
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        cap.release()
        return False
    ok = False
    for _ in range(probe_reads):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok = True
            break
    cap.release()
    return ok


'''
#LAB
MODEL_PATH = util.find_latest_model()[0]
CAMERA_DEVICE_IDS = [0,6]  # v4l2-ctl --list-devices
CAMERA_NAMES = ["logitech_1","logitech_1"]
'''

# HOME

MODEL_PATH = os.path.join(SRC_DIR, "mcmot", "models", "parts", "parts_251016_small.pt")
CAMERA_DEVICE_IDS = [5,6]#    "SELECT"#   # v4l2-ctl --list-devices
CAMERA_NAMES = ["logitech_1","logitech_1"]
CALIBRATION_METHOD = "auto"#   "landmarks"#   "aruco"#
ARUCO_POSITIONS = "square4" if CALIBRATION_METHOD == "aruco" else None
MESH_DISPLAY = False
MESH_EXPORT_PATH = "./artifacts/scene_mesh.glb"
DISPLAY_OVERHEAD = False
STRICT_CAMERA_IDS = True
###################################################################

HIGHGUI_AVAILABLE = MCMOTUtils.has_highgui()

if not HIGHGUI_AVAILABLE and CALIBRATION_METHOD in {"aruco", "landmarks"}:
    display_var = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or "<unset>"
    _log(
        "OpenCV GUI windows are unavailable in this session "
        f"(DISPLAY/WAYLAND={display_var}). "
        "Interactive calibration ('aruco'/'landmarks') requires a reachable desktop display."
    )
    _log("Falling back to CALIBRATION_METHOD=None (no interactive calibration).")
    CALIBRATION_METHOD = None
    ARUCO_POSITIONS = None

if MODEL_PATH == "LATEST":
    try:
        MODEL_PATH = MCMOTUtils.find_latest_model()[0]
    except FileNotFoundError as exc:
        _log(str(exc))
        _log("Update MODEL_PATH in test_multi_camera.py to a specific .pt weights file and retry.")
        sys.exit(1)
elif not os.path.exists(MODEL_PATH):
    _log(f"Model file not found: {MODEL_PATH}")
    _log("Check src/mcmot/models/parts and update MODEL_PATH.")
    sys.exit(1)

if CAMERA_DEVICE_IDS == "SELECT":
    CAMERA_DEVICE_IDS =  MCMOTUtils.get_camera_number(num_cameras=2)
else:
    requested_ids = list(CAMERA_DEVICE_IDS)
    invalid_ids = [cid for cid in requested_ids if not _camera_stream_ok(cid)]
    if invalid_ids:
        if STRICT_CAMERA_IDS:
            _log(
                f"Requested camera IDs are not open/readable: {invalid_ids}. "
                "Update CAMERA_DEVICE_IDS in test_multi_camera.py."
            )
            sys.exit(1)
        _log(
            f"Requested camera IDs are not open/readable: {invalid_ids}. "
            "Falling back to selection scan."
        )
        CAMERA_DEVICE_IDS = MCMOTUtils.get_camera_number(num_cameras=2)

if not isinstance(CAMERA_DEVICE_IDS, (list, tuple)) or len(CAMERA_DEVICE_IDS) == 0:
    _log("No cameras selected. Exiting.")
    sys.exit(1)

if len(CAMERA_DEVICE_IDS) < 2 and MESH_DISPLAY:
    _log("Scene mesh display requires 2 cameras; disabling mesh display.")
    MESH_DISPLAY = False

if CALIBRATION_METHOD == "landmarks":
    _log("Landmark calibration mode selected.")
    _log("For each camera: click landmark -> press SPACE/ENTER -> enter x,y,z -> press 'c' when done.")
elif CALIBRATION_METHOD == "auto":
    _log("Auto calibration mode selected.")
    _log("Estimating camera extrinsics from matched points across camera frames (RANSAC).")

mct = MCMOTracker(
    MODEL_PATH,
    CAMERA_DEVICE_IDS,
    CAMERA_NAMES,
    confidence_threshold=.05,
    tracker_yaml_path="./config/bytetrack.yaml",
    aruco_positions=ARUCO_POSITIONS,
    calibration_method=CALIBRATION_METHOD,
    mesh_display=MESH_DISPLAY,
    mesh_export_path=MESH_EXPORT_PATH,
)
try:
    mct.select_cameras()
except RuntimeError as exc:
    _log(str(exc))
    _log("Run `v4l2-ctl --list-devices` and update CAMERA_DEVICE_IDS to valid indices.")
    sys.exit(1)

running = True
try:
    while running:
        mct.capture_cameras_frames()
        mct.update_cameras_tracks()
        mct.match_global_tracks()
        if mct.mesh_display:
            mct.update_scene_mesh_display()

        overhead = mct.plot_overhead() if DISPLAY_OVERHEAD else None

        for camera_number in range(len(mct.cameras)):
            frame = mct.annotated_frame_from_camera(camera_number)
            if frame is not None and HIGHGUI_AVAILABLE:
                cv2.imshow(f"Camera {camera_number}", frame)

        if HIGHGUI_AVAILABLE and DISPLAY_OVERHEAD and overhead is not None:
            cv2.imshow("Overhead", overhead)

        if HIGHGUI_AVAILABLE and (cv2.waitKey(1) & 0xFF == ord('q')):
            running = False
        elif not HIGHGUI_AVAILABLE:
            time.sleep(0.005)
except KeyboardInterrupt:
    _log("Interrupted by user.")
finally:
    for camera in mct.cameras.values():
        cap = getattr(camera, "cap", None)
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()
