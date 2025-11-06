import cv2
import os
import time
import pathlib
import numpy as np

def find_latest_model():

    training_dir = pathlib.Path(__file__).resolve().parent.parent.parent

    print(training_dir)

    # Look for best.pt files in subdirectories
    model_files = list(training_dir.rglob("*/weights/best.pt"))
    
    # Get the most recent model based on modification timeS
    latest_model = max(model_files, key=os.path.getmtime)
    
    # Extract model name from path (e.g., "part_detection_25_10_07_12_23")
    model_name = os.path.basename(os.path.dirname(os.path.dirname(latest_model)))
    
    # Get creation time for display
    mod_time = os.path.getmtime(latest_model)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
    
    print(f"🎯 Latest model found: {model_name}")
    print(f"📁 Path: {latest_model}")

    return str(latest_model), model_name

def get_camera_number(num_cameras: int = 1, max_search: int = 8, cols: int = 4, 
                        thumb_size: tuple = (320, 240), window_name: str = "Camera Grid"):
    """
    Search for cameras up to max_search, display a grid of snapshots with camera indices
    overlaid, prompt the user in the terminal to pick one or multiple indices, then
    close windows and return the chosen index (int) for num_cameras==1 or a list of ints
    for num_cameras>1. Returns -1 (or [] for multi) if no cameras found or cancelled.
    """
    import sys
    import select

    caps = []
    frames = []
    indices = []

    # Try opening camera indices 0..max_search-1
    for i in range(max_search):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                thumb = cv2.resize(frame, thumb_size)
                frames.append(thumb)
                indices.append(i)
                caps.append(cap)
            else:
                cap.release()
        else:
            cap.release()

    if not frames:
        print("No cameras found.")
        return -1 if num_cameras == 1 else []

    # Build grid image
    n = len(frames)
    rows = (n + cols - 1) // cols
    pad_img = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
    imgs = frames.copy()
    while len(imgs) < rows * cols:
        imgs.append(pad_img.copy())

    for idx, img in enumerate(imgs[:n]):
        cam_idx = indices[idx]
        cv2.putText(img, str(cam_idx), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    row_imgs = []
    for r in range(rows):
        start = r * cols
        end = start + cols
        row_imgs.append(np.hstack(imgs[start:end]))
    grid = np.vstack(row_imgs)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, grid)
    cv2.waitKey(1)

    print(f"Found cameras: {indices}")
    if num_cameras == 1:
        print("Type the camera number to select it, or press Enter to cancel.")
    else:
        print(f"Enter {num_cameras} camera indices (comma-separated), or press Enter to cancel.")

    selected = -1 if num_cameras == 1 else []
    try:
        while True:
            cv2.imshow(window_name, grid)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                line = sys.stdin.readline().strip()
                if line == "":
                    selected = -1 if num_cameras == 1 else []
                    break
                try:
                    if num_cameras == 1:
                        val = int(line)
                        if val in indices:
                            selected = val
                            break
                        else:
                            print(f"{val} not in available cameras {indices}. Try again.")
                            continue
                    else:
                        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                        vals = []
                        for p in parts:
                            try:
                                v = int(p)
                            except ValueError:
                                v = None
                            if v is None or v not in indices:
                                v = None
                                break
                            vals.append(v)
                        if len(vals) == num_cameras and len(set(vals)) == num_cameras:
                            selected = vals
                            break
                        print(f"Please enter exactly {num_cameras} valid, unique indices from {indices}.")
                        continue
                except ValueError:
                    print("Invalid input. Enter integers (comma-separated for multiple).")
                    continue
            if (cv2.waitKey(1) & 0xFF) == 27:
                selected = -1 if num_cameras == 1 else []
                break
    except KeyboardInterrupt:
        selected = -1 if num_cameras == 1 else []

    for cap in caps:
        cap.release()
    cv2.destroyWindow(window_name)

    if (num_cameras == 1 and selected == -1) or (num_cameras > 1 and not selected):
        print("No selection made / cancelled.")
    else:
        print(f"Selected camera(s): {selected}")
    return selected

def select_cameras(num_cameras=1):
    """
    Show live previews for all available cameras up to num_cameras,
    prompt the user to select which camera indices to use,
    and return the selected indices as a list of ints.
    """
    print(f"Scanning for up to {num_cameras} cameras...")
    caps = []
    window_names = []
    available = []
    for i in range(num_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                win = f"Camera {i}"
                cv2.imshow(win, frame)
                window_names.append(win)
                available.append(i)
                caps.append(cap)
                print(f"Camera {i}: available")
            else:
                cap.release()
        else:
            cap.release()
    if not available:
        print("No cameras found.")
        return []

    print("All available cameras are displayed.")
    print("In the terminal, enter the number of cameras you want to use, then the indices (comma-separated).")
    num = 0
    selected = []
    while True:
        try:
            num = int(input(f"How many cameras do you want to use? (1-{len(available)}): "))
            if 1 <= num <= len(available):
                break
            print("Invalid number.")
        except Exception:
            print("Invalid input.")
    while True:
        selection = input(f"Enter {num} camera indices to use (comma-separated from {available}): ")
        try:
            selected = [int(x.strip()) for x in selection.split(",") if int(x.strip()) in available]
            if len(selected) == num:
                break
            print(f"Please enter exactly {num} valid indices.")
        except Exception:
            print("Invalid input.")

    # Close all windows after selection
    for cap in caps:
        cap.release()
    for win in window_names:
        cv2.destroyWindow(win)

    print(f"Selected cameras: {selected}")
    return selected


def plot_detections(frame, detections, class_names,type):
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if cls == 0 or cls == 1:
            color=(0, 0, 255) # OpenCV in BGR
        elif cls == 2 or cls == 3:
            color=(0,225,0)
        elif cls == 4 or cls == 5:
            color=(255,0,0)
        
        if type == "box":
            label = f"{class_names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif type == "point":
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            if cls == 0 or cls == 2 or cls == 4:
                #plot a square with a black boarder, and inside color
                rec_size=7
                cv2.rectangle(frame, (x_center - rec_size, y_center - rec_size), (x_center + rec_size, y_center + rec_size), color, 3)
                rec_size=11
                cv2.rectangle(frame, (x_center - rec_size, y_center - rec_size), (x_center + rec_size, y_center + rec_size), (0, 0, 0), 3)
            if cls == 1 or cls == 3 or cls == 5:
                #plot a circle with a black boarder, and inside color
                cv2.circle(frame, (x_center, y_center), 7, color, 3)
                cv2.circle(frame, (x_center, y_center), 11, (0, 0, 0), 3)
    return frame


def yolo_to_xyxy(box, img_w, img_h):
    """Convert YOLO normalized box to pixel coordinates (x1, y1, x2, y2)."""
    cls, x, y, w, h = box
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return cls, np.array([x1, y1, x2, y2])

def yolo_to_xyxy_conf_cls(boxes,img_w,img_h):
    results = []
    for box in boxes:
        
        print(box)
        cls, xyxy = yolo_to_xyxy(box, img_w, img_h)
        x1, y1, x2, y2 = xyxy
        conf = box.conf[0]            # Get the confidence score
        cls = box.cls[0]              # Get the class ID
        results.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), cls.item()])
    return results

import time

class FrameRateCounter:
    """
    A utility class for calculating and tracking frame rate (FPS).
    
    Usage:
        fps_counter = FrameRateCounter(update_interval=30)
        
        while True:
            # Process frame
            fps_counter.tick()
            
            # Get current FPS
            current_fps = fps_counter.get_fps()
            
            # Optionally annotate frame
            fps_counter.annotate_frame(frame)
    """
    
    def __init__(self, update_interval_frames: int = 30):
        """
        Initialize the frame rate counter.
        
        Args:
            update_interval (int): Number of frames between FPS updates. Default is 30.
        """
        self.update_interval_frames = update_interval_frames
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.total_start_time = time.time()
        self.total_frames = 0
    
    def tick(self):
        """Call this method once per frame to update the frame counter."""
        self.frame_count += 1
        self.total_frames += 1
        
        if self.frame_count >= self.update_interval_frames:
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            self.frame_count = 0
            self.start_time = time.time()
    
    def get_fps(self) -> float:
        """
        Get the current FPS.
        
        Returns:
            float: Current frames per second.
        """
        return self.fps
    
    def get_average_fps(self) -> float:
        """
        Get the average FPS since the counter was created.
        
        Returns:
            float: Average frames per second.
        """
        elapsed = time.time() - self.total_start_time
        return self.total_frames / elapsed if elapsed > 0 else 0
    
    def reset(self):
        """Reset the frame counter."""
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.total_start_time = time.time()
        self.total_frames = 0