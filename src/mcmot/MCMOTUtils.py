import cv2
import os
import time
import pathlib
import numpy as np

def find_latest_model():
    training_dir = pathlib.Path("/home/scblum/Projects/testbed_cv/saved_models")

    # Look for best.pt files in subdirectories
    model_files = list(training_dir.glob("*/weights/best.pt"))
    
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
    
    def __init__(self, update_interval: int = 30):
        """
        Initialize the frame rate counter.
        
        Args:
            update_interval (int): Number of frames between FPS updates. Default is 30.
        """
        self.update_interval = update_interval
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.total_start_time = time.time()
        self.total_frames = 0
    
    def tick(self):
        """Call this method once per frame to update the frame counter."""
        self.frame_count += 1
        self.total_frames += 1
        
        if self.frame_count >= self.update_interval:
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
    
    def annotate_frame(self, frame, position: tuple = (10, 30), 
                      font_scale: float = 1.0, color: tuple = (0, 255, 0), 
                      thickness: int = 2):
        """
        Draw the current FPS on a frame.
        
        Args:
            frame: The image frame to annotate (numpy array).
            position (tuple): (x, y) position for the text. Default is (10, 30).
            font_scale (float): Font scale for the text. Default is 1.0.
            color (tuple): BGR color tuple. Default is green (0, 255, 0).
            thickness (int): Text thickness. Default is 2.
        
        Returns:
            The annotated frame (modifies in place).
        """
        import cv2
        text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)
        return frame
    
    def reset(self):
        """Reset the frame counter."""
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.total_start_time = time.time()
        self.total_frames = 0