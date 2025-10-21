#!/usr/bin/env python3
"""
Real-time Part Detection with Webcam
====================================

A simple script that connects to your webcam and runs the trained YOLOv11 
part detection model on live camera frames, displaying the results in real-time.

Usage:
    python webcam_part_detection.py

Controls:
    - Press 'q' to quit
    - Press 's' to save current frame with detections
    - Press 'c' to toggle confidence display
    - Press 'f' to toggle FPS display

Author: AI Assistant
Date: October 2025
"""

import cv2
import time
import os
import glob
import yaml
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

MODEL_PATH = "/home/scblum/Projects/testbed_cv/yolo_training/part_detection_25_10_13_15_24/weights/best.pt"

def find_latest_model():
    """
    Automatically find the most recent YOLO model in the training directory.
    
    Returns:
        tuple: (model_path, model_name) or (None, None) if no models found
    """
    training_dir = "yolo_training"
    
    if not os.path.exists(training_dir):
        print(f"❌ Training directory not found: {training_dir}")
        return None, None
    
    # Look for all best.pt files in training subdirectories
    pattern = os.path.join(training_dir, "*", "weights", "best.pt")
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"❌ No trained models found in {training_dir}")
        return None, None
    
    # Get the most recent model based on modification time
    latest_model = max(model_files, key=os.path.getmtime)
    
    # Extract model name from path (e.g., "part_detection_25_10_07_12_23")
    model_name = os.path.basename(os.path.dirname(os.path.dirname(latest_model)))
    
    # Get creation time for display
    mod_time = os.path.getmtime(latest_model)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
    
    print(f"🎯 Latest model found: {model_name}")
    print(f"📅 Modified: {time_str}")
    print(f"📁 Path: {latest_model}")
    
    return latest_model, model_name

def load_detection_config(config_path="detection_config.yml"):
    """
    Load detection configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"✓ Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠ Configuration file not found: {config_path}")
        print("Using default configuration...")
        # Return default configuration
        return {
            'cameras': {'indices': [0, 4], 'resolution': {'width': 640, 'height': 480}, 'fps': 30},
            'detection': {'confidence_threshold': 0.25, 'low_confidence_threshold': 0.05},
            'tracking': {'enable': True, 'tracker_type': 'bytetrack.yaml', 'persist_frames': 60},
            'display': {'show_confidence': True, 'show_fps': True, 'fps_smoothing': 0.9},
            'class_colors': {
                'red_square': [0, 0, 255], 'red_circle': [0, 100, 255],
                'green_square': [0, 255, 0], 'green_circle': [0, 255, 100],
                'blue_square': [255, 0, 0], 'blue_circle': [255, 100, 0]
            }
        }
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML file: {e}")
        raise

class WebcamPartDetector:
    """Real-time part detection using webcam feed."""
    
    def __init__(self, model_path, model_name, config=None):
        """
        Initialize the dual webcam part detector.
        
        Args:
            model_path (str): Path to the trained YOLO model
            model_name (str): Name of the model for display
            config (dict): Configuration dictionary from YAML file
        """
        self.model_path = model_path
        self.model_name = model_name
        
        # Load configuration or use defaults
        if config is None:
            config = load_detection_config()
        self.config = config
        
        # Extract configuration values
        self.conf_threshold = config['detection']['confidence_threshold']
        self.low_conf_threshold = config['detection']['low_confidence_threshold']
        self.camera_indices = config['cameras']['indices']
        
        # Tracking settings from config
        self.enable_tracking = config['tracking']['enable']
        self.tracker_type = config['tracking']['tracker_type']
        self.track_persist = config['tracking']['persist_frames']
        self.tracker_config = self.tracker_type  # Configuration for YOLO tracker
        
        # Display settings from config
        self.show_confidence = config['display']['show_confidence']
        self.show_fps = config['display']['show_fps']
        self.fps_smoothing = config['display']['fps_smoothing']
        self.avg_fps = 0
        
        # Colors for each class from config (convert to tuples for OpenCV)
        self.class_colors = {
            name: tuple(color) for name, color in config['class_colors'].items()
        }
        
        # Output directory from config
        output_dir_name = config.get('output', {}).get('directory', 'webcam_detections')
        self.output_dir = Path(output_dir_name)
        self.output_dir.mkdir(exist_ok=True)
        self.frame_counter = 0
        
        # Initialize model and cameras
        self.model1 = None  # Model for camera 1
        self.model2 = None  # Model for camera 2
        self.cap1 = None
        self.cap2 = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the YOLO model and cameras."""
        print("🚀 Initializing Dual Webcam Part Detector")
        print("=" * 50)
        
        # Load YOLO models (separate instances for independent tracking)
        print(f"📦 Loading models: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            # Create two separate model instances for independent tracking
            self.model1 = YOLO(self.model_path)  # For camera 1
            self.model2 = YOLO(self.model_path)  # For camera 2
            print(f"✓ Two model instances loaded successfully")
            
            # Get class names from first model
            if hasattr(self.model1, 'names'):
                self.class_names = self.model1.names
                print(f"✓ Classes: {list(self.class_names.values())}")
            else:
                # Fallback class names
                self.class_names = {
                    0: 'red_square', 1: 'red_circle', 2: 'green_square',
                    3: 'green_circle', 4: 'blue_square', 5: 'blue_circle'
                }
                print(f"⚠ Using fallback class names: {list(self.class_names.values())}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")
        
        # Initialize cameras
        cam1_idx, cam2_idx = self.camera_indices[0], self.camera_indices[1]
        
        print(f"📹 Connecting to camera {cam1_idx}")
        self.cap1 = cv2.VideoCapture(cam1_idx)
        if not self.cap1.isOpened():
            raise RuntimeError(f"Failed to open camera {cam1_idx}")
        
        print(f"📹 Connecting to camera {cam2_idx}")
        self.cap2 = cv2.VideoCapture(cam2_idx)
        if not self.cap2.isOpened():
            self.cap1.release()
            raise RuntimeError(f"Failed to open camera {cam2_idx}")
        
        # Set camera properties for both cameras using config values
        cam_config = self.config['cameras']
        for cap, idx in [(self.cap1, cam1_idx), (self.cap2, cam2_idx)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config['resolution']['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config['resolution']['height'])
            cap.set(cv2.CAP_PROP_FPS, cam_config['fps'])
            # Optimize camera buffer to reduce timeout issues
            cap.set(cv2.CAP_PROP_BUFFERSIZE, cam_config.get('buffer_size', 1))
            
            # Set video format if specified
            if 'format' in cam_config and cam_config['format'] == 'MJPG':
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            elif 'format' in cam_config and cam_config['format'] == 'YUYV':
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
                
            # Enable automatic camera adjustments
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto exposure (3 = full auto)
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)        # Auto white balance
            if idx == 6:  # Camera 6 - use automatic settings
                print(f"  Using automatic adjustments for camera {idx}")
            
            # Get actual camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"✓ Camera {idx} connected: {width}x{height} @ {fps:.1f}fps")
        
        print(f"✓ Confidence threshold: {self.conf_threshold}")
        print(f"✓ Low confidence threshold: {self.low_conf_threshold}")
        print(f"✓ Output directory: {self.output_dir}")
        print()
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'c' to toggle confidence display")
        print("  - Press 'f' to toggle FPS display")
        print("=" * 50)
    
    def _draw_detections(self, frame, results):
        """
        Draw detection boxes and labels on the frame.
        Shows both high-confidence (solid) and low-confidence (dashed) detections.
        
        Args:
            frame (np.ndarray): Input frame
            results: YOLO detection results
            
        Returns:
            tuple: (annotated_frame, detection_count, low_conf_count)
        """
        annotated_frame = frame.copy()
        detection_count = 0
        low_conf_count = 0
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get track ID if available (from YOLO tracker)
                    track_id = None
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0].cpu().numpy())
                    
                    # Get class name and color
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    base_color = self.class_colors.get(class_name, (255, 255, 255))
                    
                    if confidence >= self.conf_threshold:
                        # High confidence detection - solid box
                        detection_count += 1
                        color = base_color
                        thickness = 2
                        
                        # Draw solid bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Prepare label with track ID
                        if track_id is not None:
                            if self.show_confidence:
                                label = f"{class_name} #{track_id}: {confidence:.2f}"
                            else:
                                label = f"{class_name} #{track_id}"
                        else:
                            if self.show_confidence:
                                label = f"{class_name}: {confidence:.2f}"
                            else:
                                label = class_name
                        
                        # Get text size for background
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        text_thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, text_thickness
                        )
                        
                        # Draw label background
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1),
                            color,
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - baseline - 2),
                            font,
                            font_scale,
                            (255, 255, 255),
                            text_thickness
                        )
                        
                    elif confidence >= self.low_conf_threshold:
                        # Low confidence detection - dashed/faded box
                        low_conf_count += 1
                        
                        # Make color more transparent/faded
                        faded_color = tuple(int(c * 0.6) for c in base_color)
                        
                        # Draw dashed rectangle
                        self._draw_dashed_rectangle(annotated_frame, (x1, y1), (x2, y2), faded_color, 1)
                        
                        # Small faded label with track ID
                        if self.show_confidence:
                            if track_id is not None:
                                label = f"{class_name} #{track_id}: {confidence:.2f}"
                            else:
                                label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(
                                annotated_frame,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                faded_color,
                                1
                            )
        
        return annotated_frame, detection_count, low_conf_count
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness):
        """Draw a dashed rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw dashed lines for each side
        dash_length = 10
        gap_length = 5
        
        # Top edge
        x = x1
        while x < x2:
            end_x = min(x + dash_length, x2)
            cv2.line(img, (x, y1), (end_x, y1), color, thickness)
            x += dash_length + gap_length
        
        # Bottom edge  
        x = x1
        while x < x2:
            end_x = min(x + dash_length, x2)
            cv2.line(img, (x, y2), (end_x, y2), color, thickness)
            x += dash_length + gap_length
        
        # Left edge
        y = y1
        while y < y2:
            end_y = min(y + dash_length, y2)
            cv2.line(img, (x1, y), (x1, end_y), color, thickness)
            y += dash_length + gap_length
        
        # Right edge
        y = y1
        while y < y2:
            end_y = min(y + dash_length, y2)
            cv2.line(img, (x2, y), (x2, end_y), color, thickness)
            y += dash_length + gap_length
    
    def _draw_info_overlay(self, frame, fps, detection_count, low_conf_count=0):
        """
        Draw information overlay on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            fps (float): Current FPS
            detection_count (int): Number of high-confidence detections
            low_conf_count (int): Number of low-confidence detections
        """
        # Model name at the top
        model_text = f"Model: {self.model_name}"
        cv2.putText(frame, model_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 0), 2)
        
        if self.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        
        # Detection count
        count_text = f"Detections: {detection_count}"
        cv2.putText(frame, count_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 255), 2)
        
        # Low confidence count
        if low_conf_count > 0:
            low_count_text = f"Low conf: {low_conf_count}"
            cv2.putText(frame, low_count_text, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (128, 128, 128), 2)
        
        # Threshold info
        threshold_text = f"Conf: >{self.conf_threshold:.2f} (faded: >{self.low_conf_threshold:.2f})"
        cv2.putText(frame, threshold_text, (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def save_frame(self, frame, detection_count):
        """Save the current frame with detections."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}_{detection_count}parts.jpg"
        filepath = self.output_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"💾 Saved frame: {filename}")
    
    def run(self):
        """Main detection loop for dual cameras."""
        print("🎥 Starting dual webcam detection...")
        print("Camera feeds should open in a new window")
        
        frame_time = time.time()
        
        try:
            while True:
                # Read frames from both cameras
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()
                
                if not ret1 or not ret2:
                    print("⚠ Failed to read frame from one or both cameras")
                    break
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - frame_time)
                frame_time = current_time
                
                # Smooth FPS calculation
                self.avg_fps = (self.fps_smoothing * self.avg_fps + 
                               (1 - self.fps_smoothing) * fps)
                
                # Run YOLO tracking on both frames with separate model instances
                try:
                    results1 = self.model1.track(frame1, conf=self.low_conf_threshold, verbose=False, tracker=self.tracker_config, persist=True)
                    results2 = self.model2.track(frame2, conf=self.low_conf_threshold, verbose=False, tracker=self.tracker_config, persist=True)
                    
                    # Draw detections on both frames
                    annotated_frame1, detection_count1, low_conf_count1 = self._draw_detections(frame1, results1)
                    annotated_frame2, detection_count2, low_conf_count2 = self._draw_detections(frame2, results2)
                    
                    # Add camera labels
                    cv2.putText(annotated_frame1, f"Camera {self.camera_indices[0]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame2, f"Camera {self.camera_indices[1]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # Draw info overlay on first frame only (to avoid duplication)
                    total_detections = detection_count1 + detection_count2
                    total_low_conf = low_conf_count1 + low_conf_count2
                    self._draw_info_overlay(annotated_frame1, self.avg_fps, total_detections, total_low_conf)
                    
                    # Add detection counts for each camera
                    cv2.putText(annotated_frame1, f"Cam {self.camera_indices[0]}: {detection_count1} dets", 
                               (10, annotated_frame1.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(annotated_frame2, f"Cam {self.camera_indices[1]}: {detection_count2} dets", 
                               (10, annotated_frame2.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                except Exception as e:
                    print(f"⚠ Detection error: {e}")
                    annotated_frame1 = frame1
                    annotated_frame2 = frame2
                    cv2.putText(annotated_frame1, f"Camera {self.camera_indices[0]} - Error", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(annotated_frame2, f"Camera {self.camera_indices[1]} - Error", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self._draw_info_overlay(annotated_frame1, self.avg_fps, 0, 0)
                
                # Combine frames side by side
                # Ensure both frames have the same height
                h1, w1 = annotated_frame1.shape[:2]
                h2, w2 = annotated_frame2.shape[:2]
                max_height = max(h1, h2)
                
                if h1 != max_height:
                    annotated_frame1 = cv2.resize(annotated_frame1, (int(w1 * max_height / h1), max_height))
                if h2 != max_height:
                    annotated_frame2 = cv2.resize(annotated_frame2, (int(w2 * max_height / h2), max_height))
                
                # Combine horizontally
                combined_frame = np.hstack((annotated_frame1, annotated_frame2))
                
                # Display combined frame with model name in window title
                window_title = f'Dual Camera Part Detection - {self.model_name}'
                cv2.imshow(window_title, combined_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('s'):
                    self.save_frame(combined_frame, total_detections)
                elif key == ord('c'):
                    self.show_confidence = not self.show_confidence
                    print(f"📊 Confidence display: {'ON' if self.show_confidence else 'OFF'}")
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                    print(f"⏱️ FPS display: {'ON' if self.show_fps else 'OFF'}")
                
                self.frame_counter += 1
                
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Error during detection: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")
        
        if self.cap1:
            self.cap1.release()
            print("✓ Camera 1 released")
            
        if self.cap2:
            self.cap2.release()
            print("✓ Camera 2 released")
        
        cv2.destroyAllWindows()
        
        print(f"✓ Processed {self.frame_counter} frames")
        print("✓ Cleanup complete")


def main():
    """Main function."""
    print("Real-time Dual Camera Part Detection")
    print("===================================\n")
    
    # Load configuration
    print("⚙️ Loading configuration...")
    config = load_detection_config()
    
    # Auto-detect the latest model or use specified path
    if config['model']['auto_detect']:
        print("🔍 Auto-detecting latest trained model...")
        model_path, model_name = find_latest_model()
        
        if model_path is None:
            print("❌ No trained models found!")
            print("Please train a model first using train_yolo_parts.py")
            return
    else:
        model_path = config['model']['path']
        model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        if not os.path.exists(model_path):
            print(f"❌ Specified model not found: {model_path}")
            return
    
    print(f"✓ Using model: {model_name}")
    print(f"✓ Model path: {model_path}")
    print(f"✓ Cameras: {config['cameras']['indices']}")
    print(f"✓ Confidence threshold: {config['detection']['confidence_threshold']}")
    print(f"✓ Tracking: {config['tracking']['tracker_type']}")
    print()
    
    try:
        # Create and run detector with configuration
        detector = WebcamPartDetector(
            model_path=model_path,
            model_name=model_name,
            config=config
        )
        
        detector.run()
        
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your cameras are not being used by another application")
        print("2. Check camera indices in detection_config.yml")
        print("3. Verify that the model file exists and is valid")
        print("4. Update detection_config.yml for your specific setup")


if __name__ == "__main__":
    main()