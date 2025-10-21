'''
source /home/scblum/Projects/testbed_cv/.venv/bin/activate
'''


import os
import json
import yaml
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from glob import glob
import random
from sklearn.model_selection import train_test_split
import torch
import gc

# --- Configuration ---
yolo_train_path = '/home/scblum/Projects/testbed_cv'
dataset_path = os.path.join(yolo_train_path, 'dataset')
images_dir = os.path.join(dataset_path, 'images/train')
labels_dir = os.path.join(dataset_path, 'labels/train')
train_split_ratio = 0.8  # 80% for training, 20% for validation
print(f"Dataset path: {dataset_path}")
print(f"Images directory: {images_dir}")

# --- 1. Find all image files that have a corresponding label ---
image_paths = []
for image_file in glob(os.path.join(images_dir, '*.*')):
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    label_file = os.path.join(labels_dir, f'{image_name}.txt')
    
    # Check if a non-empty label file exists
    if os.path.exists(label_file) and os.path.getsize(label_file) > 0:
        image_paths.append(image_file)

print(f"Found {len(image_paths)} images with annotations.")

# --- 2. Split the paths into training and validation sets ---
train_paths, val_paths = train_test_split(image_paths, test_size=1-train_split_ratio, random_state=42)

# --- 3. Save the lists to text files ---
with open(os.path.join(dataset_path, 'train.txt'), 'w') as f:
    for path in train_paths:
        f.write(path + '\n')

with open(os.path.join(dataset_path, 'val.txt'), 'w') as f:
    for path in val_paths:
        f.write(path + '\n')

timestamp = datetime.now().strftime("%y_%m_%d_%H_%M")
model_name = f'part_detection_{timestamp}'

training_args = {
    'data': os.path.join(yolo_train_path, 'train_yolo','dataset.yaml'),          # Path to dataset YAML
    'epochs': 10000,                     # INCREASED: Maximum epochs for best possible convergence
    'imgsz': 1600,                     # Reduce image size (was 800)
    'batch': 4,                         # Reduce batch size (was 16)
    'lr0': 0.01,                      # ORIGINAL: Back to original learning rate
    'lrf': 0.01,                      # ORIGINAL: Back to original final learning rate
    'momentum': 0.937,                 # SGD momentum
    'weight_decay': 0.0002,           # Weight decay for regularization
    'warmup_epochs': 3,               # Warmup epochs for learning rate
    'warmup_momentum': 0.8,           # Warmup momentum
    'box': 7.5,                       # Box loss gain
    'cls': 1.5,                       # Classification loss gain
    'dfl': 1.5,                       # Distribution focal loss gain
    'patience': 200,                 # Early stopping patience (increased)
    'project': os.path.join(yolo_train_path, 'saved_models'),      # Project directory
    'name': model_name,                # CHANGED: New run name for v5 with 400 epochs
    'save': True,                     # Save training checkpoints
    'save_period': 20,                # Save checkpoint every N epochs
    'cache': False,                   # Cache images for faster training (disable if low memory)
    'device': 0,                      # UPDATED: Use detected device (GPU or CPU)
    'workers': 8,                     # Number of data loading workers
    'exist_ok': True,                 # Overwrite existing project/name
    'pretrained': True,               # Use pretrained weights
    'optimizer': 'AdamW',               # Optimizer (SGD, Adam, AdamW)
    'verbose': True,                  # Verbose output
    'seed': 0,                        # Random seed for reproducibility
    'deterministic': False,           # Deterministic training for reproducibility
    'single_cls': False,              # Treat as single-class dataset
    'rect': False,                    # Rectangular training
    'cos_lr': False,                   # Enable cosine learning rate scheduler
    'close_mosaic': False,            # Disable mosaic augmentation for last 10 epochs
    'resume': False,                  # Resume from last checkpoint
    'amp': True,                      # Automatic Mixed Precision training
    'fraction': 1.0,                  # Dataset fraction to use
    'profile': False,                 # Profile ONNX and TensorRT speeds
    'freeze': 2,                      # Freeze layers (list of layer indices)
    'multi_scale': False,              # DISABLED: Keep original scale for stability
    'overlap_mask': True,             # Overlap masks during training
    'dropout': .1,                   # Use dropout regularization
    'val': True,                      # Validate/test during training
    
    # ORIGINAL LOW AUGMENTATION PARAMETERS - minimal augmentation for stability
    # https://docs.ultralytics.com/guides/yolo-data-augmentation/#copy-paste-copy_paste
    'bgr': 0,                         # Flips color channels, BGR augmentation (0-1)
    'hsv_h': .012,                    # HSV-Hue augmentation (±0.015) - original
    'hsv_s': .5,                     # HSV-Saturation augmentation - back to original
    'hsv_v': .5,                     # HSV- Brightness (Value) augmentation - back to original
    'degrees': 10,                  # ORIGINAL:
    'translate': 0.1,               # ORIGINAL: Minimal translation
    'scale': .3,                     # ORIGINAL: Back to original scale variation
    'shear': 10,                     # ORIGINAL: No shear for stability
    'perspective': .001,               # ORIGINAL: No perspective transform
    'flipud': .05,                    # ORIGINAL: No vertical flip
    'fliplr': 0.5,                    # ORIGINAL: Keep horizontal flip
    'mosaic': .6,                    # Mosaic augmentation probability
    'mixup': 0,                    # ORIGINAL: No MixUp
    'cutmix':.4,                   # ORIGINAL: No CutMix
    'copy_paste': 0,                # ORIGINAL: No copy-paste
    'erasing': .3,                   # ORIGINAL: Random erasing back to original value
}

# Clear CUDA memory
torch.cuda.empty_cache()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

model = YOLO('yolo11s.pt')  # Downloads pretrained weights automatically

results = model.train(**training_args)

# Save training args
with open('training_args.json', 'w') as f:
    json.dump(training_args, f)
