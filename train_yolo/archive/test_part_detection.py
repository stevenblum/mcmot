#!/usr/bin/env python3
"""
Custom YOLOv11 Part Detection Test Script

This script tests our trained part detection model on the images we used for training.
It loads the custom model, runs inference on training images, and displays results
with bounding boxes and confidence scores.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

YOLO_TRAIN_PATH = '/home/scblum/Projects/testbed_cv/train_yolo'
MODEL_PATH = os.path.join(YOLO_TRAIN_PATH, 'part_detection_2025_10_13_11_20/weights/best.pt')  # Updated path
DATASET_PATH = os.path.join(YOLO_TRAIN_PATH, 'datasets')
IMAGES_DIR = os.path.join(DATASET_PATH, 'images/train')
LABELS_DIR = os.path.join(DATASET_PATH, 'labels/train')
OUTPUT_DIR = os.path.join(YOLO_TRAIN_PATH, 'detection_test_results')
print(f"Dataset path: {DATASET_PATH}")
print(f"Images directory: {IMAGES_DIR}")


def load_custom_model(model_path):
    """
    Load the custom trained YOLOv11 model
    
    Args:
        model_path (str): Path to the trained model (.pt file)
    
    Returns:
        YOLO: Loaded model instance
    """
    try:
        model = YOLO(model_path)
        print(f"✓ Successfully loaded custom model: {model_path}")
        
        # Print model info
        print(f"Model classes: {model.names}")
        print(f"Number of classes: {len(model.names)}")
        
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_model_on_images(model, images_dir, output_dir, max_images=10):
    """
    Test the model on training images and save results
    
    Args:
        model (YOLO): Loaded YOLO model
        images_dir (str): Directory containing test images
        output_dir (str): Directory to save detection results
        max_images (int): Maximum number of images to process
    
    Returns:
        list: List of processed image results
    """
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(IMAGES_DIR).glob(f"*{ext}"))

    print(f"Found {len(image_files)} images in {IMAGES_DIR}")
    
    # Limit number of images to process
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
        print(f"Processing first {max_images} images...")
    
    results_summary = []
    
    for i, img_file in enumerate(image_files):
        print(f"\n--- Processing {img_file.name} ({i+1}/{len(image_files)}) ---")
        
        try:
            # Run inference
            results = model(str(img_file), conf=0.25, iou=0.45)
            
            # Get detection results
            result = results[0]
            detections = result.boxes
            
            if detections is not None and len(detections) > 0:
                num_detections = len(detections)
                print(f"✓ Found {num_detections} detections")
                
                # Print detection details
                for j, box in enumerate(detections):
                    # Extract box info
                    coords = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = model.names[cls_id]
                    
                    print(f"  Detection {j+1}: {cls_name} (conf: {conf:.3f})")
                    print(f"    Box: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}]")
                
                # Save annotated image
                annotated_img = result.plot()  # Get image with annotations
                output_file = OUTPUT_DIR / f"detected_{img_file.name}"
                cv2.imwrite(str(output_file), annotated_img)
                print(f"✓ Saved annotated image: {output_file}")
                
                # Add to summary
                results_summary.append({
                    'image': img_file.name,
                    'detections': num_detections,
                    'classes': [model.names[int(box.cls[0])] for box in detections],
                    'confidences': [float(box.conf[0]) for box in detections]
                })
                
            else:
                print("⚠ No detections found")
                results_summary.append({
                    'image': img_file.name,
                    'detections': 0,
                    'classes': [],
                    'confidences': []
                })
        
        except Exception as e:
            print(f"❌ Error processing {img_file.name}: {e}")
            continue
    
    return results_summary

def create_detection_summary(results_summary, output_dir):
    """
    Create a summary report of all detections
    
    Args:
        results_summary (list): List of detection results
        output_dir (str): Directory to save summary
    """
    print(f"\n{'='*60}")
    print("DETECTION SUMMARY")
    print(f"{'='*60}")
    
    total_images = len(results_summary)
    images_with_detections = sum(1 for r in results_summary if r['detections'] > 0)
    total_detections = sum(r['detections'] for r in results_summary)
    
    print(f"Total images processed: {total_images}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections found: {total_detections}")
    print(f"Average detections per image: {total_detections/total_images:.1f}")
    
    # Class statistics
    all_classes = []
    all_confidences = []
    for result in results_summary:
        all_classes.extend(result['classes'])
        all_confidences.extend(result['confidences'])
    
    if all_classes:
        print(f"\nCLASS DISTRIBUTION:")
        from collections import Counter
        class_counts = Counter(all_classes)
        for class_name, count in class_counts.most_common():
            avg_conf = np.mean([conf for i, conf in enumerate(all_confidences) 
                               if all_classes[i] == class_name])
            print(f"  {class_name}: {count} detections (avg conf: {avg_conf:.3f})")
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Average confidence: {np.mean(all_confidences):.3f}")
        print(f"  Min confidence: {np.min(all_confidences):.3f}")
        print(f"  Max confidence: {np.max(all_confidences):.3f}")
    
    # Save detailed summary
    summary_file = Path(output_dir) / "detection_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("CUSTOM YOLO PART DETECTION TEST RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total images processed: {total_images}\n")
        f.write(f"Images with detections: {images_with_detections}\n")
        f.write(f"Total detections found: {total_detections}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        for result in results_summary:
            f.write(f"\n{result['image']}:\n")
            f.write(f"  Detections: {result['detections']}\n")
            if result['detections'] > 0:
                for cls, conf in zip(result['classes'], result['confidences']):
                    f.write(f"    {cls}: {conf:.3f}\n")
    
    print(f"\n✓ Detailed summary saved to: {summary_file}")

def display_sample_results(output_dir, max_display=4):
    """
    Display some sample detection results using matplotlib
    
    Args:
        output_dir (str): Directory containing detection results
        max_display (int): Maximum number of images to display
    """
    # Find annotated images
    result_images = list(Path(output_dir).glob("detected_*.jpg")) + \
                   list(Path(output_dir).glob("detected_*.JPG")) + \
                   list(Path(output_dir).glob("detected_*.png"))
    
    if not result_images:
        print("No detection result images found for display")
        return
    
    # Limit display count
    display_images = result_images[:max_display]
    
    print(f"\nDisplaying {len(display_images)} sample detection results...")
    
    # Create subplot layout
    cols = 2
    rows = (len(display_images) + 1) // 2
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, img_path in enumerate(display_images):
        # Load image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Detections: {img_path.name}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(Path(output_dir) / "sample_detections.png", dpi=150, bbox_inches='tight')
    print(f"✓ Sample results saved to: {Path(output_dir) / 'sample_detections.png'}")
    
    # Show the plot
    plt.show()

def main():
    """
    Main function to test the custom part detection model
    """
    print("="*60)
    print("CUSTOM YOLO PART DETECTION TEST SCRIPT")
    print("="*60)
    
    # Configuration - Updated for v3 model
    model_path = "yolo_training/part_detection_run_v3/weights/best.pt"  # Updated path
    images_dir = "./images"
    output_dir = "./detection_test_results_v3"  # New output directory
    max_test_images = 10  # Limit for faster testing
    
    print(f"Model path: {model_path}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max test images: {max_test_images}")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please make sure the training completed successfully.")
        return
    
    # Check if images directory exists
    if not Path(images_dir).exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    # Load model
    print(f"\n📦 Loading custom model...")
    model = load_custom_model(model_path)
    if model is None:
        return
    
    # Test model on images
    print(f"\n🔍 Testing model on training images...")
    results_summary = test_model_on_images(model, images_dir, output_dir, max_test_images)
    
    # Create summary report
    create_detection_summary(results_summary, output_dir)
    
    # Display sample results
    print(f"\n📊 Creating visual summary...")
    try:
        display_sample_results(output_dir, max_display=4)
    except Exception as e:
        print(f"⚠ Could not display results: {e}")
        print("You can manually check the images in the output directory.")
    
    print(f"\n🎉 Testing completed! Check {output_dir} for all results.")
    print(f"\nFiles created:")
    print(f"  - detected_*.jpg: Individual detection results")
    print(f"  - detection_summary.txt: Detailed statistics")
    print(f"  - sample_detections.png: Visual summary")

if __name__ == "__main__":
    main()