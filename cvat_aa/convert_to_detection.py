#!/usr/bin/env python3
"""
Convert tracking function to detection function for CLI compatibility
"""

def create_detection_version():
    """Create a DetectionFunctionSpec version of the tracking function"""
    
    source_file = "/home/scblum/Projects/testbed_cv/cvat_aa/cvat_aa_yolo_tracking.py"
    output_file = "/home/scblum/Projects/testbed_cv/cvat_aa/cvat_aa_yolo_detection.py"
    
    print(f"Converting {source_file} to detection function...")
    
    try:
        with open(source_file, 'r') as f:
            content = f.read()
        
        # Replace TrackingFunctionSpec with DetectionFunctionSpec
        content = content.replace(
            'spec = cvataa.TrackingFunctionSpec(supported_shape_types=["rectangle"])',
            '''# Convert to DetectionFunctionSpec for CLI compatibility
spec = cvataa.DetectionFunctionSpec(
    labels=[
        cvataa.label_spec("red_square", 0),
        cvataa.label_spec("green_square", 1), 
        cvataa.label_spec("green_circle", 2),
        cvataa.label_spec("blue_square", 3),
        cvataa.label_spec("blue_circle", 4),
        cvataa.label_spec("red_circle", 5),
    ]
)'''
        )
        
        # Add detect function
        detect_function = '''
def detect(context, image):
    """Detection function with internal tracking for smoothing"""
    debug_print("detect function called")
    logger.info("Running detection with tracking")
    
    # Run detection and tracking
    result = model(image, conf=CONFIDENCE_THRESHOLD, iou=.7, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections_tracker = tracker.update_with_detections(detections)
    detections_smoothed = smoother.update_with_detections(detections_tracker)
    
    debug_print(f"Found {len(detections)} detections, {len(detections_tracker)} tracked detections")
    
    # Return detections (CLI doesn't support tracking output, but we can still track internally)
    for i, (box, cls_id) in enumerate(zip(detections_smoothed.xyxy, detections_smoothed.class_id)):
        rect = cvataa.rectangle(int(cls_id.item()), [float(p.item()) for p in box])
        
        # Log tracking info if available
        if detections_tracker.tracker_id is not None and i < len(detections_tracker.tracker_id):
            track_id = detections_tracker.tracker_id[i]
            debug_print(f"Detection {i}: class={cls_id.item()}, track_id={track_id}, bbox={box}")
        else:
            debug_print(f"Detection {i}: class={cls_id.item()}, bbox={box}")
            
        yield rect
'''
        
        # Add the detect function before the calculate_iou function
        content = content.replace(
            'def calculate_iou(box1, box2):',
            detect_function + '\ndef calculate_iou(box1, box2):'
        )
        
        # Write the new file
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"✓ Created detection version: {output_file}")
        print(f"\nYou can now use:")
        print(f"cvat-cli task auto-annotate 32 \\")
        print(f"  --function-file {output_file}")
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    create_detection_version()
