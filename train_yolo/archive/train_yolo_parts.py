#!/usr/bin/env python3
"""
YOLOv11 Custom Training Script for Part Detection

This script prepares data and trains YOLOv11 on custom images with cuboid annotations.
Converts cuboids to bounding boxes and reorganizes classes by shape and color.

WORKFLOW:
1. Load annotations from JSON/XML/TXT files in ./annotations/
2. Filter images that have corresponding annotations
3. Convert 3D cuboid annotations to 2D bounding boxes
4. Map part+attributes to 6 specific classes (red/green/blue + square/circle)
5. Convert to YOLO format (normalized coordinates)
6. Train YOLOv11 model on the processed dataset
7. Save trained model for integration with YOLO-3D

INPUT REQUIREMENTS:
- Images in ./images/ folder (jpg, png, etc.)
- Annotations in ./annotations/ folder (JSON format expected)
- Annotations should contain "part" objects with shape and color attributes
- Cuboid annotations with vertex points that will be converted to 2D bboxes

OUTPUT:
- Trained YOLOv11 model ready for YOLO-3D integration
- YOLO format dataset in ./yolo_training/


Script FunctionOrganization
- main: Orchestrates the entire training process
    - load_annotations: Load and parse various annotation formats
        - parse_xml_annotations: Handle XML-specific parsing (CVAT, Pascal VOC)
    - validate_dataset: Ensure images and annotations are consistent
    - convert_to_yolo_format: Convert annotations to YOLO format
        - find_matching_annotation: Flexible filename matching for annotations
        - cuboid_to_bbox: Convert 3D cuboid points to 2D bounding box
        - get_class_id: Map shape/color to class ID
        - validate_yolo_coordinates: Ensure YOLO coordinates are valid
    - create_yaml_config: Generate YOLOv11 config files
    - train_yolo_model: Train YOLOv11 model on the dataset

    
Assumptions that may be used to simplify script:
--Annotations will always be in CVAT Format
    --CVAT cuboid: [xtl1, ytl1, xbl1, ybl1, xtr1, ytr1, xbr1, ybr1, xtl2, ytl2, xbl2, ybl2, xtr2, ytr2, xbr2, ybr2]
    --CVAT bbox: [xtl, ytl, xbr, ybr]
-Annotations will be converted into YOLO axis aligned bounding boxes
    --YOLO bbox: [x_center, y_center, width, height] (normalized)
"""

import os
import json
import yaml
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

def load_annotations(annotations_dir):
    """
    Load all annotation files from the annotations directory and parse them into a unified format
    
    LOGIC:
    - Scan for JSON, TXT, XML, and specific XML files in annotations directory
    - Parse each file and extract annotation data
    - Handle different annotation formats (COCO, LabelMe, CVAT XML, custom)
    - Return dictionary mapping image names to their annotation data
    
    Args:
        annotations_dir (str): Path to annotations directory
    
    Returns:
        dict: Dictionary mapping image names to their annotations
              Format: {image_name: annotation_data}
    """
    annotations = {}
    annotation_files = []
    
    # STEP 1: Find all annotation files with common extensions
    # Look for JSON (most common), TXT (YOLO format), and XML (Pascal VOC, CVAT)
    for ext in ['*.json', '*.txt', '*.xml']:
        annotation_files.extend(Path(annotations_dir).glob(ext))
    
    # STEP 1b: Also check for specific annotation files in parent directory (CVAT export pattern)
    parent_dir = Path(annotations_dir).parent
    for filename in ['annotations.xml', 'cvat_export.xml']:
        xml_file = parent_dir / filename
        if xml_file.exists():
            annotation_files.append(xml_file)
            print(f"Found annotation file in parent directory: {xml_file}")
    
    print(f"Found {len(annotation_files)} annotation files")
    
    # STEP 2: Process each annotation file
    for ann_file in annotation_files:
        try:
            if ann_file.suffix == '.xml':
                # XML format - handle CVAT and Pascal VOC formats
                annotations_from_xml = parse_xml_annotations(ann_file)
                annotations.update(annotations_from_xml)
                
            elif ann_file.suffix == '.json':
                # JSON is our primary format - parse and handle different structures
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                    
                    # CASE 1: List of annotations (multiple images in one file)
                    if isinstance(data, list):
                        for item in data:
                            # Extract image filename from various possible keys
                            if 'filename' in item or 'image' in item:
                                img_name = item.get('filename', item.get('image', ''))
                                if img_name:
                                    annotations[img_name] = item
                    
                    # CASE 2: Dictionary structure (single image or structured format)
                    elif isinstance(data, dict):
                        # COCO-like format with separate images and annotations arrays
                        if 'annotations' in data and 'images' in data:
                            # Build mapping from image IDs to filenames
                            images = {img['id']: img['file_name'] for img in data.get('images', [])}
                            
                            # Group annotations by image
                            for ann in data['annotations']:
                                img_id = ann['image_id']
                                img_name = images.get(img_id, f"image_{img_id}")
                                
                                # Initialize annotation list for this image if needed
                                if img_name not in annotations:
                                    annotations[img_name] = {'annotations': []}
                                annotations[img_name]['annotations'].append(ann)
                        else:
                            # Single image annotation - use flexible matching
                            # Store with multiple possible keys for better matching later
                            base_name = ann_file.stem
                            annotations[base_name] = data
                            annotations[base_name + '.jpg'] = data
                            annotations[base_name + '.png'] = data
            
            # TODO: Add TXT parsing if needed for other annotation formats
            
        except Exception as e:
            print(f"Error loading {ann_file}: {e}")
            continue
    
    print(f"Loaded annotations for {len(annotations)} images")
    return annotations


def validate_dataset(annotations, images_dir):
    """
    Validate dataset before training to catch common issues early
    
    VALIDATION CHECKS:
    - Ensure images directory exists and contains images (including subdirectories)
    - Ensure annotations directory exists and contains annotations
    - Check for matching image/annotation pairs
    - Validate annotation format and required fields
    - Check for minimum dataset size requirements
    
    Args:
        annotations (dict): Loaded annotations
        images_dir (str): Path to images directory
    
    Returns:
        bool: True if dataset is valid for training, False otherwise
    """
    print("\n=== DATASET VALIDATION ===")
    
    issues = []
    warnings = []
    
    # CHECK 1: Images directory
    images_path = Path(images_dir)
    if not images_path.exists():
        issues.append(f"Images directory does not exist: {images_dir}")
    else:
        # UPDATED: Look for images recursively in subdirectories (same as convert_to_yolo_format)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']
        image_files = []
        for ext in image_extensions:
            # Use rglob to search recursively through all subdirectories
            image_files.extend(list(images_path.rglob(f"*{ext}")))
        
        if len(image_files) == 0:
            issues.append(f"No image files found in {images_dir} or its subdirectories")
        else:
            print(f"✓ Found {len(image_files)} image files in {images_dir} and subdirectories")
            
            # Log subdirectories found (same as convert_to_yolo_format)
            subdirs = set()
            for img_file in image_files:
                subdir = img_file.parent.relative_to(images_path)
                if str(subdir) != '.':  # Not in root images directory
                    subdirs.add(str(subdir))
            
            if subdirs:
                print(f"  Images found in subdirectories: {', '.join(sorted(subdirs))}")
    
    # CHECK 2: Annotations
    if len(annotations) == 0:
        issues.append("No annotations loaded")
    else:
        print(f"✓ Loaded annotations for {len(annotations)} images")
    
    # CHECK 3: Annotation format validation
    valid_annotations = 0
    for img_name, ann_data in annotations.items():
        try:
            # Check if annotation has required structure
            objects_found = False
            
            if isinstance(ann_data, dict):
                if any(key in ann_data for key in ['annotations', 'objects', 'shapes']):
                    objects_found = True
                elif all(key in ann_data for key in ['shape', 'color', 'points']):
                    objects_found = True
            elif isinstance(ann_data, list):
                objects_found = True
            
            if objects_found:
                valid_annotations += 1
            else:
                warnings.append(f"Annotation format may be invalid for {img_name}")
                
        except Exception as e:
            warnings.append(f"Error validating annotation for {img_name}: {e}")
    
    print(f"✓ {valid_annotations}/{len(annotations)} annotations appear valid")
    
    # CHECK 4: Minimum dataset size
    if valid_annotations < 5:
        warnings.append(f"Very small dataset ({valid_annotations} images) - consider adding more data")
    elif valid_annotations < 20:
        warnings.append(f"Small dataset ({valid_annotations} images) - training may not be optimal")
    
    # CHECK 5: Class distribution check (if possible)
    try:
        class_counts = {}
        for img_name, ann_data in annotations.items():
            # Try to count classes (rough estimate)
            objects_to_check = []
            if isinstance(ann_data, dict):
                if 'annotations' in ann_data:
                    objects_to_check = ann_data['annotations']
                elif 'objects' in ann_data:
                    objects_to_check = ann_data['objects']
                elif 'shapes' in ann_data:
                    objects_to_check = ann_data['shapes']
                else:
                    objects_to_check = [ann_data]
            elif isinstance(ann_data, list):
                objects_to_check = ann_data
            
            for obj in objects_to_check:
                shape = color = None
                
                if 'attributes' in obj:
                    attrs = obj['attributes']
                    shape = attrs.get('shape', '')
                    color = attrs.get('color', '')
                elif 'shape' in obj and 'color' in obj:
                    shape = obj['shape']
                    color = obj['color']
                
                if shape and color:
                    class_name = f"{color}_{shape}"
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts:
            print(f"✓ Class distribution:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} objects")
                if count < 3:
                    warnings.append(f"Very few examples for class '{class_name}' ({count})")
    
    except Exception as e:
        warnings.append(f"Could not analyze class distribution: {e}")
    
    # REPORT RESULTS
    if issues:
        print(f"\n❌ VALIDATION FAILED - {len(issues)} critical issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    if warnings:
        print(f"\n⚠ VALIDATION PASSED with {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print(f"\n✅ VALIDATION PASSED - Dataset looks good for training")
    
    return True


def parse_xml_annotations(xml_file):
    """
    Parse XML annotation files (CVAT, Pascal VOC formats)
    
    LOGIC:
    - Parse XML structure to extract image and annotation data
    - Handle CVAT cuboid format with attributes
    - Convert to unified annotation format for processing
    - Support both 2D bounding boxes and 3D cuboids
    
    Args:
        xml_file (Path): Path to XML annotation file
    
    Returns:
        dict: Dictionary mapping image names to annotation data
    """
    import xml.etree.ElementTree as ET
    
    annotations = {}
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        print(f"Parsing XML file: {xml_file}")
        
        # CVAT format detection
        if root.tag == 'annotations':
            print("Detected CVAT XML format")
            
            # Process each image in CVAT format
            for image_elem in root.findall('image'):
                img_name = image_elem.get('name')
                img_width = int(image_elem.get('width', 0))
                img_height = int(image_elem.get('height', 0))
                
                if not img_name:
                    continue
                
                # Initialize annotation structure
                img_annotations = {
                    'filename': img_name,
                    'width': img_width,
                    'height': img_height,
                    'objects': []
                }
                
                # Process cuboids (3D annotations)
                for cuboid_elem in image_elem.findall('cuboid'):
                    label = cuboid_elem.get('label', '')
                    
                    # Extract attributes (shape and color)
                    shape = None
                    color = None
                    
                    for attr_elem in cuboid_elem.findall('attribute'):
                        attr_name = attr_elem.get('name')
                        attr_value = attr_elem.text
                        
                        if attr_name == 'shape':
                            shape = attr_value
                        elif attr_name == 'color':
                            color = attr_value
                    
                    # Extract cuboid coordinates
                    # CVAT cuboid format: xtl1, ytl1, xbl1, ybl1, xtr1, ytr1, xbr1, ybr1, xtl2, ytl2, xbl2, ybl2, xtr2, ytr2, xbr2, ybr2
                    cuboid_points = []
                    point_attrs = ['xtl1', 'ytl1', 'xbl1', 'ybl1', 'xtr1', 'ytr1', 'xbr1', 'ybr1',
                                  'xtl2', 'ytl2', 'xbl2', 'ybl2', 'xtr2', 'ytr2', 'xbr2', 'ybr2']
                    
                    # Group coordinates into points (x, y pairs)
                    coords = []
                    for attr in point_attrs:
                        coord = cuboid_elem.get(attr)
                        if coord:
                            coords.append(float(coord))
                    
                    # Convert to point list format [x, y] for processing
                    if len(coords) >= 16:  # Need all 8 3D points
                        # Extract all unique x,y coordinates for 2D bounding box conversion
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                cuboid_points.append([coords[i], coords[i + 1]])
                    
                    # Create annotation object
                    if label and shape and color and cuboid_points:
                        annotation_obj = {
                            'label': label,
                            'shape': shape,
                            'color': color,
                            'points': cuboid_points,
                            'type': 'cuboid'
                        }
                        img_annotations['objects'].append(annotation_obj)
                
                # Process regular bounding boxes if present
                for box_elem in image_elem.findall('box'):
                    label = box_elem.get('label', '')
                    xtl = float(box_elem.get('xtl', 0))
                    ytl = float(box_elem.get('ytl', 0))
                    xbr = float(box_elem.get('xbr', 0))
                    ybr = float(box_elem.get('ybr', 0))
                    
                    # Extract attributes
                    shape = None
                    color = None
                    
                    for attr_elem in box_elem.findall('attribute'):
                        attr_name = attr_elem.get('name')
                        attr_value = attr_elem.text
                        
                        if attr_name == 'shape':
                            shape = attr_value
                        elif attr_name == 'color':
                            color = attr_value
                    
                    # Convert bounding box to points format
                    bbox_points = [
                        [xtl, ytl], [xbr, ytl], [xbr, ybr], [xtl, ybr]
                    ]
                    
                    if label and shape and color:
                        annotation_obj = {
                            'label': label,
                            'shape': shape,
                            'color': color,
                            'points': bbox_points,
                            'type': 'box'
                        }
                        img_annotations['objects'].append(annotation_obj)
                
                # Store annotations if we found any objects
                if img_annotations['objects']:
                    annotations[img_name] = img_annotations
                    print(f"  {img_name}: {len(img_annotations['objects'])} objects")
        
        # Pascal VOC format (if needed in future)
        elif root.tag == 'annotation':
            print("Detected Pascal VOC XML format")
            # TODO: Implement Pascal VOC parsing if needed
            pass
        
        else:
            print(f"Unknown XML format with root tag: {root.tag}")
    
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
    
    return annotations


def find_matching_annotation(img_file, annotations):
    """
    Find annotation for image file with flexible filename matching
    
    LOGIC:
    - Try exact filename match first
    - Try stem (filename without extension) match
    - Try stem with common image extensions
    - Return None if no match found
    
    Args:
        img_file (Path): Path object for image file
        annotations (dict): Dictionary of loaded annotations
    
    Returns:
        dict or None: Matching annotation data, None if not found
    """
    img_name = img_file.name
    img_stem = img_file.stem
    
    # METHOD 1: Try exact filename match first
    if img_name in annotations:
        return annotations[img_name]
    
    # METHOD 2: Try stem match (filename without extension)
    if img_stem in annotations:
        return annotations[img_stem]
    
    # METHOD 3: Try stem with common image extensions
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        test_name = img_stem + ext
        if test_name in annotations:
            return annotations[test_name]
    
    # No match found
    return None


def cuboid_to_bbox(cuboid_points,shape):
    """
    Convert 3D cuboid points to 2D bounding box by finding min/max x,y coordinates
    
    LOGIC:
    - Extract all x,y coordinates from cuboid vertices (ignore z if present)
    - Find minimum and maximum x,y values
    - Return as bounding box (x_min, y_min, x_max, y_max)
    
    ROBUSTNESS:
    - Handle different point formats: [[x,y], [x,y,z]], [{'x':x, 'y':y}], etc.
    - Return None if no valid points found
    - Convert all coordinates to float for consistency
    
    Args:
        cuboid_points (list): List of cuboid vertices in various formats
                             Could be: [[x,y], [x,y,z]], [{'x':x, 'y':y}], etc.
        shape (str): Shape attribute (square or circle), will change how the bbox is calculated
    
    Returns:
        tuple: (x_min, y_min, x_max, y_max) bounding box coordinates
               None if conversion fails
    """
    if not cuboid_points:
        return None
    
    # Extract x and y coordinates from various point formats
    x_coords = []
    y_coords = []

    if shape == 'square':
        
        for point in cuboid_points:
            # CASE 1: Point is a list/tuple [x, y] or [x, y, z]
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x_coords.append(float(point[0]))
                y_coords.append(float(point[1]))
            
            # CASE 2: Point is a dictionary {'x': x_val, 'y': y_val}
            elif isinstance(point, dict):
                if 'x' in point and 'y' in point:
                    x_coords.append(float(point['x']))
                    y_coords.append(float(point['y']))
        
        # VALIDATION: Check if we extracted any valid coordinates
        if not x_coords or not y_coords:
            return None
                
    # CONVERSION: Find bounding box from all points
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # SANITY CHECK: Ensure we have a valid bounding box
    if x_max <= x_min or y_max <= y_min:
        print(f"Warning: Invalid bounding box - width or height is zero or negative")
        return None
        
    return (x_min, y_min, x_max, y_max)

def get_class_id(shape, color, strict_mode=True):
    """
    Map shape and color attributes to class ID for the 6-class system
    
    LOGIC:
    - Combine shape (square/circle) with color (red/green/blue) 
    - Map to class IDs 0-5 for YOLO training
    - Use consistent lowercase comparison for robustness
    
    CLASS MAPPING:
    0: red_square    1: red_circle
    2: green_square  3: green_circle  
    4: blue_square   5: blue_circle
    
    Args:
        shape (str): Shape name (square or circle)
        color (str): Color name (red, green, or blue)
        strict_mode (bool): If True, return None for unknown combinations
    
    Returns:
        int or None: Class ID (0-5), None if unknown and strict_mode=True
    """
    # Define the 6-class mapping as specified in requirements
    class_mapping = {
        'red_square': 0,
        'red_circle': 1,
        'green_square': 2,
        'green_circle': 3,
        'blue_square': 4,
        'blue_circle': 5
    }
    
    # Normalize inputs to lowercase for consistent matching
    class_name = f"{color.lower()}_{shape.lower()}"
    
    # Handle unknown combinations based on strict mode
    if class_name not in class_mapping:
        if strict_mode:
            print(f"Error: Unknown class combination '{class_name}' - skipping annotation")
            return None  # Skip this annotation
        else:
            print(f"Warning: Unknown class combination '{class_name}', defaulting to red_square")
            return 0
    
    return class_mapping[class_name]

def validate_yolo_coordinates(center_x, center_y, width, height, img_name, obj_idx):
    """
    Validate and fix YOLO coordinates to ensure they are within valid ranges
    
    LOGIC:
    - Check if coordinates are within 0-1 range (YOLO requirement)
    - Clamp invalid coordinates to valid bounds
    - Log warnings for any fixes applied
    - Ensure width/height are positive and non-zero
    
    Args:
        center_x, center_y, width, height (float): YOLO format coordinates
        img_name (str): Image filename for logging
        obj_idx (int): Object index for logging
    
    Returns:
        tuple: (center_x, center_y, width, height) - validated coordinates
        bool: True if coordinates are valid, False if invalid and unfixable
    """
    issues = []
    
    # VALIDATION: Check and fix center coordinates (0-1 range)
    if not (0 <= center_x <= 1):
        issues.append(f"center_x {center_x:.4f} out of bounds")
        center_x = max(0, min(1, center_x))
    
    if not (0 <= center_y <= 1):
        issues.append(f"center_y {center_y:.4f} out of bounds")
        center_y = max(0, min(1, center_y))
    
    # VALIDATION: Check and fix dimensions (positive, non-zero, max 1.0)
    if not (0 < width <= 1):
        issues.append(f"width {width:.4f} invalid")
        if width <= 0:
            print(f"Error: Zero or negative width for object {obj_idx} in {img_name} - skipping")
            return None, None, None, None, False
        width = min(1, width)
    
    if not (0 < height <= 1):
        issues.append(f"height {height:.4f} invalid")
        if height <= 0:
            print(f"Error: Zero or negative height for object {obj_idx} in {img_name} - skipping")
            return None, None, None, None, False
        height = min(1, height)
    
    # LOG: Report any fixes applied
    if issues:
        print(f"Fixed coordinate issues for object {obj_idx} in {img_name}: {issues}")
    
    return center_x, center_y, width, height, True

def convert_to_yolo_format(annotations, images_dir, output_dir):
    """
    Convert loaded annotations to YOLO format and filter out images without annotations
    
    MAIN LOGIC:
    1. Create YOLO directory structure (images/train, labels/train)
    2. For each image file in images_dir (including subdirectories):
       a. Check if it has corresponding annotations
       b. If no annotations, skip the image (filtering requirement)
       c. Load image to get dimensions for normalization
       d. Process each annotation object:
          - Extract shape and color attributes
          - Extract cuboid points
          - Convert cuboid to 2D bounding box
          - Convert to YOLO format (normalized coordinates)
          - Map to appropriate class ID
       e. Copy image and save YOLO label file
    
    YOLO FORMAT: Each line in label file is:
    class_id center_x center_y width height
    Where all coordinates are normalized (0.0 to 1.0)
    
    Args:
        annotations (dict): Loaded annotations mapping image names to annotation data
        images_dir (str): Path to images directory
        output_dir (str): Path to output directory for YOLO dataset
    
    Returns:
        list: List of successfully processed image files
    """
    # STEP 1: Create YOLO directory structure
    train_images_dir = Path(output_dir) / "images" / "train"
    train_labels_dir = Path(output_dir) / "labels" / "train"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    
    processed_images = []
    
    # STEP 2: Find all image files in the images directory and subdirectories
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']
    image_files = []
    
    # UPDATED: Recursively search through all subdirectories
    images_path = Path(images_dir)
    for ext in image_extensions:
        # Use ** to search recursively through all subdirectories
        image_files.extend(images_path.rglob(f"*{ext}"))
    
    print(f"Found {len(image_files)} image files in {images_dir} and subdirectories")
    
    # Log subdirectories found
    subdirs = set()
    for img_file in image_files:
        subdir = img_file.parent.relative_to(images_path)
        if str(subdir) != '.':  # Not in root images directory
            subdirs.add(str(subdir))
    
    if subdirs:
        print(f"Images found in subdirectories: {', '.join(sorted(subdirs))}")
    
    successful_conversions = 0
    
    # STEP 3: Process each image file
    for img_file in image_files:
        img_name = img_file.name
        
        # STEP 3a: Get correct entry from annotations.xml
        annotation_data = find_matching_annotation(img_file, annotations)
        
        # FILTERING: Skip images without annotations as requested
        if annotation_data is None:
            print(f"No annotations found for {img_name}, skipping...")
            continue
        
        # STEP 3b: Load image to get dimensions for YOLO normalization
        try:
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Could not load image {img_name}, skipping...")
                continue
            
            img_height, img_width = image.shape[:2]
            print(f"Processing {img_name}: {img_width}x{img_height}")
            
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            continue
        
        # STEP 3c: Process annotations and convert to YOLO format
        yolo_annotations = []
        
        # ANNOTATION PARSING: Handle different annotation structures
        objects_to_process = []
        
        if isinstance(annotation_data, dict):
            # COCO-like structure with annotations array
            if 'annotations' in annotation_data:
                objects_to_process = annotation_data['annotations']
            # Custom structure with objects array
            elif 'objects' in annotation_data:
                objects_to_process = annotation_data['objects']
            # LabelMe-like structure with shapes array
            elif 'shapes' in annotation_data:
                objects_to_process = annotation_data['shapes']
            else:
                # Single object annotation (treat entire dict as one object)
                objects_to_process = [annotation_data]
        elif isinstance(annotation_data, list):
            # Direct list of objects
            objects_to_process = annotation_data
        

        ''' EXAMPLE CVAT ANNOTATION
        <image id="363" name="DSC_0708.JPG" subset="default" task_id="4" width="6000" height="4000">
            <cuboid label="part" source="manual" occluded="0" xtl1="3067.22" ytl1="2059.48" xbl1="3067.22" ybl1="2105.82" xtr1="3121.51" ytr1="2191.52" xbr1="3121.51" ybr1="2237.95" xtl2="3340.02" ytl2="2160.20" xbl2="3340.02" ybl2="2206.44" xtr2="3285.73" ytr2="2028.07" xbr2="3285.73" ybr2="2074.31" z_order="0">
            <attribute name="shape">square</attribute>
            <attribute name="color">green</attribute>
            </cuboid>
            <cuboid label="part" source="manual" occluded="0" xtl1="3091.62" ytl1="2102.25" xbl1="3091.62" ybl1="2148.58" xtr1="3145.91" ytr1="2234.28" xbr1="3145.91" ybr1="2280.71" xtl2="3364.42" ytl2="2202.97" xbl2="3364.42" ybl2="2249.20" xtr2="3309.90" ytr2="2071.57" xbr2="3309.90" ybr2="2117.71" z_order="0">
            <attribute name="shape">circle</attribute>
            <attribute name="color">blue</attribute>
            </cuboid>
        '''

        # STEP 3d: Process each object annotation
        for obj_idx, obj in enumerate(objects_to_process):
            try:
                # ATTRIBUTE EXTRACTION: Get shape and color from various possible locations
                shape = None
                color = None
                cuboid_points = None
                
                # METHOD 1: Explicit attributes dictionary
                if 'attributes' in obj:
                    attrs = obj['attributes']
                    shape = attrs.get('shape', '')
                    color = attrs.get('color', '')
                
                # METHOD 2: Direct shape and color fields (XML/CVAT format)
                elif 'shape' in obj and 'color' in obj:
                    shape = obj['shape']
                    color = obj['color']
                
                # METHOD 3: Parse from label string (fallback)
                elif 'label' in obj:
                    label = obj['label'].lower()
                    # Extract shape from label
                    if 'square' in label:
                        shape = 'square'
                    elif 'circle' in label:
                        shape = 'circle'
                    
                    # Extract color from label
                    if 'red' in label:
                        color = 'red'
                    elif 'green' in label:
                        color = 'green'
                    elif 'blue' in label:
                        color = 'blue'
                
                # Annotations will alwways be in CVAT Format
                #    CVAT cuboid: [xtl1, ytl1, xbl1, ybl1, xtr1, ytr1, xbr1, ybr1, xtl2, ytl2, xbl2, ybl2, xtr2, ytr2, xbr2, ybr2]
                #    CVAT bbox: [xtl, ytl, xbr, ybr]
                # Need to be convered to YOLO bbox: [x_center, y_center, width, height] (normalized)
                

                if 'cuboid' in obj:
                    cuboid_points = obj['cuboid']
                    bbox = cuboid_to_bbox(cuboid_points,shape)
                elif 'bbox' in obj:
                    # If already a bounding box, they will be in CVAT format: [x_min, y_min, x_max, y_max]
                    bbox = obj['bbox']
                
                
                center_x = (bbox[0] + bbox[2]) / 2 / img_width
                center_y = (bbox[1] + bbox[3]) / 2 / img_height
                width = (bbox[2] - bbox[0]) / img_width
                height = (bbox[3] - bbox[1]) / img_height

                # BOUNDS CHECKING: Validate and fix normalized coordinates
                center_x, center_y, width, height, is_valid = validate_yolo_coordinates(
                    center_x, center_y, width, height, img_name, obj_idx
                )
                
                if not is_valid:
                    print(f"Skipping object {obj_idx} in {img_name} due to invalid coordinates")
                    continue
                
                # CLASS MAPPING: Get class ID for this shape+color combination (strict mode)
                class_id = get_class_id(shape, color, strict_mode=True)
                if class_id is None:
                    print(f"Skipping object {obj_idx} in {img_name} due to unknown class combination")
                    continue
                
                # FORMAT: Create YOLO annotation line
                yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                yolo_annotations.append(yolo_line)
                
                print(f"  Object {obj_idx}: {color} {shape} -> class {class_id}")
                
            except Exception as e:
                print(f"Error processing object {obj_idx} in {img_name}: {e}")
                continue
        
        # STEP 3e: Save files only if we have valid annotations
        if not yolo_annotations:
            print(f"No valid annotations processed for {img_name}, skipping...")
            continue
        
        # FILE OPERATIONS: Copy image and create label file
        try:
            # Copy image to training directory (flatten structure - all images go to train folder)
            shutil.copy2(img_file, train_images_dir / img_name)
            
            # Write YOLO label file (same name as image but .txt extension)
            label_file = train_labels_dir / f"{img_file.stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            processed_images.append(img_name)
            successful_conversions += 1
            
            print(f"✓ Processed {img_name}: {len(yolo_annotations)} objects")
            
        except Exception as e:
            print(f"Error copying files for {img_name}: {e}")
            continue
    
    print(f"\n=== CONVERSION SUMMARY ===")
    print(f"Successfully processed {successful_conversions} images with annotations")
    print(f"Skipped {len(image_files) - successful_conversions} images (no annotations or errors)")
    
    return processed_images

def create_yaml_config(output_dir, class_names):
    """
    Create YAML configuration file for YOLOv11 training
    
    LOGIC:
    - Define dataset paths (train images and labels)
    - List all class names with their IDs
    - Save in YOLO-expected format for training
    
    Args:
        output_dir (str): Path to dataset output directory
        class_names (list): List of class names in order (index = class_id)
    
    Returns:
        str: Path to created YAML config file
    """
    yaml_content = {
        'path': str(Path(output_dir).absolute()),  # Dataset root path
        'train': 'images/train',  # Relative path to training images
        'val': 'images/train',    # Using same for validation (small dataset)
        'nc': len(class_names),   # Number of classes
        'names': class_names      # Class names list (index = class_id)
    }
    
    # Write YAML file to dataset directory
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created YAML config: {yaml_path}")
    print(f"Dataset root: {yaml_content['path']}")
    print(f"Classes ({len(class_names)}): {class_names}")
    
    return str(yaml_path)

def train_yolo_model(yaml_config_path, output_dir):
    """
    Train YOLOv11 model using the prepared dataset
    
    TRAINING LOGIC:
    - Load YOLOv11 base model (smallest version for faster training)
    - Configure training parameters for custom dataset
    - Run training loop with validation
    - Save best model for deployment
    
    Args:
        yaml_config_path (str): Path to dataset YAML configuration
        output_dir (str): Path to save training results
    
    Returns:
        str: Path to best trained model
    """
    print("\n=== STARTING YOLO TRAINING ===")
    
    # STEP 0: Check GPU availability and aggressively clear memory
    import torch
    import gc
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB total)")
            print(f"    Before clear: {memory_allocated:.1f} GB allocated, {memory_cached:.1f} GB cached")
        
        # AGGRESSIVE GPU MEMORY CLEARING
        print("🧹 Aggressively clearing GPU memory...")
        
        # Method 1: Standard PyTorch clearing
        torch.cuda.empty_cache()
        
        # Method 2: Force garbage collection
        gc.collect()
        
        # Method 3: Additional PyTorch cleanup
        try:
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"    Advanced cleanup failed (not critical): {e}")
        
        # Method 4: Try to reset cached memory allocator
        try:
            # This is more aggressive and may help with stubborn cached memory
            if hasattr(torch.cuda, 'memory_summary'):
                print("    Detailed memory before reset:")
                print(f"    {torch.cuda.memory_summary(0)}")
            
            # Reset the memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
        except Exception as e:
            print(f"    Memory stats reset failed (not critical): {e}")
        
        # Method 5: Force another round of cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        print("    ✓ GPU memory clearing completed")
        
        # Show memory after clearing
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            available_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free_memory = available_memory - memory_allocated - memory_cached
            print(f"    After clear:  {memory_allocated:.1f} GB allocated, {memory_cached:.1f} GB cached")
            print(f"    Estimated free: {free_memory:.1f} GB")
        
        # Set device to GPU
        device = '0' if torch.cuda.device_count() > 0 else 'cpu'
        print(f"Using device: GPU {device}")
        
        # Test GPU functionality with small tensor
        try:
            # Use a small tensor to test without using much memory
            test_tensor = torch.randn(10, 10).cuda()
            print(f"  ✓ GPU test successful")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ❌ GPU test failed: {e}")
            device = 'cpu'
            print("  Falling back to CPU")
            
    else:
        device = 'cpu'
        print("⚠ Warning: CUDA not available, using CPU (training will be much slower)")
        print("Diagnostics:")
        print(f"  PyTorch CUDA compiled: {torch.backends.cudnn.enabled if hasattr(torch.backends, 'cudnn') else 'N/A'}")
        print("To fix GPU issues:")
        print("  1. Check: python -c 'import torch; print(torch.cuda.is_available())'")
        print("  2. Kill GPU processes: kill -9 139511")
        print("  3. Reinstall PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("  4. Make sure NVIDIA drivers are properly installed")
    
    # STEP 1: Load YOLOv11 base model
    # Using 'nano' version for faster training - can upgrade to 's', 'm', 'l', 'x' for better accuracy
    model = YOLO('yolo11n.pt')  # Downloads pretrained weights automatically

    # Create unique model name with timestamp
    timestamp = datetime.now().strftime("%y_%m_%d_%H_%M")
    model_name = f'part_detection_{timestamp}'
    
    # STEP 2: Configure training parameters
    training_args = {
        'data': yaml_config_path,          # Path to dataset YAML
        'epochs': 700,                     # INCREASED: Maximum epochs for best possible convergence
        'imgsz': 800,                     # Input image size for training
        'batch': 16,                       # Batch size (reduce if out of memory)
        'lr0': 0.01,                      # ORIGINAL: Back to original learning rate
        'lrf': 0.01,                      # ORIGINAL: Back to original final learning rate
        'momentum': 0.937,                 # SGD momentum
        'weight_decay': 0.0005,           # Weight decay for regularization
        'warmup_epochs': 3,               # Warmup epochs for learning rate
        'warmup_momentum': 0.8,           # Warmup momentum
        'box': 7.5,                       # Box loss gain
        'cls': 0.5,                       # Classification loss gain
        'dfl': 1.5,                       # Distribution focal loss gain
        'project': output_dir,            # Project directory
        'name': model_name,                # CHANGED: New run name for v5 with 400 epochs
        'save': True,                     # Save training checkpoints
        'save_period': 20,                # Save checkpoint every N epochs
        'cache': True,                   # Cache images for faster training (disable if low memory)
        'device': device,                 # UPDATED: Use detected device (GPU or CPU)
        'workers': 4,                     # Number of data loading workers
        'exist_ok': True,                 # Overwrite existing project/name
        'pretrained': True,               # Use pretrained weights
        'optimizer': 'SGD',               # Optimizer (SGD, Adam, AdamW)
        'verbose': True,                  # Verbose output
        'seed': 0,                        # Random seed for reproducibility
        'deterministic': True,            # Deterministic training for reproducibility
        'single_cls': False,              # Treat as single-class dataset
        'rect': False,                    # Rectangular training
        'cos_lr': True,                   # Enable cosine learning rate scheduler
        'close_mosaic': 10,               # Disable mosaic augmentation for last 10 epochs
        'resume': False,                  # Resume from last checkpoint
        'amp': True,                      # Automatic Mixed Precision training
        'fraction': 1.0,                  # Dataset fraction to use
        'profile': False,                 # Profile ONNX and TensorRT speeds
        'freeze': 3,                   # Freeze layers (list of layer indices)
        'multi_scale': True,             # DISABLED: Keep original scale for stability
        'overlap_mask': True,             # Overlap masks during training
        'mask_ratio': 4,                  # Mask downsample ratio
        'dropout': 0.5,                   # Use dropout regularization
        'val': True,                      # Validate/test during training
        
        # ORIGINAL LOW AUGMENTATION PARAMETERS - minimal augmentation for stability
        'bgr': 0,                         # Flips color channels, BGR augmentation (0-1)
        'hsv_h': 0.015,                   # HSV-Hue augmentation (±0.015) - original
        'hsv_s': 0.4,                     # HSV-Saturation augmentation - back to original
        'hsv_v': 0.3,                     # HSV-Value augmentation - back to original
        'degrees': 20,                    # ORIGINAL:
        'translate': 0.2,                 # ORIGINAL: Minimal translation
        'scale': 0.1,                     # ORIGINAL: Back to original scale variation
        'shear': 0.2,                     # ORIGINAL: No shear for stability
        'perspective': 0,               # ORIGINAL: No perspective transform
        'flipud': 0.1,                    # ORIGINAL: No vertical flip
        'fliplr': 0.5,                    # ORIGINAL: Keep horizontal flip
        'mosaic': .8,                    # Mosaic augmentation probability
        'mixup': 0.2,                     # ORIGINAL: No MixUp
        'copy_paste': 0.0,                # ORIGINAL: No copy-paste
        'erasing': 0.3,                   # ORIGINAL: Random erasing back to original value
    }
    
    # STEP 2b: Adjust batch size for CPU training or high GPU memory usage
    if device == 'cpu':
        print("⚠ Reducing batch size for CPU training")
        training_args['batch'] = 8  # Smaller batch for CPU
        training_args['workers'] = 4  # Fewer workers for CPU
    elif device != 'cpu' and torch.cuda.is_available():
        # Check available GPU memory and adjust batch size if needed
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        estimated_free = total_memory - allocated_memory - cached_memory
        
        print(f"GPU Memory Analysis:")
        print(f"  Total: {total_memory:.1f} GB")
        print(f"  Allocated: {allocated_memory:.1f} GB")  
        print(f"  Cached: {cached_memory:.1f} GB")
        print(f"  Estimated free: {estimated_free:.1f} GB")
        
        # Be conservative with memory estimates due to existing process
        if estimated_free < 6:  # Very conservative due to existing 6.5GB usage
            print("⚠ Very limited GPU memory available, using small batch size")
            training_args['batch'] = 4
            training_args['workers'] = 4
        elif estimated_free < 8:
            print("⚠ Limited GPU memory available, reducing batch size")
            training_args['batch'] = 8
            training_args['workers'] = 6
        elif estimated_free < 12:
            print("⚠ Moderate GPU memory available, using smaller batch size")
            training_args['batch'] = 12
        # else keep default batch size of 16
    
    # STEP 3: Start training
    print("Training configuration:")
    for key, value in training_args.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training... This may take a while depending on your dataset size and hardware.")
    
    try:
        # Execute training
        results = model.train(**training_args)
        
        # STEP 4: Find and return path to best model
        run_dir = Path(output_dir) / model_name  # FIXED: Use dynamic model_name instead of hardcoded string
        best_model_path = run_dir / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            print(f"\n=== TRAINING COMPLETED ===")
            print(f"Best model saved to: {best_model_path}")
            print(f"Training results in: {run_dir}")
            
            # Validate model loading
            try:
                test_model = YOLO(str(best_model_path))
                print(f"✓ Model validation successful")
                return str(best_model_path)
            except Exception as e:
                print(f"⚠ Warning: Model validation failed: {e}")
                return str(best_model_path)  # Return path anyway
        else:
            print(f"⚠ Warning: Best model not found at expected location: {best_model_path}")
            # Look for last.pt as fallback
            last_model_path = run_dir / 'weights' / 'last.pt'
            if last_model_path.exists():
                print(f"Using last checkpoint: {last_model_path}")
                return str(last_model_path)
            else:
                print("No trained model found!")
                return None
                
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("This could be due to:")
        print("  - Insufficient memory (try reducing batch size)")
        print("  - No valid annotations found")
        print("  - Corrupted image files")
        print("  - Missing dependencies")
        return None

def main():
    """
    Main training pipeline - orchestrates the entire process
    
    PIPELINE FLOW:
    1. Validate input directories exist
    2. Load and parse all annotations
    3. Validate dataset integrity
    4. Convert annotations to YOLO format
    5. Create YOLO configuration file
    6. Train YOLOv11 model
    7. Report results and next steps
    """
    # CONFIGURATION: Set input and output paths
    images_dir = "./images"
    annotations_dir = "./annotations"
    output_dir = "./yolo_training"
    
    # CLASS NAMES: Define the 6 classes in order (index = class_id)
    class_names = [
        'red_square',    # class_id: 0
        'red_circle',    # class_id: 1
        'green_square',  # class_id: 2
        'green_circle',  # class_id: 3
        'blue_square',   # class_id: 4
        'blue_circle'    # class_id: 5
    ]
    
    print("=== YOLOv11 CUSTOM TRAINING PIPELINE ===")
    print(f"Images directory: {images_dir}")
    print(f"Annotations directory: {annotations_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target classes: {class_names}")
    
    # STEP 1: Validate input directories
    if not Path(images_dir).exists():
        print(f"❌ Images directory not found: {images_dir}")
        print("Please create the ./images/ directory and add your image files")
        return
    
    if not Path(annotations_dir).exists():
        print(f"❌ Annotations directory not found: {annotations_dir}")
        print("Please create the ./annotations/ directory and add your annotation files")
        return
    
    # STEP 2: Load annotations
    print(f"\n=== LOADING ANNOTATIONS ===")
    annotations = load_annotations(annotations_dir)
    
    if not annotations:
        print("❌ No annotations loaded. Please check your annotation files.")
        return
    
    # STEP 3: Validate dataset
    if not validate_dataset(annotations, images_dir):
        print("❌ Dataset validation failed. Please fix the issues above before training.")
        return
    
    # STEP 4: Convert to YOLO format
    print(f"\n=== CONVERTING TO YOLO FORMAT ===")
    processed_images = convert_to_yolo_format(annotations, images_dir, output_dir)
    
    if not processed_images:
        print("❌ No images were successfully processed. Please check your data.")
        return
    
    # STEP 5: Create YAML configuration
    print(f"\n=== CREATING YOLO CONFIG ===")
    yaml_config_path = create_yaml_config(output_dir, class_names)
    
    # STEP 6: Train model
    print(f"\n=== TRAINING MODEL ===")
    best_model_path = train_yolo_model(yaml_config_path, output_dir)
    
    if best_model_path:
        print(f"\n🎉 SUCCESS! Training completed successfully.")
        print(f"\n=== NEXT STEPS ===")
        print(f"1. Your trained model is saved at: {best_model_path}")
        print(f"2. To use with YOLO-3D:")
        print(f"   a. Copy {best_model_path} to your YOLO-3D directory")
        print(f"   b. Update the model path in YOLO-3D run.py")
        print(f"   c. Update class names to: {class_names}")
        print(f"3. Test your model:")
        print(f"   from ultralytics import YOLO")
        print(f"   model = YOLO('{best_model_path}')")
        print(f"   results = model('path/to/test/image.jpg')")
    else:
        print("❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()