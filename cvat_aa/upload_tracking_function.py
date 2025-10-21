#!/usr/bin/env python3
"""
Script to help upload and use the YOLO tracking function in CVAT web UI.
Since CLI native function creation might not be supported, this provides
instructions for manual upload.
"""

import os
import shutil

def prepare_function_for_upload():
    """Prepare the tracking function for upload to CVAT web UI"""
    
    source_file = "/home/scblum/Projects/testbed_cv/cvat_aa/cvat_aa_yolo_tracking.py"
    
    print("=" * 60)
    print("CVAT Tracking Function Upload Instructions")
    print("=" * 60)
    
    print(f"\n1. Your tracking function is ready at:")
    print(f"   {source_file}")
    
    print(f"\n2. To upload via CVAT Web UI:")
    print(f"   - Open CVAT web interface: http://localhost:8080")
    print(f"   - Login as: scblum2")
    print(f"   - Go to 'Functions' or 'Models' section")
    print(f"   - Look for 'Upload Function' or 'Add Function' button")
    print(f"   - Upload the file: {source_file}")
    
    print(f"\n3. Alternative: Copy function to CVAT functions directory")
    print(f"   - Find your CVAT installation directory")
    print(f"   - Look for a 'functions' or 'serverless' folder")
    print(f"   - Copy the tracking function there")
    
    print(f"\n4. Function Details:")
    print(f"   - Name: YOLO Tracking")
    print(f"   - Type: TrackingFunctionSpec")
    print(f"   - Supports: rectangle shapes")
    print(f"   - Features: Object detection + tracking with ByteTrack")
    
    print(f"\n5. Usage in CVAT UI:")
    print(f"   - Open a task with video/image sequence")
    print(f"   - Go to annotation view")
    print(f"   - Look for 'Auto Annotation' or 'AI Tools' menu")
    print(f"   - Select 'YOLO Tracking' function")
    print(f"   - Draw initial bounding boxes on first frame")
    print(f"   - Run tracking to propagate across frames")
    
    # Check if we can determine CVAT version
    print(f"\n6. Debugging the 404 error:")
    print(f"   - Your CVAT version might not support CLI function creation")
    print(f"   - Try checking CVAT documentation for your version")
    print(f"   - Alternative: Use as a local function file")
    
    return source_file

def create_standalone_script():
    """Create a standalone script that can be used independently"""
    
    script_path = "/home/scblum/Projects/testbed_cv/cvat_aa/standalone_tracking.py"
    
    script_content = '''#!/usr/bin/env python3
"""
Standalone YOLO tracking script for CVAT
This can be used independently or uploaded to CVAT
"""

# Copy your tracking function content here
# This makes it easier to upload to CVAT web UI

# Your existing imports and code...
'''
    
    # Copy the main function content
    source_file = "/home/scblum/Projects/testbed_cv/cvat_aa/cvat_aa_yolo_tracking.py"
    
    try:
        with open(source_file, 'r') as f:
            content = f.read()
        
        # Remove the multiline comment at the top
        lines = content.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("'''") and i > 0:
                start_idx = i + 1
                break
        
        clean_content = '\n'.join(lines[start_idx:])
        
        with open(script_path, 'w') as f:
            f.write(script_content)
            f.write(clean_content)
        
        print(f"\n7. Created standalone script:")
        print(f"   {script_path}")
        print(f"   This version is easier to upload to CVAT web UI")
        
    except Exception as e:
        print(f"\n7. Could not create standalone script: {e}")
    
    return script_path

if __name__ == "__main__":
    source_file = prepare_function_for_upload()
    standalone_file = create_standalone_script()
    
    print(f"\n" + "=" * 60)
    print("Next Steps:")
    print("1. Try uploading via CVAT web UI")
    print("2. Check CVAT documentation for function upload")
    print("3. Consider using Docker serverless functions")
    print("=" * 60)
