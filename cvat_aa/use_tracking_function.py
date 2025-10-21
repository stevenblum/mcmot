#!/usr/bin/env python3

import os

def use_tracking_function(task_id):
    """Use the registered tracking function on a task"""
    
    # CVAT server details
    server_host = "http://localhost:8080"
    username = "scblum2"
    password = "Basketball1!"
    
    function_name = "YOLO Tracking"
    
    print(f"Using tracking function '{function_name}' on task {task_id}")
    
    # Build the command for tracking
    cmd = [
        "cvat-cli",
        "--server-host", server_host,
        "--auth", f"{username}:{password}",
        "task", "auto-annotate", str(task_id),
        "--function", function_name,
        "--clear-existing",
        "--allow-unmatched-labels"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        os.system(' '.join(cmd))
        print(f"✓ Successfully applied tracking function to task {task_id}")
    except Exception as e:
        print(f"✗ Failed to apply tracking function: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python use_tracking_function.py <task_id>")
        sys.exit(1)
    
    task_id = int(sys.argv[1])
    use_tracking_function(task_id)
