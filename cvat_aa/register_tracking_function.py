#!/usr/bin/env python3

import os
import sys
from cvat_cli._internal.client import CLI
from cvat_cli._internal.common import make_client
import argparse

def register_tracking_function():
    """Register the YOLO tracking function as a native function"""
    
    # CVAT server details
    server_host = "http://localhost:8080"
    username = "scblum2"
    password = "Basketball1!"
    
    # Function details
    function_name = "YOLO Tracking"
    function_file = "/home/scblum/Projects/testbed_cv/cvat_aa/cvat_aa_yolo_tracking.py"
    
    print(f"Registering tracking function: {function_name}")
    print(f"Function file: {function_file}")
    
    # Build the command
    cmd = [
        "cvat-cli", 
        "--server-host", server_host,
        "--auth", f"{username}:{password}",
        "function", "create-native",
        function_name,
        "--function-file", function_file
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the registration
    try:
        os.system(' '.join(cmd))
        print(f"✓ Successfully registered function: {function_name}")
        print("You can now use this function in the CVAT web UI for tracking!")
    except Exception as e:
        print(f"✗ Failed to register function: {e}")

if __name__ == "__main__":
    register_tracking_function()
