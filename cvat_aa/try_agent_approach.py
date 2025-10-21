#!/usr/bin/env python3
"""
Try to use the CVAT agent approach for tracking function
"""

import os
import subprocess

def try_create_and_run_agent():
    """Try to create a native function and run an agent"""
    
    server_host = "http://localhost:8080"
    auth = "scblum2:Basketball1!"
    function_file = "/home/scblum/Projects/testbed_cv/cvat_aa/cvat_aa_yolo_tracking.py"
    
    print("Attempting to create native function...")
    
    # Step 1: Try to create the function
    create_cmd = [
        "cvat-cli", "--server-host", server_host, "--auth", auth,
        "function", "create-native", "YOLO Tracking",
        "--function-file", function_file
    ]
    
    print(f"Command: {' '.join(create_cmd)}")
    
    try:
        result = subprocess.run(create_cmd, capture_output=True, text=True)
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
        if result.returncode == 0:
            # Extract function ID from output
            lines = result.stdout.strip().split('\n')
            function_id = None
            for line in lines:
                if 'ID' in line or line.isdigit():
                    function_id = line.strip()
                    break
            
            if function_id:
                print(f"Function created with ID: {function_id}")
                
                # Step 2: Try to run the agent
                agent_cmd = [
                    "cvat-cli", "--server-host", server_host, "--auth", auth,
                    "function", "run-agent", function_id,
                    "--function-file", function_file
                ]
                
                print(f"Running agent: {' '.join(agent_cmd)}")
                subprocess.run(agent_cmd)
            else:
                print("Could not extract function ID from output")
        else:
            print("Function creation failed - likely need CVAT Enterprise/Cloud")
            
    except Exception as e:
        print(f"Error: {e}")

def check_cvat_version():
    """Check what version of CVAT is running"""
    
    print("Checking CVAT version and capabilities...")
    
    # Try to list functions
    cmd = ["cvat-cli", "--server-host", "http://localhost:8080", 
           "--auth", "scblum2:Basketball1!", "function", "list"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Function list result: {result.returncode}")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error checking functions: {e}")

def suggest_alternatives():
    """Suggest alternative approaches"""
    
    print("\n" + "="*60)
    print("ALTERNATIVE APPROACHES")
    print("="*60)
    
    print("\n1. Use as local function file (current approach):")
    print("   cvat-cli task auto-annotate 32 \\")
    print("     --function-file /path/to/your/tracking/function.py")
    print("   Note: Only works with DetectionFunctionSpec")
    
    print("\n2. Convert to DetectionFunctionSpec:")
    print("   - Modify your function to use DetectionFunctionSpec")
    print("   - Lose tracking capabilities but gain CLI compatibility")
    
    print("\n3. Use CVAT SDK directly:")
    print("   - Write Python scripts using cvat_sdk")
    print("   - More control but requires custom implementation")
    
    print("\n4. Use Docker serverless functions:")
    print("   - Package your function in a Docker container")
    print("   - Deploy as serverless function")
    
    print("\n5. Upgrade to CVAT Enterprise/Cloud:")
    print("   - Get access to native function features")
    print("   - Commercial solution")

if __name__ == "__main__":
    check_cvat_version()
    try_create_and_run_agent()
    suggest_alternatives()
