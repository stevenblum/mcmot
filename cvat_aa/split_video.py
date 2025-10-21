'''

python3 split_video.py "/home/scblum/Projects/testbed_cv/raw_images/2025 06 01 testbed simulation/best_video_normal.mp4" 10

'''

import subprocess
import os
import sys
import argparse
from pathlib import Path

def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'csv=p=0', str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting video duration: {e}")
        return None

def split_video(input_path, num_segments, output_dir=None):
    """Split video into n equal segments."""
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return False
    
    # Get video duration
    duration = get_video_duration(input_path)
    if duration is None:
        return False
    
    # Calculate segment duration
    segment_duration = duration / num_segments
    
    # Set output directory
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_segments"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Splitting {input_path.name} ({duration:.2f}s) into {num_segments} segments of {segment_duration:.2f}s each")
    
    # Split video into segments
    for i in range(num_segments):
        start_time = i * segment_duration
        
        # Create individual folder for each segment
        segment_folder = output_dir / f"segment_{i+1:03d}"
        segment_folder.mkdir(exist_ok=True)
        
        output_file = segment_folder / f"{input_path.stem}_segment_{i+1:03d}.mp4"
        
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-ss', str(start_time),
            '-t', str(segment_duration),
            '-c', 'copy',  # Copy streams without re-encoding for speed
            '-y',  # Overwrite output files
            str(output_file)
        ]
        
        print(f"Creating segment {i+1}/{num_segments}: {segment_folder.name}/{output_file.name}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating segment {i+1}: {e}")
            return False
    
    print(f"Successfully created {num_segments} segments in {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Split MP4 video into equal segments")
    parser.add_argument("input_video", help="Path to input MP4 video")
    parser.add_argument("num_segments", type=int, help="Number of segments to create")
    parser.add_argument("-o", "--output", help="Output directory (default: input_filename_segments)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_segments <= 0:
        print("Error: Number of segments must be positive")
        return 1
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return 1
    
    # Split the video
    success = split_video(args.input_video, args.num_segments, args.output)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
