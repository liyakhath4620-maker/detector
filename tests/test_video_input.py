import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import os
import cv2
import numpy as np
import logging
from datetime import datetime

try:
    from detector.inputs.video_file import VideoFileSource
except ImportError:
    from inputs.video_file import VideoFileSource

logging.basicConfig(level=logging.INFO)

import urllib.request

def download_sample_video(filename: str):
    """Downloads a short sample video for realistic testing."""
    url = "https://download.samplelib.com/mp4/sample-5s.mp4"
    print(f"Downloading real sample video from {url}...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")

def run_test():
    video_path = "sample_test_video.mp4"
    if not os.path.exists(video_path):
        download_sample_video(video_path)

    # Initialize video source
    print("\n--- Testing VideoFileSource ---")
    start_time = datetime(2026, 9, 26, 12, 0, 0)
    source = VideoFileSource(filepath=video_path, sample_fps=2.0, start_time=start_time)
    
    frame_count = 0
    first_timestamp = None
    last_timestamp = None
    
    for frame_data in source.get_frames():
        if frame_count == 0:
            first_timestamp = frame_data.timestamp
            cv2.imwrite("first_frame_saved.jpg", frame_data.bgr_image)
            print("-> Saved first_frame_saved.jpg for visual verification.")
        last_timestamp = frame_data.timestamp
        frame_count += 1

        
    print(f"\nResults:")
    print(f"Total sampled frames: {frame_count}")
    print(f"First timestamp: {first_timestamp}")
    print(f"Last timestamp:  {last_timestamp}")

if __name__ == "__main__":
    run_test()
