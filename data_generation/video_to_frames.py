"""
Video Data Generation Module - Convert video files to image frame datasets
"""

import os
import cv2
import glob
from tqdm import tqdm
from datetime import datetime

from config import (
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLING_INTERVAL,
    OUTPUT_DATASET_DIR,
    SUPPORTED_VIDEO_FORMATS,
    OUTPUT_IMAGE_FORMAT,
    IMAGE_QUALITY
)


class VideoToFrames:
    """Process video files and extract frames to generate training datasets"""
    
    def __init__(
        self, 
        input_path, 
        output_path=None, 
        resolution=DEFAULT_RESOLUTION, 
        sampling_interval=DEFAULT_SAMPLING_INTERVAL
    ):
        """
        Initialize video processor
        
        Parameters:
            input_path (str): Input video file or directory containing videos
            output_path (str, optional): Output directory path
            resolution (tuple, optional): Output image resolution as (width, height)
            sampling_interval (int, optional): Milliseconds between frames
        """
        self.input_path = input_path
        self.output_path = output_path or os.path.join(os.getcwd(), OUTPUT_DATASET_DIR)
        self.resolution = resolution
        self.sampling_interval = sampling_interval
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Log file
        self.log_file = os.path.join(self.output_path, f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
    def log(self, message):
        """Record processing logs"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    def get_video_files(self):
        """Get paths for all video files"""
        video_files = []
        
        if os.path.isfile(self.input_path):
            # If input is a single file
            _, ext = os.path.splitext(self.input_path.lower())
            if ext in SUPPORTED_VIDEO_FORMATS:
                video_files.append(self.input_path)
        else:
            # If input is a directory, find all supported video files
            for ext in SUPPORTED_VIDEO_FORMATS:
                files = glob.glob(os.path.join(self.input_path, f"*{ext}"))
                files.extend(glob.glob(os.path.join(self.input_path, f"*{ext.upper()}")))
                video_files.extend(files)
                
        return sorted(video_files)
    
    def process_video(self, video_path):
        """
        Process a single video file and extract frames
        
        Parameters:
            video_path (str): Video file path
            
        Returns:
            int: Number of frames successfully extracted
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(self.output_path, video_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.log(f"Cannot open video: {video_path}")
            return 0
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate sampling interval in frames
        # Convert milliseconds to frames: interval_ms / (1000 / fps)
        frame_interval = max(1, int((self.sampling_interval / 1000) * fps))
        expected_frames = total_frames // frame_interval
        
        self.log(f"Processing video: {video_path}")
        self.log(f"  - Frame rate: {fps:.2f} fps")
        self.log(f"  - Total frames: {total_frames}")
        self.log(f"  - Duration: {duration:.2f} seconds")
        self.log(f"  - Sampling interval: {self.sampling_interval} ms (every {frame_interval} frames)")
        self.log(f"  - Estimated output frames: ~{expected_frames}")
        
        # Extract frames
        frame_count = 0
        saved_count = 0
        
        with tqdm(total=expected_frames, desc=f"Extracting frames from {video_name}") as pbar:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frames according to sampling interval
                if frame_count % frame_interval == 0:
                    # Adjust resolution
                    if self.resolution:
                        frame = cv2.resize(frame, self.resolution)
                    
                    # Save frame
                    output_file = os.path.join(output_dir, f"frame_{saved_count:06d}.{OUTPUT_IMAGE_FORMAT}")
                    cv2.imwrite(output_file, frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
                    
                    saved_count += 1
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        self.log(f"Completed processing {video_path}, saved {saved_count} frames")
        
        return saved_count
    
    def process_all(self):
        """Process all video files and generate dataset"""
        video_files = self.get_video_files()
        
        if not video_files:
            self.log(f"No supported video files found in {self.input_path}")
            return False
        
        self.log(f"Found {len(video_files)} video files")
        self.log(f"Output path: {self.output_path}")
        self.log(f"Output resolution: {self.resolution}")
        self.log(f"Sampling interval: {self.sampling_interval} ms")
        
        total_frames = 0
        
        for video_file in video_files:
            frames = self.process_video(video_file)
            total_frames += frames
        
        self.log(f"Processing complete! Generated {total_frames} image frames")
        
        # Create metadata file
        metadata_file = os.path.join(self.output_path, "metadata.txt")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processed video files: {len(video_files)}\n")
            f.write(f"Total frames: {total_frames}\n")
            f.write(f"Resolution: {self.resolution}\n")
            f.write(f"Sampling interval: {self.sampling_interval} ms\n")
            f.write("\n--- Video List ---\n")
            for video in video_files:
                f.write(f" - {os.path.basename(video)}\n")
        
        return True 