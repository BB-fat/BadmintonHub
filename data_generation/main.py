import argparse
from video_to_frames import VideoToFrames
from config import DEFAULT_RESOLUTION, DEFAULT_SAMPLING_INTERVAL, OUTPUT_DATASET_DIR

def main():
    parser = argparse.ArgumentParser(description="Convert videos to image frame datasets")
    
    parser.add_argument("input", help="Input video file or directory containing videos")
    parser.add_argument("-o", "--output", help=f"Output directory path (default: ./{OUTPUT_DATASET_DIR})", default=OUTPUT_DATASET_DIR)
    parser.add_argument("--width", type=int, default=DEFAULT_RESOLUTION[0], help=f"Output image width (default: {DEFAULT_RESOLUTION[0]})")
    parser.add_argument("--height", type=int, default=DEFAULT_RESOLUTION[1], help=f"Output image height (default: {DEFAULT_RESOLUTION[1]})")
    parser.add_argument("-i", "--interval", type=int, default=DEFAULT_SAMPLING_INTERVAL, help=f"Sampling interval (milliseconds between frames, default: {DEFAULT_SAMPLING_INTERVAL}ms)")
    
    args = parser.parse_args()
    
    # Set resolution
    resolution = (args.width, args.height)
    
    # Create processor instance
    processor = VideoToFrames(
        input_path=args.input,
        output_path=args.output,
        resolution=resolution,
        sampling_interval=args.interval
    )
    
    # Process all videos
    success = processor.process_all()
    
    if success:
        print("🎉 Dataset generation completed!")
    else:
        print("❌ Dataset generation failed, please check the logs")

if __name__ == "__main__":
    main() 