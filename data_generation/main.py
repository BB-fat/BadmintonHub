import argparse
from video_to_frames import VideoToFrames

def main():
    parser = argparse.ArgumentParser(description="Convert videos to image frame datasets")
    
    parser.add_argument("input", help="Input video file or directory containing videos")
    parser.add_argument("-o", "--output", help="Output directory path (default: ./dataset)")
    parser.add_argument("--width", type=int, default=640, help="Output image width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Output image height (default: 480)")
    parser.add_argument("-r", "--rate", type=float, default=5, help="Sampling rate (frames per second, default: 5)")
    
    args = parser.parse_args()
    
    # Set resolution
    resolution = (args.width, args.height)
    
    # Create processor instance
    processor = VideoToFrames(
        input_path=args.input,
        output_path=args.output,
        resolution=resolution,
        sampling_rate=args.rate
    )
    
    # Process all videos
    success = processor.process_all()
    
    if success:
        print("🎉 Dataset generation completed!")
    else:
        print("❌ Dataset generation failed, please check the logs")

if __name__ == "__main__":
    main() 