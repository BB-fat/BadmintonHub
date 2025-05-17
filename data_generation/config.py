"""
Configuration file containing default parameters for video processing and frame extraction
"""

# Default output image resolution
DEFAULT_RESOLUTION = (1920, 1080)  # width x height

# Default sampling interval in milliseconds (time between frames)
DEFAULT_SAMPLING_INTERVAL = 1000  # ms

# Output dataset directory
OUTPUT_DATASET_DIR = "dataset"

# Supported video formats
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

# Output image format
OUTPUT_IMAGE_FORMAT = "jpg"

# Image quality (for JPEG only, 0-100)
IMAGE_QUALITY = 95 