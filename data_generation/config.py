"""
Configuration file containing default parameters for video processing and frame extraction
"""

# Default output image resolution
DEFAULT_RESOLUTION = (640, 480)  # width x height

# Default sampling rate (frames per second)
DEFAULT_SAMPLING_RATE = 5

# Output dataset directory
OUTPUT_DATASET_DIR = "dataset"

# Supported video formats
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

# Output image format
OUTPUT_IMAGE_FORMAT = "jpg"

# Image quality (for JPEG only, 0-100)
IMAGE_QUALITY = 95 