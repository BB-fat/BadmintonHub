# Badminton Training Data Generation Module

This module is used to convert video files into image frame datasets for training deep learning models. It can process a single video file or a directory containing multiple videos, and extract frames at a configurable frequency.

## Features

- Support for multiple video formats (.mp4, .avi, .mov, .mkv)
- Configurable output image resolution
- Configurable sampling rate (frames extracted per second)
- Generate detailed processing logs and metadata

## Installing Dependencies

Using uv to install dependencies:

```bash
cd data_generation
uv pip install -r requirements.txt
```

Or using traditional pip:

```bash
cd data_generation
pip install -r requirements.txt
```

## Usage

### Command Line Usage

```bash
python main.py /path/to/videos -o /path/to/output
```

Parameter description:
- `input`: Input video file or directory containing videos (required)
- `-o, --output`: Output directory path (optional, default: ./dataset)
- `--width`: Output image width (optional, default: 640)
- `--height`: Output image height (optional, default: 480)
- `-r, --rate`: Sampling rate, frames per second (optional, default: 5)

### Using as a Module

```python
from video_to_frames import VideoToFrames

# Create processor instance
processor = VideoToFrames(
    input_path="/path/to/videos",
    output_path="/path/to/output",
    resolution=(640, 480),
    sampling_rate=5
)

# Process all videos
processor.process_all()
```

## Output Structure

```
output_directory/
├── video1/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── video2/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── metadata.txt
└── generation_log_YYYYMMDD_HHMMSS.txt
```

- Each video has its own subdirectory in the output directory
- `metadata.txt` contains detailed information about the dataset
- `generation_log_YYYYMMDD_HHMMSS.txt` contains detailed logs of the processing 