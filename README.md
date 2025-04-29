# BadmintonHub 🏸

Badminton video analysis system for amateur training, featuring shuttlecock tracking and motion data conversion.

## Key Features

- Badminton trajectory tracking

## Modules

### Training Data Generation

* Convert multiple video files into keyframe image datasets for training
* Customizable output image resolution and video sampling frequency
* Support multiple video formats (mp4, avi, mov, mkv)
* Generate detailed processing logs and metadata

Usage example:

```bash
cd data_generation
python main.py /path/to/videos -o /path/to/output -w 640 -h 480 -r 5
```

See [data_generation/README.md](data_generation/README.md) for more details.

## License

MIT License
