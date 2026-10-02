#!/usr/bin/env bash
# Convenience script to extract frames from data/train.mp4

set -e

# Use the virtual environment Python if available
PYTHON="${PWD}/venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "Extracting frames from data/train.mp4 ..."
$PYTHON src/video_to_frames.py --video data/train.mp4 --output data/frames --format jpg

echo "Frame extraction complete!"
