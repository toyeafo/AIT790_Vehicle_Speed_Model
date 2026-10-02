"""
Video to Frames Extractor

Extracts individual frames from a video file and saves them as images.
Uses OpenCV for efficient frame extraction.
"""

import cv2
import os
import sys
import argparse
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str,
    image_format: str = "jpg",
    naming_pattern: str = "frame_{:04d}.{}",
    max_frames: int = None,
) -> int:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        image_format: Image file format (jpg, png, etc.).
        naming_pattern: Frame naming pattern with placeholder for index and extension.
        max_frames: Maximum number of frames to extract (None for all).

    Returns:
        Number of frames extracted.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video capture
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output directory: {output_dir}")
    print("-" * 40)

    frame_count = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stop if max_frames reached
        if max_frames is not None and extracted >= max_frames:
            break

        # Build filename and save
        filename = naming_pattern.format(frame_count, image_format)
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), frame)

        extracted += 1

        # Progress report every 100 frames
        if extracted % 100 == 0:
            print(f"  Extracted {extracted} frames...")

        frame_count += 1

    cap.release()
    print(f"Done! Extracted {extracted} frames to {output_dir}")
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file into individual images."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/train.mp4",
        help="Path to the input video file (default: data/train.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/frames",
        help="Directory to save extracted frames (default: data/frames)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "png", "bmp", "tiff"],
        help="Output image format (default: jpg)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: all)",
    )

    args = parser.parse_args()

    try:
        extract_frames(
            video_path=args.video,
            output_dir=args.output,
            image_format=args.format,
            max_frames=args.max_frames,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
