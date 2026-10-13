"""
Orchestrator script for the full data preparation pipeline.

1. Compute optical flow images from train.mp4  -> data/flow/
2. Compute optical flow images from test.mp4   -> data/flow_test/

After this runs, you can train with:
    venv/bin/python src/train.py --flow-dir data/flow

You can run inference on test video with:
    venv/bin/python src/inference.py --video data/test.mp4
"""

import argparse
from pathlib import Path
from compute_optical_flow import compute_optical_flow


def main():
    parser = argparse.ArgumentParser(description="Prepare optical flow data from videos.")
    parser.add_argument("--train-video", type=str, default="data/train.mp4", help="Training video path.")
    parser.add_argument("--test-video", type=str, default="data/test.mp4", help="Test video path.")
    parser.add_argument("--train-flow-dir", type=str, default="data/flow", help="Output dir for train flow images.")
    parser.add_argument("--test-flow-dir", type=str, default="data/flow_test", help="Output dir for test flow images.")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224], metavar=("W", "H"), help="Resize dimensions.")
    parser.add_argument("--max-frames", type=int, default=None, help="Max flow frames to generate (useful for quick tests).")
    args = parser.parse_args()

    print("=" * 50)
    print("STEP 1: Computing optical flow for TRAIN video")
    print("=" * 50)
    compute_optical_flow(
        video_path=args.train_video,
        output_dir=args.train_flow_dir,
        target_size=tuple(args.size),
        max_frames=args.max_frames,
    )

    print()
    print("=" * 50)
    print("STEP 2: Computing optical flow for TEST video")
    print("=" * 50)
    compute_optical_flow(
        video_path=args.test_video,
        output_dir=args.test_flow_dir,
        target_size=tuple(args.size),
        max_frames=args.max_frames,
    )

    print()
    print("=" * 50)
    print("Data preparation complete!")
    print(f"Train flow images: {args.train_flow_dir}")
    print(f"Test  flow images: {args.test_flow_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
