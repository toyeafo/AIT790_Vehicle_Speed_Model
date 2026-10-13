"""
Optical Flow Computation

Reads video frames, computes dense optical flow between consecutive frames,
and saves flow visualization images suitable for CNN input.

We use the standard Farneback dense optical flow algorithm and visualize
flow as HSV -> RGB images (hue = direction, saturation/value = magnitude).
"""

import cv2
import os
import sys
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def make_flow_visualization(flow_uv: np.ndarray) -> np.ndarray:
    """
    Convert 2-channel optical flow (u, v) to an RGB visualization image.

    Args:
        flow_uv: Array of shape (H, W, 2) with (u, v) displacements.

    Returns:
        uint8 RGB image of shape (H, W, 3).
    """
    u = flow_uv[..., 0]
    v = flow_uv[..., 1]

    mag, ang = cv2.cartToPolar(u, v)

    # Create HSV image
    hsv = np.zeros((flow_uv.shape[0], flow_uv.shape[1], 3), dtype=np.uint8)
    # Hue encodes direction (0-180 in OpenCV)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    # Saturation encodes magnitude (clipped to 255)
    mag_clipped = np.clip(mag * 20, 0, 255).astype(np.uint8)
    hsv[..., 1] = mag_clipped
    # Value channel also encodes magnitude for visibility
    hsv[..., 2] = 255

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


def compute_optical_flow(
    video_path: str,
    output_dir: str,
    target_size: tuple = (224, 224),
    max_frames: int = None,
) -> int:
    """
    Compute optical flow between consecutive frames of a video and save
    visualization images.

    Args:
        video_path: Path to input video.
        output_dir: Directory to save flow images.
        target_size: (width, height) to resize images to.
        max_frames: Maximum number of *flow frames* to generate.

    Returns:
        Number of flow images saved.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing: {video_path.name}")
    print(f"  Total frames: {total_frames}, FPS: {fps:.2f}")
    print(f"  Target size: {target_size}")
    print(f"  Output: {output_dir}")
    print("-" * 40)

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, target_size)

    flow_count = 0
    pbar = tqdm(total=(max_frames if max_frames else total_frames - 1), desc="Flow frames")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        if max_frames is not None and flow_count >= max_frames:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, target_size)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        flow_vis = make_flow_visualization(flow)
        filename = f"flow_{flow_count:05d}.png"
        cv2.imwrite(str(output_dir / filename), flow_vis)

        prev_gray = curr_gray
        flow_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"Done! Saved {flow_count} flow images to {output_dir}")
    return flow_count


def main():
    parser = argparse.ArgumentParser(description="Compute dense optical flow images from a video.")
    parser.add_argument("--video", type=str, default="data/train.mp4", help="Input video path.")
    parser.add_argument("--output", type=str, default="data/flow", help="Output directory.")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224], metavar=("W", "H"), help="Resize dimensions (width height).")
    parser.add_argument("--max-frames", type=int, default=None, help="Max flow frames to generate.")

    args = parser.parse_args()
    compute_optical_flow(
        video_path=args.video,
        output_dir=args.output,
        target_size=tuple(args.size),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
