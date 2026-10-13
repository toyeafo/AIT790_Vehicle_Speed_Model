"""
Compare raw model predictions vs. post-processed (smoothed + thresholded) predictions.

Runs inference on a video subset and saves a side-by-side plot.
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from model import SpeedEstimatorCNN
from compute_optical_flow import make_flow_visualization
from utils import RollingAverageFilter, apply_speed_threshold


class RawPredictor:
    """Predictor with NO post-processing — returns raw model outputs."""
    def __init__(self, model_path, device="cuda", input_size=(224, 224)):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.model = SpeedEstimatorCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

    def preprocess(self, frame):
        frame = cv2.resize(frame, self.input_size)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return (tensor - self.mean) / self.std

    @torch.no_grad()
    def predict(self, flow_vis):
        return self.model(self.preprocess(flow_vis)).item()


class SmoothedPredictor:
    """Predictor WITH post-processing — threshold + rolling average."""
    def __init__(self, model_path, device="cuda", input_size=(224, 224), smooth=5, threshold=2.0):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.threshold = threshold
        self.model = SpeedEstimatorCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self.smooth_filter = RollingAverageFilter(window_size=smooth)

    def preprocess(self, frame):
        frame = cv2.resize(frame, self.input_size)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return (tensor - self.mean) / self.std

    @torch.no_grad()
    def predict(self, flow_vis):
        raw = self.model(self.preprocess(flow_vis)).item()
        speed = apply_speed_threshold(raw, self.threshold)
        return self.smooth_filter.update(speed)

    def reset(self):
        self.smooth_filter.reset()


def run_comparison(video_path, model_path, max_frames=2000, save_path="outputs/postprocessing_comparison.png"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Empty video")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (224, 224))

    raw_pred = RawPredictor(model_path)
    smooth_pred = SmoothedPredictor(model_path, smooth=5, threshold=2.0)

    raw_speeds = []
    smooth_speeds = []

    pbar = tqdm(total=max_frames, desc="Comparing")
    for _ in range(max_frames):
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, (224, 224))

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow_vis = make_flow_visualization(flow)

        raw_speeds.append(raw_pred.predict(flow_vis))
        smooth_speeds.append(smooth_pred.predict(flow_vis))

        prev_gray = curr_gray
        pbar.update(1)

    pbar.close()
    cap.release()

    # Plot side-by-side
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(raw_speeds, color="tab:orange", linewidth=0.8)
    axes[0].set_title("Raw Model Output (No Post-Processing)")
    axes[0].set_ylabel("Speed (mph)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="Stop threshold")
    axes[0].legend()

    axes[1].plot(smooth_speeds, color="tab:blue", linewidth=0.8)
    axes[1].set_title("Post-Processed Output (Stop Threshold + 5-Frame Rolling Average)")
    axes[1].set_ylabel("Speed (mph)")
    axes[1].set_xlabel("Frame Index")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")

    # Compute stats
    raw_arr = np.array(raw_speeds)
    smooth_arr = np.array(smooth_speeds)
    print(f"\nStatistics over {len(raw_speeds)} frames:")
    print(f"  Raw    — min: {raw_arr.min():.2f} | max: {raw_arr.max():.2f} | std: {raw_arr.std():.2f}")
    print(f"  Smooth — min: {smooth_arr.min():.2f} | max: {smooth_arr.max():.2f} | std: {smooth_arr.std():.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/test.mp4")
    parser.add_argument("--model", type=str, default="outputs/best_model.pth")
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--save", type=str, default="outputs/postprocessing_comparison.png")
    args = parser.parse_args()

    run_comparison(args.video, args.model, args.max_frames, args.save)
