"""
Inference pipeline for real-time (or batch) vehicle speed estimation.

Given a video file, extracts frames, computes optical flow on the fly,
runs the trained model, applies post-processing, and outputs speed estimates.
"""

import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import SpeedEstimatorCNN
from compute_optical_flow import make_flow_visualization
from utils import RollingAverageFilter, apply_speed_threshold


class SpeedPredictor:
    """
    End-to-end speed predictor from a video stream or file.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        input_size: tuple = (224, 224),
        smooth_window: int = 5,
        stop_threshold: float = 2.0,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.stop_threshold = stop_threshold

        # Load model
        self.model = SpeedEstimatorCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing
        self.mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

        # Smoothing
        self.smooth_filter = RollingAverageFilter(window_size=smooth_window)

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Resize and normalize a BGR frame to model input tensor."""
        frame = cv2.resize(frame, self.input_size)
        # Convert BGR -> RGB, normalize to [0, 1], then to tensor (C, H, W)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # Normalize with dataset statistics
        tensor = (tensor - self.mean) / self.std
        return tensor

    def postprocess(self, raw_speed: float) -> float:
        """Apply threshold and rolling average smoothing."""
        speed = apply_speed_threshold(raw_speed, threshold=self.stop_threshold)
        speed = self.smooth_filter.update(speed)
        return speed

    @torch.no_grad()
    def predict_frame(self, frame: np.ndarray) -> float:
        """Predict speed from a single BGR frame (resized internally)."""
        tensor = self.preprocess(frame)
        pred = self.model(tensor)
        speed = pred.item()
        return self.postprocess(speed)

    def predict_from_video(
        self,
        video_path: str,
        output_csv: str = None,
        save_plot: str = None,
        max_frames: int = None,
    ):
        """
        Run inference on a video by computing optical flow between consecutive frames.

        Args:
            video_path: Path to video file.
            output_csv: Optional path to save predicted speeds as CSV.
            save_plot: Optional path to save a plot of predicted speeds.
            max_frames: Max frames to process.

        Returns:
            List of predicted speed values.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Inference on {Path(video_path).name}: {total_frames} frames @ {fps:.1f} fps")

        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Video is empty.")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, self.input_size)

        speeds = []
        frame_idx = 0
        self.smooth_filter.reset()

        pbar = tqdm(total=(max_frames if max_frames else total_frames - 1), desc="Inference")

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.resize(curr_gray, self.input_size)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

            flow_vis = make_flow_visualization(flow)
            tensor = self.preprocess(flow_vis)
            pred = self.model(tensor).item()
            speed = self.postprocess(pred)
            speeds.append(speed)

            prev_gray = curr_gray
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        print(f"Predicted {len(speeds)} speed values.")

        # Save results
        if output_csv:
            with open(output_csv, "w") as f:
                f.write("frame,speed\n")
                for i, s in enumerate(speeds, start=1):
                    f.write(f"{i},{s:.6f}\n")
            print(f"Saved predictions to {output_csv}")

        if save_plot and speeds:
            plt.figure(figsize=(12, 4))
            plt.plot(speeds, label="Predicted Speed")
            plt.xlabel("Frame Index")
            plt.ylabel("Speed")
            plt.title(f"Predicted Speeds — {Path(video_path).name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_plot)
            plt.close()
            print(f"Saved plot to {save_plot}")

        return speeds


def main():
    parser = argparse.ArgumentParser(description="Run speed estimation inference on a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--model", type=str, default="outputs/best_model.pth", help="Path to trained model weights.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV output path.")
    parser.add_argument("--output-plot", type=str, default="outputs/inference_plot.png", help="Path to save prediction plot.")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    parser.add_argument("--smooth", type=int, default=5, help="Rolling average window size.")
    parser.add_argument("--threshold", type=float, default=2.0, help="Zero-speed threshold.")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process.")
    args = parser.parse_args()

    predictor = SpeedPredictor(
        model_path=args.model,
        device=args.device,
        smooth_window=args.smooth,
        stop_threshold=args.threshold,
    )

    predictor.predict_from_video(
        video_path=args.video,
        output_csv=args.output_csv,
        save_plot=args.output_plot,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
