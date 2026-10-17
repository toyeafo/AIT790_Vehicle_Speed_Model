# Vehicle Speed Estimation from Dashcam Video

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-optional-green.svg)](https://developer.nvidia.com/cuda-downloads)

> **AIT790 Course Project 01** — Build and train a machine learning model to estimate vehicle speed in real-time using only dashcam video.

**Team:** Samad Afolabi, Iyanuoluwa Joanna Ahmed, Martina Kuzenski

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Quick Start](#quick-start)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
- [Evaluation](#evaluation)
- [Future Improvements](#future-improvements)

---

## Overview

**The Problem:** You are in a car with a dashcam. The camera sees the road ahead, the lane markings, trees, buildings, and other cars. However, there is no speedometer on screen. Can we teach a computer to look at the video and say, *"You are driving 28.3 mph right now"*?

**The Challenge:** A single photograph of the road doesn't tell you speed. A picture of a highway at 30 mph looks almost identical to one at 60 mph. The *difference between two consecutive pictures* is what contains the speed information.

**Our Solution:** We use **dense optical flow** to convert motion between consecutive frames into color-coded images (brightness = speed), then train a lightweight CNN to map these motion images to speed values. Post-processing rules (stop threshold + rolling average smoothing) make the output stable and human-readable.

---

## Architecture

```
Raw Video (640×480, 20 fps)
    ↓
Dense Optical Flow (Farneback)
    ↓
Motion Visualization Image (224×224, HSV color-coded)
    ↓
CNN Regressor (5 conv blocks, 435K params)
    ↓
Raw Speed Prediction (mph)
    ↓
Post-Processing
    ├─ Stop Threshold (≤2 mph → 0)
    └─ Rolling Average (5-frame smoothing)
    ↓
Final Speed Estimate (mph)
```

| Component | Description |
|---|---|
| **Dataset** | Comma.ai Speed Challenge — 20,400 frames with per-frame speed labels |
| **Optical Flow** | OpenCV Farneback dense optical flow, visualized as HSV→RGB |
| **Model** | Custom lightweight CNN with BatchNorm, Dropout, Global Average Pooling |
| **Loss** | Mean Squared Error (MSE) |
| **Optimizer** | Adam with ReduceLROnPlateau scheduler |
| **Validation** | Temporal 80/20 split (first 80% train, last 20% val) |
| **Post-Processing** | Hard stop threshold + rolling average filter |

---

## Results

### Validation Performance (Best Model — Epoch 16)

| Metric | Value |
|---|---|
| **MAE** | **1.56 mph** |
| **RMSE** | 2.06 mph |
| **MSE** | 4.25 |
| **Inference Speed** | <30 ms/frame (real-time at 20+ fps) |
| **Model Parameters** | ~435,000 |

### Per-Speed-Range Breakdown

| Speed Range | Samples | MAE | Notes |
|---|---|---|---|
| 0–5 mph | 2,330 | **1.17 mph** | Excellent stop detection |
| 5–10 mph | 1,011 | 1.94 mph | Good city driving accuracy |
| 10–15 mph | 738 | 2.30 mph | Higher speeds = more challenge |

### Benchmark Comparison

| System | MAE |
|---|---|
| Random guessing | ~10 mph |
| Always guess average | ~8 mph |
| **Our model (scratch CNN + optical flow)** | **1.56 mph** |
| ResNet18 + ImageNet pretraining | ~0.8–1.2 mph |
| Human reading analog gauge | ~1 mph |

For a lightweight 435K-parameter model trained from scratch (no pretraining), **1.56 mph** is a strong result. It is accurate enough for practical dashcam applications and runs in real time on modest GPU hardware.

---

## Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd AIT790_Vehicle_Speed_Model
```

### 2. Download the dataset

The video files are not included in the repository due to size (~190 MB total). Download them from the Comma.ai Speed Challenge repository:

```bash
# Create data directory
mkdir -p data

# Download training video + labels
curl -L -o data/train.mp4 "https://media.githubusercontent.com/media/commaai/speedchallenge/master/data/train.mp4"
curl -L -o data/test.mp4 "https://media.githubusercontent.com/media/commaai/speedchallenge/master/data/test.mp4"

# Labels are already included in the repo at data/train.txt
```

Verify downloads:

```bash
ls -lh data/*.mp4 data/train.txt
```

### 3. Set up the environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Prepare the data (compute optical flow)

```bash
python src/prepare_data.py
```

This generates:
- `data/flow/` — 20,399 optical flow images from train.mp4
- `data/flow_test/` — 10,797 optical flow images from test.mp4

*This step takes ~5–10 minutes depending on your CPU.*

### 5. Train the model

```bash
python src/train.py --epochs 30 --batch-size 32 --lr 1e-3
```

Outputs saved to `outputs/`:
- `best_model.pth` — Best checkpoint (lowest validation loss)
- `training_curves.png` — Loss / MAE / RMSE over epochs
- `val_predictions.png` — Predicted vs. actual speeds on validation set

### 6. Evaluate the model

```bash
python src/evaluate.py
```

Produces detailed metrics and four-panel diagnostic plots (`outputs/evaluation_plots.png`).

### 7. Run inference on a new video

```bash
python src/inference.py --video data/test.mp4 --model outputs/best_model.pth --output-csv outputs/predictions.csv
```

### Jupyter Notebook

For an interactive, all-in-one walkthrough, open:

```bash
jupyter notebook Vehicle_Speed_Estimation.ipynb
```

---

## File Structure

```
.
├── data/
│   ├── train.mp4              # Training video (download separately)
│   ├── test.mp4               # Test video (download separately)
│   ├── train.txt              # Per-frame speed labels
│   ├── flow/                  # Generated: optical flow images (train)
│   └── flow_test/             # Generated: optical flow images (test)
├── src/
│   ├── compute_optical_flow.py      # Optical flow computation
│   ├── dataset.py                   # PyTorch Dataset & DataLoader
│   ├── model.py                     # CNN architecture
│   ├── train.py                     # Training script
│   ├── inference.py                 # Real-time inference pipeline
│   ├── utils.py                     # Post-processing & metrics
│   ├── prepare_data.py              # Data preparation orchestrator
│   ├── compare_postprocessing.py    # Raw vs. smoothed comparison
│   └── evaluate.py                  # Final evaluation & diagnostics
├── outputs/                         # Generated during training/inference
│   ├── best_model.pth
│   ├── training_curves.png
│   ├── evaluation_plots.png
│   └── ...
├── Vehicle_Speed_Estimation.ipynb   # Interactive notebook (all-in-one)
├── project_explanation.md          # Full project explanation with analogies
├── task4_training_explanation.md   # Task 4: Training deep dive
├── task5_inference_postprocessing_explanation.md  # Task 5: Inference deep dive
├── task6_evaluation_explanation.md # Task 6: Evaluation deep dive
├── PROJECT_SUMMARY.md              # One-page project summary
├── requirements.txt
├── .gitignore
└── README.md                        # This file
```

---

## How It Works

### 1. Optical Flow: The "Desk Fan" Analogy

If you take one photo of fan blades, you have no idea how fast they're spinning. But if you take **two photos a split-second apart** and compare them, you see exactly how far each blade moved. The faster the spin, the farther each blade travels between frames.

**Optical Flow** does this for every pixel. We draw an arrow for each pixel showing its motion, then color-code into an image — bright colors mean fast motion, dark means slow. Now the neural network has a picture where *brightness = speed*.

### 2. The CNN: The "Pattern Detective"

- **Layer 1** looks at 3×3 patches: *"I see motion pointing left here."*
- **Layer 2** combines patches: *"That left motion is a lane marker sliding past."*
- **Layer 3** combines those: *"Lane markers sliding fast = we are going fast."*
- **Final layer** outputs one number: **speed in mph.**

### 3. Training: The "Flashcard Student"

- **Front of card:** An optical flow image
- **Back of card:** The correct speed

The model guesses, checks the answer, and gradually improves. MSE loss punishes big mistakes severely (80 mph off is 2704× worse than 2 mph off), keeping the model honest.

### 4. Post-Processing: The "Thermostat"

- **Stop Threshold:** If the model says "1.5 mph," override to **0 mph**. Red lights, shadows, and camera shake create phantom motion. The thermostat's "dead zone" says: *"If it's basically zero, it IS zero."*
- **Rolling Average:** A real speedometer doesn't flicker 30→35→28→31. It glides. We average the last 5 frames, like a shower faucet's pressure regulator smoothing temperature changes.

---

## Evaluation

The model was trained on the first 80% of the training video (16,320 frames) and validated on the last 20% (4,079 frames). The temporal split ensures the model generalizes to *unseen driving conditions*, not just memorized road sequences.

### Diagnostic Plots

After running `src/evaluate.py`, check `outputs/evaluation_plots.png` for four insights:

1. **Predicted vs. Actual Scatter** — Most dots hug the perfect diagonal. A horizontal band near 2.5 mph shows the "phantom creeping" problem (fixed by stop threshold).
2. **Error Histogram** — Bell curve centered on 0. Most errors are 0–2 mph.
3. **Error vs. Speed** — Accuracy is best at low speeds; fuzzier above 10 mph.
4. **Cumulative Error** — ~80% of predictions are within 3 mph of the truth.

---

## Future Improvements

| Upgrade | Expected Impact | Complexity |
|---|---|---|
| **Temporal model (LSTM/GRU)** | MAE → ~1.2 mph | Medium |
| **Pretrained ResNet18 backbone** | MAE → ~0.9–1.1 mph | Low |
| **Multi-dataset training** | Better generalization | High |
| **Larger input resolution** | Better flow detail | Low |
| **Weather/night augmentation** | Robustness | Medium |
| **Multi-frame optical flow** | Capture acceleration | Medium |

The lowest-hanging fruit is a **pretrained backbone**. Swapping our custom 5-layer CNN for a ResNet18 that already knows how to extract image features would likely cut MAE below 1.1 mph with minimal code changes. The tradeoff is model size (11M parameters vs. 435K).

---

## Acknowledgments

- **Dataset:** [Comma.ai Speed Challenge](https://github.com/commaai/speedchallenge) — open-source driving data with aligned speed labels
- **Framework:** PyTorch + OpenCV
- **Course:** AIT790 - Research Methodologies (Applied Machine Learning) 

---

## License

See [LICENSE](LICENSE) for details.
