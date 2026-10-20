# AIT790 Vehicle Speed Model — Project Summary

## Team Members
- Samad Afolabi
- Martina Kuzenski
- Iyanuoluwa Joanna Ahmed

---

## Goal
Build and train a machine learning model to estimate vehicle speed in real-time using only dashcam video.

**Input:** Dashcam video (640×480, 20 fps)
**Output:** Speed estimate in mph

---

## Dataset
- **Source:** Comma.ai Speed Challenge
- **Training video:** `train.mp4` — 20,400 frames (~17 minutes)
- **Test video:** `test.mp4` — 10,798 frames (~9 minutes)
- **Labels:** `train.txt` — 20,400 speed values (one per frame)

---

## Pipeline Architecture

```
Raw Video → Optical Flow → CNN Regression → Post-Processing → Speed (mph)
```

| Component | What It Does |
|---|---|
| **Optical Flow** (Farneback) | Computes motion between consecutive frames, visualizes as HSV color image |
| **CNN** (434K params) | 5 convolutional blocks + global average pooling + 3 fully connected layers |
| **Post-Processing** | Stop threshold (≤2 mph → 0) + 5-frame rolling average smoother |

---

## Results

### Training
- **Framework:** PyTorch 2.14 (CUDA)
- **Train/Val split:** 80/20 temporal (16,320 / 4,079 frames)
- **Best epoch:** 16 (early stopped at epoch 21)
- **Training time:** ~12 minutes on GPU

### Validation Performance (Best Model)
| Metric | Value |
|---|---|
| **MAE** | **1.56 mph** |
| **RMSE** | 2.06 mph |
| **MSE** | 4.25 |

### Per-Speed-Range Accuracy
| Range | Samples | MAE |
|---|---|---|
| 0–5 mph | 2,330 | **1.17 mph** |
| 5–10 mph | 1,011 | 1.94 mph |
| 10–15 mph | 738 | 2.30 mph |

### Inference
- **Speed:** <30 ms per frame (real-time at 20+ fps)
- **Test video predictions:** 10,797 speed values saved to CSV

---

## File Structure

```
.
├── data/
│   ├── train.mp4              # Training video
│   ├── test.mp4               # Test video
│   ├── train.txt              # Speed labels
│   ├── flow/                  # Train optical flow images (20,399)
│   └── flow_test/             # Test optical flow images (10,797)
├── src/
│   ├── compute_optical_flow.py    # Flow computation
│   ├── dataset.py                 # PyTorch Dataset/DataLoader
│   ├── model.py                   # CNN architecture
│   ├── train.py                   # Training script
│   ├── inference.py               # Real-time inference
│   ├── utils.py                   # Post-processing & metrics
│   ├── prepare_data.py            # Data prep orchestrator
│   ├── compare_postprocessing.py  # Raw vs. smoothed comparison
│   └── evaluate.py                # Final evaluation
├── outputs/
│   ├── best_model.pth             # Trained weights
│   ├── history.json               # Training history
│   ├── training_curves.png
│   ├── val_predictions.png
│   ├── test_inference_plot.png
│   ├── postprocessing_comparison.png
│   ├── evaluation_plots.png
│   ├── final_metrics.json
│   └── test_predictions.csv
├── project_explanation.md              # Full project explanation with analogies
├── task4_training_explanation.md       # Task 4 explanation
├── task5_inference_postprocessing_explanation.md  # Task 5 explanation
├── task6_evaluation_explanation.md     # Task 6 explanation
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Setup environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data (compute optical flow)
```bash
python src/prepare_data.py
```

### 3. Train the model
```bash
python src/train.py --epochs 30 --batch-size 32
```

### 4. Run inference on test video
```bash
python src/inference.py --video data/test.mp4 --model outputs/best_model.pth
```

### 5. Evaluate the model
```bash
python src/evaluate.py
```

---

## Key Design Decisions

1. **Optical flow instead of raw frames** — Raw pixels don't encode speed. Motion images do.
2. **Temporal train/val split** — Prevents the model from "cheating" by memorizing road sequences.
3. **Lightweight CNN** — 434K parameters run fast enough for real-time inference.
4. **Stop threshold** — Eliminates phantom 1–2 mph readings when the car is actually stopped.
5. **Rolling average smoother** — Replaces jittery per-frame predictions with a stable speedometer feel.

---

## Future Improvements

| Improvement | Expected Gain |
|---|---|
| Temporal model (LSTM/GRU) | MAE → ~1.2 mph |
| Pretrained ResNet18 backbone | MAE → ~0.9–1.1 mph |
| Multi-dataset training | Better generalization |
| Higher resolution input | Better flow detail |
| Weather/night augmentation | Robustness |

---

## Conclusion

We successfully built an end-to-end vehicle speed estimation system from scratch using optical flow and a lightweight CNN. The model achieves **1.56 mph mean absolute error** on unseen validation data, runs in real time, and produces human-readable speed traces. The system demonstrates that speed can be inferred from visual motion alone — no GPS, no OBD-II, no speed sensor required.
