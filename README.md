# Vehicle Speed Estimation from Dashcam Video

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **AIT790 Course Project** — Build and train a machine learning model to estimate vehicle speed in real-time using only dashcam video.

**Team:** Samad Afolabi, Martina Kuzenski, Iyanuoluwa Joanna Ahmed

---

## Quick Start

The entire project — data preparation, model architecture, training, evaluation, and inference — is contained in a single Jupyter notebook:

📓 **[`Vehicle_Speed_Estimation.ipynb`](Vehicle_Speed_Estimation.ipynb)**

Open the notebook in Jupyter or Google Colab and run the cells top-to-bottom. The notebook includes **all code and results** (training curves, evaluation metrics, diagnostic plots, and inference outputs) embedded directly in the cells.

---

## Results Preview

| Metric | Value |
|---|---|
| **MAE** | **1.56 mph** |
| **RMSE** | 2.06 mph |
| **Validation Samples** | 4,079 |
| **Model Size** | ~435K parameters |

The notebook contains:
- ✅ Full training history (21 epochs, early stopping at epoch 16)
- ✅ Training curves (Loss, MAE, RMSE)
- ✅ Four-panel evaluation diagnostics (scatter, error histogram, error vs. speed, cumulative distribution)
- ✅ Raw vs. post-processed comparison
- ✅ Test video inference speed trace (10,797 predictions)

---

## Project Structure

```
.
├── Vehicle_Speed_Estimation.ipynb    ← Main deliverable (code + results)
├── data/
│   ├── train.mp4                   ← Download separately (~125 MB)
│   ├── test.mp4                    ← Download separately (~64 MB)
│   └── train.txt                   ← Speed labels (20,400 values)
├── project_explanation.md          ← Full project explanation with analogies
├── task4_training_explanation.md   ← Training deep dive
├── task5_inference_postprocessing_explanation.md  ← Inference deep dive
├── task6_evaluation_explanation.md ← Evaluation deep dive
├── PROJECT_SUMMARY.md              ← One-page project summary
├── requirements.txt
├── .gitignore
└── README.md                       ← This file
```

> **Note:** `data/train.mp4` and `data/test.mp4` are **not** included in the repository due to GitHub file size limits. Download instructions are in the notebook.

---

## How It Works (In Brief)

1. **Optical Flow** — Compute dense motion between consecutive frames. Bright colors = fast motion.
2. **CNN Regression** — A lightweight 435K-parameter network maps motion images to speed values.
3. **Post-Processing** — Stop threshold (≤2 mph → 0) + 5-frame rolling average smoother.

For full details and analogies (the "Desk Fan," the "Pattern Detective," the "Flashcard Student," etc.), see [`project_explanation.md`](project_explanation.md).

---

## Requirements

```
PyTorch >= 2.0.0
OpenCV >= 4.5.0
NumPy
Matplotlib
tqdm
scikit-learn
Pillow
```

Install via:

```bash
pip install -r requirements.txt
```

---

## Dataset

- **Source:** [Comma.ai Speed Challenge](https://github.com/commaai/speedchallenge)
- **Training video:** ~17 minutes, 20,400 frames
- **Test video:** ~9 minutes, 10,798 frames
- **Labels:** Per-frame speed values in `data/train.txt`

---

## Report Cards

- **Task 4 (Training):** [`task4_training_explanation.md`](task4_training_explanation.md)
- **Task 5 (Inference & Post-Processing):** [`task5_inference_postprocessing_explanation.md`](task5_inference_postprocessing_explanation.md)
- **Task 6 (Evaluation & Wrap-Up):** [`task6_evaluation_explanation.md`](task6_evaluation_explanation.md)

## Presentation Prep

- **Presentation Feedback:** [`presentation_feedback.md`](presentation_feedback.md) — Suggested changes for your slides
- **Presentation Prep Guide:** [`presentation_prep.md`](presentation_prep.md) — Simple explanations, analogies, Q&A prep

---

## License

See [LICENSE](LICENSE) for details.
