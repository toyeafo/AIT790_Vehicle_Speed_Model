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

## Project Structure

```
.
├── Vehicle_Speed_Estimation.ipynb    ← Main deliverable (code + results)
├── data/
│   ├── train.mp4                   ← Download separately (~125 MB)
│   ├── test.mp4                    ← Download separately (~64 MB)
│   └── train.txt                   ← Speed labels (20,400 values)
├── docs/
│   ├── project_explanation.md          ← Full project explanation with analogies
│   ├── task4_training_explanation.md   ← Training deep dive
│   ├── task5_inference_postprocessing_explanation.md  ← Inference deep dive
│   ├── task6_evaluation_explanation.md ← Evaluation deep dive
│   ├── PROJECT_SUMMARY.md              ← One-page project summary
│   ├── presentation_feedback.md        ← Suggested changes for your slides
│   └── presentation_prep.md            ← Simple explanations, analogies, Q&A prep
├── requirements.txt
├── .gitignore
└── README.md                       ← This file
```

> **Note:** `data/train.mp4` and `data/test.mp4` are **not** included in the repository due to GitHub file size limits. Download instructions are in the notebook.

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

## License

See [LICENSE](LICENSE) for details.
