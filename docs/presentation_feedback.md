# Presentation Feedback & Recommended Changes

## Overall Assessment

The presentation is well-structured and covers the right topics. The methodology diagram and results tables are clear and professional. However, there are **content gaps**, **outdated references**, and **missing visuals** that should be addressed before presenting.

---

## Critical Issues (Fix Before Presenting)

### 1. Empty Abstract Slide
**Current:** The slide just says "Abstract" with no text.\n
**Fix:** Add 2–3 sentences:
> *"We built a neural network that estimates vehicle speed from dashcam video alone — no GPS, no internet required. Using optical flow motion detection and a lightweight CNN, our model predicts speed in real time with an average error of 1.56 mph. This enables speed display even in remote areas with no network coverage."*

Or simply **remove the Abstract slide** — academic presentations often skip abstracts.

---

### 2. File Structure Slide Is Outdated
**Current:** Still lists `src/` directory with `.py` files (`compute_optical_flow.py`, `train.py`, etc.).\n
**Problem:** You restructured the project to be **notebook-only**. The `src/` folder no longer exists in the repo.\n
**Fix:** Update the tree to match the current repository:

```
.
├── Vehicle_Speed_Estimation.ipynb   ← Main deliverable (code + results)
├── data/
│   ├── train.mp4                    # Download separately (~125 MB)
│   ├── test.mp4                     # Download separately (~64 MB)
│   └── train.txt                    # Speed labels
├── project_explanation.md
├── task4_training_explanation.md
├── task5_inference_postprocessing_explanation.md
├── task6_evaluation_explanation.md
├── PROJECT_SUMMARY.md
├── requirements.txt
└── README.md
```

**Talking point:** *"Everything lives in a single Jupyter notebook — you can run it top-to-bottom and reproduce every result."*

---

### 3. Missing Comma2k19 Testing
**Current:** The professor explicitly asked you to test on the [Comma2k19](https://github.com/commaai/comma2k19) dataset. This is not mentioned anywhere in the presentation.\n
**Fix:** Add one of the following:
- **A new slide** titled "Cross-Dataset Testing: Comma2k19" showing your notebook's comma2k19 section
- **A bullet in Future Improvements** mentioning comma2k19 highway testing
- **In the Demo slide**, mention you will show both Speed Challenge and comma2k19 results

**Suggested slide content:**
> *"We extended the notebook to test on Comma2k19 (33+ hours of highway driving with CAN bus logs). This tests whether a city-trained model generalizes to highway speeds (60–80 mph). The notebook includes HEVC handling instructions and a generic `predict_speeds_from_video()` function for any input format."*

---

### 4. Empty DEMO Slide
**Current:** Just says "DEMO" with no content.\n
**Fix:** Add context so the audience knows what to expect:
> *"Live walkthrough of the Jupyter notebook showing: (1) optical flow visualization, (2) training curves, (3) real-time inference on test video, (4) evaluation plots, and (5) comma2k19 cross-dataset testing."*

---

## Medium-Priority Improvements

### 5. Add Visuals to the Results Section
**Current:** The Results slides are text-only tables.\n
**Fix:** Embed these key plots (they are already generated and in your `outputs/` folder):
- **Training curves** (`training_curves.png`) — shows the model learning over 21 epochs
- **Predicted vs. Actual scatter** (from `evaluation_plots.png`, top-left panel) — the "hug the diagonal" visual
- **Raw vs. Post-Processed comparison** (`postprocessing_comparison.png`) — dramatic before/after

**Why:** A picture of the model's output is worth more than a table of numbers. The raw vs. smoothed comparison especially tells a great story.

---

### 6. Add an Optical Flow Visualization Slide
**Current:** The Methodology slide has a diagram, but no actual image of what optical flow looks like.\n
**Fix:** Insert a screenshot from your notebook showing the HSV-colored optical flow image (cell 9). Caption it:
> *"Optical flow between two consecutive frames. Brightness = speed. Hue = direction. This is what the neural network 'sees' instead of a normal photograph."*

This is the single most important visual for explaining the project — it answers *"How does a photo know speed?"*

---

### 7. Evaluation Slide Is Too Sparse
**Current:** Lists four plot names but doesn't show them or explain what they reveal.\n
**Fix:** Either:
- Show the 4-panel `evaluation_plots.png` and briefly describe each quadrant
- Or replace with a single summary: *"Four diagnostic plots confirm the model tracks speed accurately: most predictions fall within 2 mph of truth, errors are Gaussian-distributed, and accuracy is best at low speeds (1.17 mph MAE for 0–5 mph range)."*

---

### 8. Related Work Section: Date Typo
**Related Work 2** says *"Aug 17, 2026"* — this is a future date. Either:\n- It's a typo in the paper (check and correct if needed)\n- Or leave it as-is but be prepared for a professor or classmate to ask *"How did you cite a paper from the future?"*

**Also consider:** The Related Work slides are very text-heavy. For a 10–15 minute presentation, you could condense each to **3 bullets** instead of paragraph-style explanations.

---

## Minor Polish Items

### 9. Grammar / Wording Fixes

| Current | Fix |
|---|---|
| "WIFI" | "Wi-Fi" |
| "internet connection" | "an Internet connection" |
| "no network zones" | "areas with no network coverage" |
| "if speed up or captured" | "if sped up or captured" |
| "car's GPS coordinates" | "car GPS coordinates" (remove apostrophe for consistency) |
| "frame by frame" | "frame-by-frame" |
| "frame to frame" | "frame-to-frame" |
| "cheap dashcam lenses" | "inexpensive dashcam lenses" (more professional) |
| "Ego-Vehicle" | "ego vehicle" (no caps needed) |
| "Lane Segmentation" | "Lane segmentation" (sentence case) |
| "Projects those digital pixel displacements" | "Projects digital pixel displacements" (remove "those") |

---

### 10. Introduction Could Be Tighter
**Current:** Three bullets with technical terms (WIFI, GPS, GNSS).\n
**Suggested rewrite:**
> *"Dashcams today display speed using GPS — but GPS needs an Internet connection and fails in tunnels or remote areas. Our project asks: Can the camera figure out speed just by looking at the road? We built a system that reads motion from the video itself, no connectivity required."*

This gets to the "why" faster and avoids jargon.

---

### 11. Future Improvements: Add Comma2k19
**Current:** Six improvements listed, none mention comma2k19.\n
**Add:**
> *"Cross-dataset fine-tuning on Comma2k19 highway data — would extend speed range from 0–20 mph to 0–80 mph."*

---

### 12. Team Name Order
The presentation lists: *"Samad Afolabi, Iyanuoluwa Joanna Ahmed, and Martina Kuzenski"*

The README/notebook list: *"Samad Afolabi, Martina Kuzenski, Iyanuoluwa Joanna Ahmed"*

**Fix:** Pick one order and use it consistently everywhere. (Alphabetical by last name is standard: Afolabi, Ahmed, Kuzenski.)

---

## Suggested Slide Order (Optional Reorganization)

If you want to tighten the narrative flow:

1. **Title** *(keep)*
2. **Introduction** *(tighten, drop GPS jargon)*
3. **Related Work** *(condense to 3 bullets each)*
4. **Problem & Proposed Solution** *(keep — this is your "hook")*
5. **Methodology** *(keep diagram, add optical flow image)*
6. **Dataset** *(NEW — show sample video frame + optical flow side by side)*
7. **Training** *(NEW — show training curves plot)*
8. **Results — Validation** *(show scatter plot or error histogram)*
9. **Results — Post-Processing** *(show raw vs. smoothed comparison)*
10. **Results — Benchmark** *(keep table)*
11. **Cross-Dataset Testing** *(NEW — comma2k19)*
12. **Future Improvements** *(add comma2k19 bullet)*
13. **Demo** *(add description)*
14. **Q&A**

---

## Summary Checklist

- [ ] Fix or remove empty Abstract slide
- [ ] Update File Structure to remove `src/` references
- [ ] Add Comma2k19 testing mention (new slide or bullet)
- [ ] Add content to DEMO slide
- [ ] Embed training curves plot in Results
- [ ] Embed optical flow visualization in Methodology
- [ ] Embed raw vs. smoothed comparison
- [ ] Fix "Aug 17, 2026" typo
- [ ] Grammar pass (Wi-Fi, frame-by-frame, etc.)
- [ ] Standardize team member order
- [ ] Add comma2k19 to Future Improvements
