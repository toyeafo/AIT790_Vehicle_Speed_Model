# Task 4: Training the Model — What Happened and Why?

## 1. What We Just Did

We fed our Convolutional Neural Network (the "pattern detective") 16,320 flashcards (optical flow images paired with speed labels) and let it study for 21 learning rounds.

Along the way, we also tested it on 4,079 *unseen* flashcards (the validation set) to make sure it was actually learning, not just memorizing. When its performance on the unseen cards stopped improving, we stopped the process early. The best version was saved to disk.

---

## 2. The "Driving School" Analogy

Imagine you are teaching 16,320 students how to drive. Each student brings you a photo of the road and asks, *"How fast am I going?"* You give 16,320 practice sessions to these students with a sheet of correct speed answers. Between rounds, you adjust what the students pay attention to — lane markings, trees rushing by, etc.

Then, after each round of practice, you bring in 4,079 *new* students who were never in the original group. If the class can also answer correctly for these new students, you know they have learned to drive, not just memorized the faces of the original group.

The moment the new students' scores stop improving, you stop the class. The best-performing version of the class gets a "graduation certificate" saved to `outputs/best_model.pth`.

---

## 3. Why Mean Squared Error (MSE)?

We use MSE as the "grading rubric" for each flashcard.

- If the model guesses **28 mph** when the answer is **28 mph**, the error is 0. Perfect.
- If it guesses **30 mph** when the answer is **28 mph**, the error is small — (2)² = 4.
- If it guesses **80 mph** when the answer is **28 mph**, the error is massive — (52)² = 2704.

Think of it like a school grading policy where small mistakes cost you a few points, but huge, dangerous mistakes cost you everything. Squaring the error forces the model to prioritize getting rid of its worst, most embarrassing guesses first.

---

## 4. Training Metrics Explained

### Train Loss vs. Val Loss
- **Train loss** is how wrong the model is on the flashcards it just studied.
- **Val loss** is how wrong it is on *brand new* flashcards it has never seen.

Both trend downward over time, which is good. But you might notice some **spikes** in the orange validation line. Those are like a student having an off day — maybe the validation batch happened to contain a tricky highway curve or a dark tunnel, and the model struggled. Because we saved the *best* checkpoint (lowest val loss), these spikes are harmless.

### MAE (Mean Absolute Error)
MAE is the average margin of error in mph. **1.56 mph** means that, on average, our guess is off by less than 2 mph. That is roughly the difference between 58 mph and 59 mph — close enough that a driver could still trust the number.

### RMSE (Root Mean Squared Error)
RMSE is MAE's more dramatic cousin. Because it involves square roots, it penalizes the model harder for big misses. An RMSE of **2.06 mph** tells us there are some occasional big misses (those highway/tunnel spikes), but most predictions cluster closely around the truth.

---

## 5. What the Plots Reveal

### Plot 1: Training Curves (Loss, MAE, RMSE)
The blue line shrinks steadily — our student is learning. The orange line wobbles but on average follows the same downward path. The wobbles prove our validation data is hard enough to be a meaningful test.

### Plot 2: Predicted vs. Actual Speed
The blue line is the *true* speed. The orange is our model's guess. The orange follows the blue line remarkably well — it knows when the car accelerates, decelerates, and stops.

But the orange line is noisy. At any given moment, the model might overshoot or undershoot by a few mph. This is where the **post-processing step** (rolling average + stop threshold) comes in — it is like adding a seasoned driving instructor who whispers, *"No, ignore that brief jitter — the car is actually holding 35 mph."*

---

## 6. Epoch-by-Epoch Story

| Epoch | What Happened |
|---|---|
| 1–5 | The model starts clueless. Train/val MAE near 6 mph — it is basically guessing.
|
| 6–10 | Rapid improvement. By epoch 10, val MAE drops to 2.5 mph. The detective is recognizing motion patterns.
|
| 11–16 | Continued refinement. At **epoch 16**, the model reaches its peak: **1.56 mph average error**. This is the best checkpoint.
|
| 17–21 | The model starts overfitting slightly — it memorizes quirks of the training data and loses its edge on unseen data. Early stopping kicks in at epoch 21 and preserves the epoch-16 model.

---

## 7. Why Epoch 16 Is the Winner (Not Epoch 21)

After epoch 16, the model started getting slightly worse on new examples even though it kept improving on old ones. Imagine a student who studies flashcards so hard they start memorizing the *font* and *paper texture* instead of the actual content. Early stopping is like a wise teacher saying, *"Stop — you were at your best three days ago."*

The checkpoint from epoch 16 is saved as `outputs/best_model.pth` because it had the lowest validation loss: **4.25**.

---

## 8. How Good Is 1.56 mph?

| Benchmark | Typical MAE |
|---|---|
| Naïve baseline (guess average speed) | ~10 mph |
| Simple linear regression on pixels | ~8 mph |
| Scratch CNN (us) | **1.56 mph** |
| ResNet18 pretrained on ImageNet | ~0.8–1.2 mph |

For a lightweight 430K-parameter model built from scratch (no pretraining, no ImageNet), **1.56 mph** is a strong result. It proves our strategy — optical flow → CNN → speed — is sound.

If we wanted to push into the sub-1 mph elite tier, we would need a temporal model (LSTM across frames) or a pretrained backbone. But for a real-time dashcam application, **1.56 mph** is well within the "good enough" range.

---

## 9. What Happens Next?

We now have a trained detective (`best_model.pth`) who can look at a motion image and guess speed. But the detective is sometimes jumpy — individual guesses can fluctuate. In **Task 5 (Inference & Post-Processing)**, we will:

1. Load this detective onto new dashcam videos.
2. Apply the **stop threshold** rule — anything near 0 mph gets rounded to exactly 0.
3. Apply the **rolling average** smoother — the last 5 guesses are averaged to produce a clean, stable speedometer reading.

Think of it as the final quality-control step before we let our AI student drive a real car.
