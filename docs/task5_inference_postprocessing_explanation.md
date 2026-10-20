# Task 5: Inference and Post-Processing — What Happened and Why?

## 1. What We Just Did

We loaded our trained "speed detective" (the CNN model saved at epoch 16) and unleashed it on the **test video** — a 10,797-frame dashcam recording the model had never seen before. Frame by frame, we computed optical flow, fed the motion images to the model, and got raw speed guesses. Then we applied two sanity-check rules on top: the **stop threshold** and the **rolling average smoother**.

The result is a CSV file with 10,797 predicted speed values and a clean speedometer-style plot showing how the vehicle's speed changed over the entire test drive.

---

## 2. The "Nervous Driver vs. Calm Instructor" Analogy

Imagine two people riding with you, both trying to read the speedometer:

- **The Nervous Driver** (raw model output) looks at the needle every millisecond and shouts numbers: *"30! 35! 28! 31! 40! 25!"* Their eyes are jumpy, and they are reacting to every tiny vibration of the car.
- **The Calm Instructor** (post-processed output) glances at the needle every few seconds and reports: *"You're holding about 30 mph."* They ignore momentary jitters and give you the *trend*, not the noise.

Our raw CNN is the nervous driver. It sees every optical flow frame and guesses independently — sometimes it misreads a shadow or a passing bird as a speed change. Post-processing is the calm instructor who steps in and says, *"Slow down — let me average the last 5 readings, and if you're basically stopped, I'll just call it zero."*

---

## 3. The Two Post-Processing Rules Explained

### Rule 1: Stop Threshold
**Logic:** If the model predicts speed ≤ 2.0 mph, override it to exactly 0 mph.

**Why?** When a car is parked at a red light, the camera still sees tiny motions — windshield wipers, leaves rustling, a pedestrian walking past, or even just camera shake. The optical flow algorithm picks these up and the model might say *"0.8 mph"* or *"1.3 mph."* A human driver knows the car is stopped. The threshold rule codifies that common sense: *"If it's close enough to zero, it IS zero."*

Think of it like a digital scale that refuses to display 0.2 lbs when nothing is on it — it just shows 0.0.

### Rule 2: Rolling Average Smoothing
**Logic:** Instead of reporting each frame's prediction instantly, maintain a buffer of the last 5 predictions and report their average.

**Why?** Real speedometers don't flicker wildly. They glide. If you're driving at a steady 35 mph, a good speedometer reads 35, not 33→36→34→37. The rolling average acts as a low-pass filter — it lets slow, meaningful trends through while damping out high-frequency noise.

Think of it like a shower faucet's pressure regulator. Without it, every tiny turn of the handle blasts you with scalding or freezing water. With it, temperature changes feel gradual and comfortable.

---

## 4. What the Comparison Plot Reveals

### Top Panel: Raw Model Output
The raw predictions look like a seismograph during an earthquake. Speed spikes up to 20+ mph and crashes back down within single frames. The orange line is so jagged it is almost unreadable. This is the "nervous driver" in full panic mode.

### Bottom Panel: Post-Processed Output
The same predictions, but after applying threshold + rolling average, look like a real speedometer trace. The curve rises smoothly during acceleration, holds steady during cruising, drops cleanly during braking, and flatlines at exactly 0 mph during stops. The "calm instructor" has taken over.

### The Numbers Back It Up
| Metric | Raw Output | Post-Processed | Improvement |
|---|---|---|---|
| Min speed | 0.37 mph | **0.00 mph** | Stop threshold catches near-zeros |
| Max speed | 21.05 mph | 19.79 mph | Smoother eliminates extreme spikes |
| Std dev | 4.42 | **4.10** | 7% reduction in volatility |

The standard deviation dropped from 4.42 to 4.10, proving the smoother is doing its job of suppressing noise while preserving the overall driving pattern.

---

## 5. Why Inference Speed Matters

Our lightweight CNN processes each frame in well under 30 milliseconds on a CUDA GPU. At 20 fps, that means the model keeps up with the video feed in real time. A heavy model like ResNet-152 or a transformer might take 100+ ms per frame, causing the display to lag behind reality. Our 430K-parameter design hits the sweet spot: accurate enough to be useful, fast enough to be real-time.

Think of it like choosing between a sports car and a tank. The tank is armored and powerful but slow. The sports car is nimble and gets you there fast. For a dashcam speedometer, nimble wins.

---

## 6. Test Video Story: Reading the Speed Plot

Looking at the post-processed output for the full 10,797 frames:

- **Mean speed: 10.85 mph** — This is suburban/city driving, not highway.
- **Speed oscillates between 3–20 mph** — Lots of acceleration and deceleration, typical of traffic lights and stop signs.
- **Flat zero regions** — Clearly visible stops where the car is parked or waiting.
- **No sustained high-speed stretches** — Unlike the training video which had highway segments, this test drive appears to be entirely local roads.

The model produces a **plausible and internally consistent** speed trace. Even without ground-truth labels for the test set, the output passes the "smell test" — it looks like a real car's speedometer.

---

## 7. What Task 6 Will Cover

With training, inference, and post-processing all working, the final task is **evaluation and project wrap-up**. We will:

1. Summarize the entire pipeline end-to-end.
2. Quantify the final model performance on validation data (MAE 1.56 mph, RMSE 2.06 mph).
3. Discuss real-world deployment considerations (edge cases, camera angles, weather).
4. Identify future improvements (temporal models, pretrained backbones, multi-camera fusion).
5. Package everything into a clean deliverable the team can present.

Think of Task 6 as writing the final report after the experiment — documenting what worked, what didn't, and what comes next.
