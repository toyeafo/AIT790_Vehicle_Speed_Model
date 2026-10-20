# Vehicle Speed Estimation from Dashcam Video — Project Explanation

## 1. The Big Goal: "How Fast Are We Going?"

**The Problem:** You are in a car with a dashcam. The camera sees the road ahead — lane markings, trees, buildings, other cars. But there is no speedometer on screen. Can we teach a computer to look at the video and say, *"You are driving 28.3 mph right now"*?

**The Challenge:** A single photograph of the road doesn't tell you speed. A picture of a highway at 30 mph looks almost identical to one at 60 mph. The *difference between two consecutive pictures* is what contains the speed information.

---

## 2. The Core Idea: Optical Flow (The "Desk Fan" Analogy)

Imagine you are sitting at a desk with a fan blowing air at you. If you take one photograph of the fan blades, you have no idea how fast they're spinning. They might be frozen still. But if you take **two photos a split-second apart** and compare them, you can see exactly how far each blade moved. The faster the spin, the farther each blade travels between frames.

**Optical Flow is that comparison.** For every pixel in Frame 1, we ask: "Where did this pixel move to in Frame 2?" We draw an arrow for every single pixel showing its motion. The longer the arrows, the faster the motion. We then color-code those arrows into an image — bright colors mean fast motion, dark means slow.

So instead of feeding the AI raw video frames (which look the same at slow and fast speeds), we feed it the **difference between frames** — a picture where *brightness = speed*. Now the AI has a fighting chance.

---

## 3. The Model: A Convolutional Neural Network (The "Pattern Detective" Analogy)

A CNN is like a detective who looks at clues in small patches and builds up to the big picture.

- **Layer 1** looks at tiny 3×3 patches and says, *"I see some motion pointing left here."*
- **Layer 2** combines those patches and says, *"That left motion is actually the edge of a lane marker sliding past."*
- **Layer 3** combines those and says, *"Lane markers are sliding fast, so we must be going fast."*
- **The final layer** outputs a single number: **speed in mph.**

We built a lightweight CNN (about 300,000 parameters) so it can run in real-time. A huge model like GPT or a massive ResNet would be like using a supercomputer to check the weather — accurate, but way too slow for a car driving down the road. Our model is designed to make predictions in under 30 milliseconds, which is fast enough for a 30 fps video stream.

---

## 4. The Training Process (The "Student Learning from Flashcards" Analogy)

Think of training like a student studying with flashcards.

- **Front of the card:** An optical flow image.
- **Back of the card:** The correct speed.

The model looks at a flow image, guesses a speed, and checks the answer. If it's wrong, we calculate how wrong (Mean Squared Error) and nudge the model's internal "weights" slightly in the right direction. We do this millions of times across thousands of flashcards.

**Why MSE Loss?** Because if the model guesses 30 mph when the answer is 28 mph, the penalty is small. If it guesses 80 mph, the penalty is huge. Squaring the error punishes big mistakes severely, which keeps the model honest about egregious guesses.

---

## 5. Why This Specific Dataset? (The "Comma.ai Homework" Analogy)

We are using the **Comma.ai Speed Challenge** dataset. It is like getting the answer key to a homework assignment. It contains:

- A 20-minute dashcam video.
- A text file with **exactly one speed value for every single frame.**

This is rare and valuable. Most driving datasets only give you video. Here, we know the ground truth for every frame, so we can grade our model accurately. It is the perfect "textbook" for this problem.

---

## 6. Data Preparation: What We Actually Did

### Step 1: Download the Videos
The original video files were stored with Git LFS (Large File Storage) — basically placeholders instead of real files. We fetched the actual MP4s from GitHub's media server.

### Step 2: Compute Optical Flow
For every pair of consecutive frames, we ran OpenCV's Farneback algorithm to generate a "motion picture." We saved 20,399 of these for the training video and 10,797 for the test video.

### Step 3: Pair Flow Images with Speed Labels
Each flow image gets matched with the speed label of its *ending frame*. Why the ending frame? Because the flow image represents the motion that happened *between* frame N and frame N+1. By the time that motion is complete, the car is at the speed recorded for frame N+1.

### Step 4: Temporal Train/Validation Split
We split the data **chronologically**: the first 80% of the drive is training, the last 20% is validation.

**Why not random?** Imagine studying for a history exam by memorizing random pages from a textbook, then being tested on pages from a completely different chapter. Cars drive in patterns — traffic lights, highways, turns. If we randomly shuffled frames, the model might "cheat" by seeing the future of a road curve it already learned from the past. A temporal split forces the model to generalize to *new driving conditions* it hasn't seen yet.

---

## 7. Post-Processing: The "Stop Rule" (The "Thermostat" Analogy)

After the model predicts a speed, we apply two sanity checks:

### Threshold Rule
If the model says "1.5 mph," we override it to **0 mph**. Why? Because when you're stopped at a red light, tiny optical flow movements (windshield wipers, shadows, camera shake) can trick the model into thinking you're crawling forward. A hard rule says: *"If it's basically zero, it's zero."*

Think of it like a thermostat with a "dead zone" — the heater doesn't turn on until the temperature drops 2 degrees below the target, preventing frantic on/off switching.

### Rolling Average Smoothing
Speedometers in real cars don't jump from 30 → 35 → 28 → 31 every fraction of a second. They glide smoothly. We apply a **rolling average** over the last 5 predictions so the live display feels natural and stable.

Think of it like a shower faucet: instead of blasting you with scalding or freezing water every time you nudge the handle, a pressure regulator smooths out the temperature.

---

## 8. Why PyTorch and Not TensorFlow?

TensorFlow is like building a house with blueprints made by an architect — powerful but rigid. PyTorch is like building with LEGO blocks — you can snap pieces together and immediately see if they fit. Since our team needs to experiment quickly (trying different CNN depths, tweaking optical flow parameters), PyTorch's flexibility wins. Plus, PyTorch 2.14 with CUDA support was already installed in the environment, so we hit the ground running.

---

## 9. The Full Pipeline (The "Assembly Line" Analogy)

Here is the entire production line from raw materials to finished product:

```
RAW VIDEO          →   OPTICAL FLOW          →   CNN MODEL           →   POST-PROCESSING   →   LIVE SPEED
(dashcam MP4)         (motion images)           (pattern detective)     (thermostat +        (mph display)
                                                               smoothing faucet)
```

1. **Video Frame Extractor** grabs consecutive frames.
2. **Optical Flow Engine** calculates the motion between them and paints it into a color image.
3. **Dataset Handler** pairs each motion image with the correct speed label and splits data 80/20.
4. **CNN Model** studies millions of flashcards and learns to guess speed from motion patterns.
5. **Trainer Loop** grades the model, punishes big errors, and saves the best version.
6. **Inference Script** loads the trained model and runs it on new videos in real-time.
7. **Post-Processor** catches "almost stopped" edge cases and smooths the output.

---

## 10. What Happens Next?

With optical flow images and labels ready, the next phase is **training**. We will run the `train.py` script, which feeds the flashcards to our CNN detective. Over ~20–30 epochs, the model will learn to map motion patterns to speed values. We will watch training curves (loss, MAE, RMSE) to know if the detective is getting smarter or just memorizing answers. If all goes well, we will have a lightweight, real-time speed estimator ready for testing on the held-out validation data.

---

## Summary

**The problem is hard because photos don't contain speed.** Our breakthrough is **optical flow**, which turns speed into brightness that a neural network can see. We train on a dataset with known answers, validate on unseen driving conditions, and apply common-sense rules at the end to make the output feel realistic. The result is a system that can estimate vehicle speed from nothing but a dashcam video.
