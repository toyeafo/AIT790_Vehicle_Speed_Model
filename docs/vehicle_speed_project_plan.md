# Project Plan: Live Vehicle Speed Estimation from Dashcam Video

## Objective
To build and train a machine learning model to estimate the speed of a vehicle in real-time using only a dashcam video.
*   **Input:** Live or pre-recorded dashcam video.
*   **Output:** Live speed estimate.

---

## Step 1: Data Acquisition
*Goal: Acquire quality, dashcam video and speed data.*

We will use open-source driving datasets to avoid the cost and time of manual data collection.
1.  **Comma.ai Speed Prediction Challenge (Primary Start):** 26 minutes of driving at 20fps. We will use this to test possibilities because it also provides a synced `train.txt` file with exact speeds for every frame. [Dataset](https://github.com/commaai/speedchallenge/tree/master/data)
2.  **Comma2k19:** 33+ hours of highway driving with raw CAN bus speed logs.

---

## Step 2: Data Preparation
*Goal: Translate raw videos and text files into a format the model can understand.*

1.  **Extract Frames:** We will use Python and OpenCV to break the `.mp4` video into individual image frames.
2.  **Synchronize Data:** Read the `train.txt` file. Map Frame 0 to Line 0, Frame 1 to Line 1, ensuring every image has a precise speed label.
3.  **Compute Optical Flow:** Instead of feeding raw images to the model, use OpenCV (`cv2.calcOpticalFlowFarneback`) to calculate "Dense Optical Flow." This creates a new image that represents the *motion* of pixels between consecutive frames, making it much easier for the AI to understand speed.
4.  **Normalize:** Resize the images to a standard dimension (e.g., 224x224) and scale the pixel values so the neural network trains efficiently.

---

## Step 3: Model Architecture & Training
*Goal: Teach the model the mathematical relationship between pixel motion and vehicle speed.*

1.  **Select Framework:** Use PyTorch or TensorFlow/Keras.
2.  **Architecture:** Build a Convolutional Neural Network (CNN) to analyze the spatial patterns of the Optical Flow, potentially paired with a Recurrent Neural Network (RNN/LSTM) to remember the sequence of motion over time.
3.  **Training Loop:** 
    *   Feed the Optical Flow data into the model.
    *   Compare the model's guess to the true speed (Mean Squared Error loss).
    *   Update the model's weights to improve accuracy.
4.  **Validation:** Test the model on a hidden set of the comma.ai data (the validation set) to ensure it isn't just memorizing the training data.

---

## Step 4: Post-Processing & The "Stop" Rule
*Goal: Ensure practical accuracy, especially at low speeds.*

1.  **Apply Threshold:** Write a hardcoded logic rule on top of the model's output. If the model outputs `speed <= 2.0 mph`, override the output to `0 mph`.
2.  **Filter Noise:** Apply a smoothing filter (like a rolling average over the last 5 frames) to the output so the live speed display doesn't wildly jump around (e.g., jumping from 30 to 35 to 29 in a single second).

---

## Step 5: Real-World Testing & Optimization
*Goal: Make the model run fast enough for live video.*

1.  **Inference Speed:** Optimize the code so that extracting a frame, computing optical flow, and running the model takes less than 30 milliseconds. This ensures it can process a standard 30fps camera feed in real-time.
2.  **Edge Case Testing:** Run the BDD100K rain/night videos through the model to identify weaknesses and retrain if necessary.
