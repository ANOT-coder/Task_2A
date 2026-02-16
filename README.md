# 🐾 Cats vs. Dogs Image Classifier (CNN & Transfer Learning)
A high-performance Computer Vision system designed to classify images into two categories: **Cats and Dogs.** By evolving from a custom CNN to a **MobileNetV2** transfer learning architecture, this project achieved elite accuracy despite hardware constraints.

## 📊 Dataset Background
The model was trained using the **Kaggle Dogs vs. Cats Dataset**, containing thousands of RGB images. Image classification in this domain is challenging due to variations in lighting, background clutter, and the morphological diversity of different animal breeds.

* **Dataset Source:** [Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogsvscats)

* **Target Requirement:** >85% Validation Accuracy
* **Final Achievement:** **96.25% Validation Accuracy**

---

## 🛠 Deep-Dive: The Technical Rescue (3-Stage Evolution)
To reach the final objective under a strict deadline and GPU quota limits, the project underwent three distinct technical iterations:

1. **Iteration 1: Custom Deep CNN (Overfit/Imbalanced)**
   * **Architecture:** 4-block Conv2D stack with Batch Normalization.
   * **Failure:** Validation accuracy stuck at **51%**.
   * **Root Cause:** Lack of data shuffling meant the validation split contained only one class, causing the model to "rob" itself of the ability to generalize.

2. **Iteration 2: CPU-Optimized Light CNN (Underfit)**
   * **Architecture:** 3-block CNN using `GlobalAveragePooling2D`.
   * **Failure:** Accuracy plateaued at **61-71%**.
   * **Root Cause:** The model was too "lean" to extract complex features from scratch without significant training time.

3. **Iteration 3: Transfer Learning (MobileNetV2) — Final Choice**
   * **Architecture:** Pre-trained **MobileNetV2** backbone + Custom Global Average Pooling + Dropout.
   * **Success:** Achieved **96.25%** accuracy in just 5 epochs.

---

## 🧠 Why MobileNetV2? (Technical Architecture)
The project utilizes the **MobileNetV2** architecture, which is a "GOAT" model for constrained environments. It uses **Depthwise Separable Convolutions**, which drastically reduces the number of parameters compared to standard convolutions.

### 🔬 The Concept of Global Average Pooling (GAP)
Instead of using a traditional `Flatten()` layer that creates millions of trainable parameters, we used **Global Average Pooling**.
* **Mathematical Concept:** It takes a feature map of size $(H \times W \times D)$ and reduces it to $(1 \times 1 \times D)$ by taking the average of all pixels in each feature map. 
* **The Advantage:** It significantly reduces overfitting by minimizing the total number of weights the model has to learn.



---

## 📐 Mathematical Foundations
The model utilizes **Binary Cross-Entropy Loss** to calculate the error between the predicted probability and the actual label (0 for Cat, 1 for Dog):

$$L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$$

Where:
* $N$ is the number of samples.
* $y_i$ is the true label (0 or 1).
* $p_i$ is the probability predicted by the **Sigmoid** activation function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

---

## 📈 Final Evaluation Results
The final model was evaluated on a 20% validation split, showing exceptional stability and high confidence.

### Performance Summary:
| Metric | Iteration 1 | Iteration 2 | **Iteration 3 (Final)** |
| :--- | :--- | :--- | :--- |
| **Training Acc** | 94.03% | 71.04% | **98.75%** |
| **Validation Acc** | 51.00% | 61.37% | **96.25%** |
| **Status** | Failed (Overfit) | Failed (Underfit) | **Success (Satisified)** |

### 📊 Training Visualization


---

## 📂 Project Structure Checklist
The following components were implemented for the final system:
1. **`X_final` & `y_final`** – Shuffled Numpy arrays stored in RAM for high-speed access.
2. **MobileNetV2 Backbone** – Pre-trained weights from ImageNet (frozen).
3. **Adam Optimizer** – Used for adaptive learning rate management.
4. **Binary Crossentropy** – The loss function for our two-class problem.

---

## 🔍 Result Analysis

### Why 96% Accuracy?
* **Transfer Learning:** By using a model already trained on millions of images, we inherited the ability to detect edges, textures, and shapes instantly.
* **Shuffling:** Correcting the data order ensured the validation set was a fair representation of both classes.
* **Dropout Layer:** Applying `Dropout(0.3)` prevented the model from memorizing specific training images, forcing it to learn general features.

### 💡 Future Improvements
* **Fine-Tuning:** Unfreezing the top layers of MobileNetV2 once GPU access is restored to push accuracy toward 99%.
* **TFLite Conversion:** Compressing the model for real-time inference on mobile devices.
* **Data Augmentation:** Using `RandomFlip` and `RandomRotation` layers to increase the model's robustness against different camera angles.

---