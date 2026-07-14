# 🩺 Pneumonia Detection Using Deep Learning (CNN)

Pneumonia is a serious respiratory infection that affects millions of people worldwide and can be life-threatening if not diagnosed early.

This project uses **Convolutional Neural Networks (CNNs)** to automatically detect **Pneumonia from chest X-ray images**, helping in faster and more reliable diagnosis using **deep learning**.

The model classifies chest X-ray images into two categories:

- **NORMAL**
- **PNEUMONIA**

This project demonstrates the application of **deep learning techniques in medical image classification** using **TensorFlow and Keras**.

---

## 💡 Why I Chose This Project
I chose this project because:

- Medical image analysis is a real-world, high-impact application of deep learning
- Pneumonia detection is a **binary classification problem**, making it ideal for learning CNN fundamentals
- It helped me understand:
  - Image preprocessing and data augmentation
  - CNN architecture design
  - Overfitting and validation
  - Model evaluation and prediction on unseen data
- Healthcare-related ML projects highlight how **AI can assist doctors and improve patient outcomes**

---

## ✅ Dataset Source
- **Dataset:** Chest X-Ray Images (Pneumonia)
- **Source:** Kaggle
- **Originally published by:** Guangzhou Women and Children's Medical Center

This is a widely used and well-known dataset for pneumonia detection using CNNs.

---

## 🧠 Dataset Description
The dataset is organized as follows:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Dataset Details

- Grayscale chest X-ray images
- Images resized to **224 × 224**
- Used for **binary classification**

### 🔹 Dataset Split Sizes
- **Train:** 5,216 images
- **Validation:** 16 images
- **Test:** 624 images

---

## 🏗 Model Architecture
- Convolutional layers with **ReLU activation**
- Max Pooling layers for downsampling
- Fully connected dense layer
- **Sigmoid output layer** for binary classification

### 🔧 Training Details
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy

---

## ⚙ Technologies Used
- Python
- TensorFlow & Keras
- NumPy
- VS Code

---

## 📊 Results

The CNN model was trained for **10 epochs** using grayscale images and data augmentation.

### 🔹 Training & Validation Performance
- Training accuracy improved from **~78.99% to ~92.91%**
- Validation accuracy fluctuated between **~62.5% and ~81.25%**
- Loss consistently decreased on training data, indicating stable learning

> **Note on validation accuracy:** The official Kaggle validation set
> (`val/`) contains only **16 images**, so each misclassified image
> swings validation accuracy by roughly 6%. This explains the noisy,
> non-monotonic validation curve across epochs — it's a property of
> the tiny validation set size, not a sign of unstable training.

### 🔹 Final Test Performance
- **Test Accuracy:** 90.71%
- **Test Loss:** 0.2890

### 🔹 Class Mapping
```python
{'NORMAL': 0, 'PNEUMONIA': 1}
```

### 🔹 Sample Prediction on Unseen Image
- Input: Chest X-ray image (NORMAL)
- Predicted Class: NORMAL
- The model successfully classified the image correctly

### 🔹 Key Observations
- The model generalizes reasonably well on unseen data
- Data augmentation improved robustness
- A simple binary CNN architecture proved effective for medical image classification

📌 **Overall Test Accuracy: ~90.71%** (evaluated on a held-out test set of 624 unseen images)

---

## ▶ How to Run the Project
```bash
git clone https://github.com/Adityaraj1005/deep-learning-pneumonia-detection.git
cd deep-learning-pneumonia-detection
pip install -r requirements.txt
python project3.py
```

---

## ⚠️ Challenges & Problems Faced
During development, several real-world challenges were encountered and resolved:

### 1️⃣ TensorFlow & Python Version Compatibility
- TensorFlow failed to run with Python 3.13
- **Solution:** Downgraded to Python 3.10, which is fully compatible with TensorFlow

### 2️⃣ Dataset Path & Extraction Issues
- Dataset was initially referenced directly from a `.zip` file
- This caused `FileNotFoundError`
- **Solution:** Properly extracted the dataset and updated absolute directory paths

### 3️⃣ Image Prediction Import Error
Earlier code (caused error):
```python
from keras.preprocessing import Image
```
This import is deprecated in newer TensorFlow/Keras versions.

Fixed code:
```python
from tensorflow.keras.utils import load_img, img_to_array
```
**What changed:**
- Replaced deprecated Keras import
- Used TensorFlow-recommended utility functions
- Image prediction worked correctly after the fix

### 4️⃣ Slow Training Time
Each epoch took several minutes due to:
- CPU-only training (no GPU)
- Large image size (224 × 224)

**Solution:**
- Increased batch size to reduce steps per epoch
- Accepted slower training as a CPU hardware limitation

### 5️⃣ Validation/Test Data Leakage
Initially, the validation and test generators both pointed to the `train/` folder by mistake, so the originally reported ~93% accuracy actually reflected training-set performance, not true generalization.

**Solution:** Corrected the paths so validation uses `val/` (16 images) and test uses `test/` (624 images). The real, held-out test accuracy is **90.71%**.

---

## 📌 Conclusion
This project demonstrates how deep learning and CNNs can be effectively applied to medical image analysis. Despite hardware limitations, the model achieved strong performance and provided valuable hands-on experience with real-world deep learning workflows — including debugging a real data evaluation issue and correcting it end-to-end.