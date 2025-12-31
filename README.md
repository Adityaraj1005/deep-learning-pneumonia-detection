- **NORMAL**
- **PNEUMONIA**

This project demonstrates the application of deep learning techniques in **medical image classification** using **TensorFlow and Keras**.

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
- **Originally published by:** Guangzhou Women and Children’s Medical Center  

This is a widely used and well-known dataset for pneumonia detection using CNNs.

---

## 🧠 Dataset Description
The dataset is organized as follows:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```


markdown
Copy code

- Grayscale chest X-ray images  
- Images resized to **224 × 224**
- Used for **binary classification**

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
- Training accuracy improved from **~79% to ~92%**
- Validation accuracy peaked at **~95%**
- Loss consistently decreased, indicating stable learning without major overfitting

### 🔹 Final Test Performance
- **Test Accuracy:** 92.73%
- **Test Loss:** 0.1847

### 🔹 Class Mapping
```python
{'NORMAL': 0, 'PNEUMONIA': 1}
🔹 Sample Prediction on Unseen Image
Input: Chest X-ray image (NORMAL)

Predicted Class: NORMAL

The model successfully classified the image correctly

🔹 Key Observations
The model generalizes well on unseen data

Data augmentation improved robustness

A simple binary CNN architecture proved effective for medical image classification

📌 Overall Accuracy: ~93%

▶ How to Run the Project
bash
Copy code
git clone https://github.com/Adityaraj1005/deep-learning-pneumonia-detection.git
cd deep-learning-pneumonia-detection
pip install -r requirements.txt
python project3.py
⚠️ Challenges & Problems Faced
During development, several real-world challenges were encountered and resolved:
```

1️⃣ TensorFlow & Python Version Compatibility
TensorFlow failed to run with Python 3.13

Solution: Downgraded to Python 3.10, which is fully compatible with TensorFlow

2️⃣ Dataset Path & Extraction Issues
Dataset was initially referenced directly from a .zip file

This caused FileNotFoundError

Solution: Properly extracted the dataset and updated absolute directory paths

3️⃣ Image Prediction Import Error
Earlier code (caused error):

```python
from keras.preprocessing import Image
```
This import is deprecated in newer TensorFlow/Keras versions

Fixed code:

```python
from tensorflow.keras.utils import load_img, img_to_array
```
What changed:

Replaced deprecated Keras import

Used TensorFlow-recommended utility functions

Image prediction worked correctly after the fix

4️⃣ Slow Training Time
Each epoch took several minutes due to:

CPU-only training (no GPU)

Large image size (224 × 224)

Solution:

Increased batch size to reduce steps per epoch

Accepted slower training as a CPU hardware limitation

📌 Conclusion
This project demonstrates how deep learning and CNNs can be effectively applied to medical image analysis.
Despite hardware limitations, the model achieved strong performance and provided valuable hands-on experience with real-world deep learning workflows.
