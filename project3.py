# ===================== IMPORT LIBRARIES =====================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===================== IMAGE AUGMENTATION FOR TRAINING =====================
# ImageDataGenerator helps:
# 1. Normalize images (rescale)
# 2. Create augmented versions of images to avoid overfitting

train_datagen = ImageDataGenerator(
    rescale=1./255,               # Normalize pixel values (0–255 → 0–1)

    rotation_range=15,            # Randomly rotate images
    width_shift_range=0.1,        # Random horizontal shifts
    height_shift_range=0.1,       # Random vertical shifts

    zoom_range=0.2,               # Random zoom
    brightness_range=[0.8, 1.2],  # Random brightness changes
    horizontal_flip=True          # Flip images horizontally
)

# ===================== LOAD TRAINING DATA =====================
# flow_from_directory:
# - Reads images from folders
# - Assigns labels based on folder names

training_set = train_datagen.flow_from_directory(
    r"C:\Users\LENOVO\Downloads\archive (1)\chest_xray\train",
    target_size=(224, 224),   # Resize all images to 224x224
    batch_size=64,            # Number of images per batch
    class_mode='binary',      # Binary classification (NORMAL vs PNEUMONIA)
    color_mode='grayscale'    # Convert images to grayscale (1 channel)
)

# ===================== BUILD CNN MODEL =====================

cnn = tf.keras.models.Sequential()

# First Convolution + Pooling layer
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu',
    input_shape=[224, 224, 1]   # Image size + grayscale channel
))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Second Convolution + Pooling layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flatten feature maps into 1D vector
cnn.add(tf.keras.layers.Flatten())

# Fully connected layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output layer
# Sigmoid is used for binary classification
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# ===================== COMPILE MODEL =====================
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===================== VALIDATION DATA =====================
# Validation data is used to check model performance during training

val_datagen = ImageDataGenerator(rescale=1./255)

validation_set = val_datagen.flow_from_directory(
    r"C:\Users\LENOVO\Downloads\archive (1)\chest_xray\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'
)

# ===================== TRAIN MODEL =====================
cnn.fit(
    training_set,
    validation_data=validation_set,
    epochs=10
)

# ===================== TEST DATA =====================
# Test data is used for final evaluation

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    r"C:\Users\LENOVO\Downloads\archive (1)\chest_xray\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'
)

# ===================== EVALUATE MODEL =====================
test_loss, test_accuracy = cnn.evaluate(test_set)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# ===================== SAVE MODEL =====================
cnn.save("pneumonia_cnn_model.keras")

# ===================== SINGLE IMAGE PREDICTION =====================

from tensorflow.keras.utils import load_img, img_to_array

# Load a single image
test_image = load_img(
    r"C:\Users\LENOVO\Downloads\archive (1)\chest_xray\test\NORMAL\IM-0011-0001-0001.jpeg",
    target_size=(224, 224),
    color_mode='grayscale'
)

# Convert image to NumPy array
test_image = img_to_array(test_image)

# Add batch dimension (model expects batches)
test_image = np.expand_dims(test_image, axis=0)

# Predict
result = cnn.predict(test_image)

# Print class labels mapping
print(training_set.class_indices)  # {'NORMAL': 0, 'PNEUMONIA': 1}

# Convert prediction to readable label
prediction = 'pneumonia' if result[0][0] > 0.5 else 'normal'
print("Prediction:", prediction)
