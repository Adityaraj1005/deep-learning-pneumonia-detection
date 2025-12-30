import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,

    zoom_range=0.2,

    brightness_range=[0.8, 1.2],

    horizontal_flip=True,
    
)

training_set = train_datagen.flow_from_directory(
    r"C:\Users\LENOVO\Downloads\archive (1)\chest_xray\train",
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    color_mode='grayscale'
)


cnn=tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[224,224,1]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
val_datagen = ImageDataGenerator(rescale=1./255)

validation_set = val_datagen.flow_from_directory(
  r"C:\Users\LENOVO\Downloads\archive (1)\chest_xray\train",

    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'
)


cnn.fit(training_set,validation_data=validation_set,epochs=10)

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
   r"C:\Users\LENOVO\Downloads\archive (1)\chest_xray\train",
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'
)

test_loss, test_accuracy = cnn.evaluate(test_set)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

cnn.save("pneumonia_cnn_model.keras")


from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Load image
test_image = load_img(
    r"C:\Users\LENOVO\Downloads\archive (1)\chest_xray\test\NORMAL\IM-0011-0001-0001.jpeg",
    target_size=(224,224),
    color_mode='grayscale'
)

# Convert to array
test_image = img_to_array(test_image)

# Expand dims to create batch of 1
test_image = np.expand_dims(test_image, axis=0)

# Predict
result = cnn.predict(test_image)

# Get class mapping
print(training_set.class_indices)  # e.g. {'NORMAL':0, 'PNEUMONIA':1}

# Convert to label
prediction = 'pneumonia' if result[0][0] > 0.5 else 'normal'
print("Prediction:", prediction)
