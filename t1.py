import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load dataset directly from Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Select 10 images and resize to ResNet50 input size
imgs = tf.image.resize(x_test[:10], (224,224))
x = preprocess_input(imgs)

# Pre-trained CNN
model = ResNet50(weights="imagenet")

# Predict Top-5
pred = model.predict(x, verbose=0)

for i in range(10):
    print("\nImage", i+1)
    print(decode_predictions(pred[i:i+1], top=5)[0])
    plt.imshow(x_test[i])
    plt.axis("off")
    plt.show()