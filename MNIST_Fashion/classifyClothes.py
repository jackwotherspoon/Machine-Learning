#Import dependencies
import tensorflow as tf
import math
import numpy as np
import sys
import matplotlib.pyplot as plt

# Verify tensorflow version
print(tf.__version__)

# Load the MNIST fashion dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "x_test shape:", x_test.shape)

# Show one of the images
plt.imshow(x_train[0])
#plt.show()

# Normalize the data so that they are the same scale and flatten
x_train = x_train.astype('float32') / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Create sequential model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),      # transforms image from 2D array of 28x28 pixels to a 1D 1x728 pixels
    tf.keras.layers.Dense(128, activation=tf.nn.relu),   # dense layer (hidden layer) of 128 neurons with relu activation function
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # dense layer (output layer) of 10 neurons, one for each class
])
# Look at model summary
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
batch_size = 32
model.fit(x_train, y_train, batch_size, epochs=10)

# Evaluate accuracy on test set
test_loss , test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy on test set is:", round(test_accuracy*100, 2), "%")

