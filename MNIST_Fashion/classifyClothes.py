#Import dependencies
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt

# Verify tensorflow version
print(tf.__version__)

# Load the MNIST fashion dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "x_test shape:", x_test.shape)

# Show one of the images
plt.imshow(x_train[0])
plt.show()

# Normalize the data so that they are the same scale
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

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
train_dataset = x_train.repeat().shuffle(len(x_train)).batch(batch_size)
test_dataset = x_test.batch(batch_size)
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(len(train_dataset)/batch_size))
