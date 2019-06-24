# Import dependencies
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the Boston Housing dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
