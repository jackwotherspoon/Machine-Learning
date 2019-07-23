# Import dependencies
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

# load the boston housing dataset
boston = load_boston()  
print(boston.keys())
# seperate data into its features and labels 
features = pd.DataFrame(np.array(boston["data"]), columns = boston["feature_names"])
print(features.head())
features.shape()
print(test)