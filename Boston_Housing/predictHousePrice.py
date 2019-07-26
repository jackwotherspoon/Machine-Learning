# Import dependencies
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import seaborn as sns

# load the boston housing dataset
boston_dataset = load_boston()  
print("All the different keys in dataset are :", boston_dataset.keys())

# seperate data into its features 
boston = pd.DataFrame(np.array(boston_dataset["data"]), columns = boston_dataset["feature_names"])
print("Snapshot of what dataframe looks like :\n", boston.head())
print("Shape of the datasets inputs (features) : ", boston.shape)

# add median value of homes (labels) to dataframe
boston['MEDV'] = boston_dataset["target"]
print("Snapshot of dataframe with target values MEDV now shown :\n", boston.head())

# check if any values are null before preprocessing
print("Checking if any values are missing in dataset:\n", boston.isnull().sum())

# plot median value of houses using seaborn
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()