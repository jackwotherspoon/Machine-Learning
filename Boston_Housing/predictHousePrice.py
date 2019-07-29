# Import dependencies
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
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
# plt.show()

# create correlation matrix to measure relationship between variables
correlation_matrix = boston.corr().round(2)
sns.heatmap(data = correlation_matrix, annot = True)
#plt.show()

# split data into training set and test set
features = boston.drop('MEDV', axis = 1)
labels   = boston['MEDV']
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size = 0.3, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)