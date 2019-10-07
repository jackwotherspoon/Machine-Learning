from sklearn import datasets
# import dependencies
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 1. load dataset -------------------------------------------------------------------------------------------------------------
iris = datasets.load_iris()
print(type(iris))

# 2. summarize dataset --------------------------------------------------------------------------------------------------------
print(iris.keys())

# print type for the data and labels
print(type(iris.data), type(iris.target))

# print shape of data
print(iris.data.shape)

# print all the different labels
print(iris.target_names)

# 3. peek the data ------------------------------------------------------------------------------------------------------------
# perform EDA
X = iris.data
Y = iris.target
df = pd.DataFrame(X, columns = iris.feature_names)
print(df.head())

# statistical summary 
print(df.describe())

# class distribution
print(df.groupby(iris.target).size())

# 4. Data Visualization -------------------------------------------------------------------------------------------------------
# perform visual EDA
graphs = pd.plotting.scatter_matrix(df, c = Y, figsize = [10, 10], s = 150, marker = 'D')

# 5. Evaluate Algorithms ------------------------------------------------------------------------------------------------------
# split data into training and test set
test_set_size = 0.2
seed = 7
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_set_size, random_state = seed)

# validation will be done using 10-fold cross-validation method
models = []
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'auto')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma = 'auto')))
results = []
names = []

# loop through models and train
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = seed)
    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# compare algorithms
fig = plt.figure()
fig.suptitle('Alogrithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# KNN and SVM seem to be best for the data as displayed from cross validation results

# 6. Make Predictions (Test) ------------------------------------------------------------------------------------------------
# train KNN model
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# two ways of calculating accuracy scores on test set
score = knn.score(x_test, y_test)
print("KNN accuracy score method 1:", score)
predictions = knn.predict(x_test)
score2 = accuracy_score(y_test, predictions)
print("KNN accuracy score method 2: ", score2)

# print confusion matrix and classification report for KNN model
conf_mat = confusion_matrix(y_test, predictions)
print("Confusion matrix:\n", conf_mat)
class_rep = classification_report(y_test, predictions)
print("\nClassification Report:\n", class_rep)

# train the SVM
svm = SVC(gamma = 'auto')
svm.fit(x_train, y_train)

# predict class for test set and score accuracy
# show both scoring techniques
score3 = svm.score(x_test, y_test)
print("SVM accuracy score method 1: ", score3)
predictions2 = svm.predict(x_test)
score4 = accuracy_score(y_test, predictions2)
print("SVM accuracy score method 2: ", score4)

# print confusion matrix and classification report for SVM model
conf_mat2 = confusion_matrix(y_test, predictions2)
print("Confusion matrix:\n", conf_mat2)
class_rep2 = classification_report(y_test, predictions2)
print("\nClassification Report:\n", class_rep2)