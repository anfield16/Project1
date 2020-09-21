
from sklearn import preprocessing, metrics
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("student-prf.csv", sep=';', header=0)

df = df.apply(preprocessing.LabelEncoder().fit_transform)
df = np.array(df)

selected_column = np.arange(25)
selected_column = np.append(selected_column, [28, 29])
X = df[:400, selected_column]
y = df[:400, 26]
K = np.arange(1, 23, 2)
K = np.append(K,30)

for i in range(12):
    knn_clf = KNeighborsClassifier(n_neighbors=K[i])
    knn_clf.fit(X, y)
    predicted = knn_clf.predict(df[400:, selected_column])
    expected = df[400:, 26]
    report = metrics.classification_report(expected, predicted)
    print (report)
    print
# average is manually collected from the print statement
precision = [0.54, 0.53, 0.53, 0.53, 0.49, 0.50, 0.54, 0.63, 0.63, 0.63, 0.45, 0.45]
recall =    [0.57, 0.66, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67]
