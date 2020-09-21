
from sklearn import preprocessing, metrics
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("alpha-recognition.csv")
df = df.apply(preprocessing.LabelEncoder().fit_transform)

df = np.array(df)

X = df[0:16000, 1:]
y = df[0:16000, 0]


knn_clf = KNeighborsClassifier()


K = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

for i in range(12):
     knn_clf = KNeighborsClassifier(n_neighbors=K[i])
     knn_clf.fit(X, y)
     predicted = knn_clf.predict(df[16000:, 1:])  # when printing np.ndarray, no coma between members
     expected = df[16000:, 0]
     report = metrics.classification_report(expected, predicted)
     print (report)
     print
