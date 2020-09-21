

from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("alpha-recognition.csv")

df = np.array(df)


min_samples_split = np.arange(2,61,2)
X = df[0:16000, 1:]
y = df[0:16000, 0]
score = []
for i in range(len(min_samples_split)):
    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split[i])
    clf = clf.fit(X, y)
    expected = df[16000:, 0]
    predicted = clf.predict(df[16000:, 1:])
    score=np.append(score, accuracy_score(expected, predicted))

plt.figure()

plt.scatter(min_samples_split, score, c="darkorange")
plt.plot(min_samples_split, score, color="yellowgreen", label="accuracy_score", linewidth=2)
plt.xlabel("Min Samples Split")
plt.ylabel("accuracy")
plt.title("Min Samples Split on DT Accuracy")
plt.legend()
plt.show()



