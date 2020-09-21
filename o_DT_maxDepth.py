

from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("alpha-recognition.csv")

df = np.array(df)


max_depth = np.arange(1,41,3)
X = df[0:16000, 1:]
y = df[0:16000, 0]
score = []
for i in range(14):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth[i])
    clf = clf.fit(X, y)
    expected = df[16000:, 0]
    predicted = clf.predict(df[16000:, 1:])
    score=np.append(score, accuracy_score(expected, predicted))

plt.figure()

plt.scatter(max_depth, score, c="darkorange")
plt.plot(max_depth, score, color="yellowgreen", label="accuracy_score", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Max Depth on DT accuracy")
plt.legend()
plt.show()



