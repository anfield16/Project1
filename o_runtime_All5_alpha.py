import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv("alpha-recognition.csv")

df = np.array(df)

X = df[:1000, 1:]
y = df[:1000, 0]

seed = 7

models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('NEURO', MLPClassifier(max_iter=2000)))
models.append(('BOOST', AdaBoostClassifier()))
models.append(('SVM', SVC()))
models.append(('KNN', KNeighborsClassifier()))


results = []
names = []
scoring = 'accuracy'
for name, model in models:
    start = time.time()
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    end = time.time()
    results.append((end-start))
    names.append(name)
    print (names, results)

fig = plt.figure()
fig.suptitle('Algorithm Comparison(Alphabets)')
ax = fig.add_subplot(111)
plt.plot(results)
plt.ylabel("running time(sec)")

plt.show()