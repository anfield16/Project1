import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import model_selection, preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv("student-prf.csv", sep=';', header=0)

df = df.apply(preprocessing.LabelEncoder().fit_transform)
df = np.array(df)

all_column = np.arange(25)
all_column = np.append(all_column, [28, 29, 30, 31])
X = df[:, all_column]
y = df[:, 26]

seed = 5

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
fig.suptitle('Algorithm Comparison(letter_recognition)')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel("running time(sec)")
plt.show()