import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
models.append(('NEURO', MLPClassifier()))
models.append(('BOOST', AdaBoostClassifier()))
models.append(('SVM', SVC()))
models.append(('KNN', KNeighborsClassifier()))


results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison(Student performance)')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel("Accuracy")
plt.show()