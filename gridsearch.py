from sklearn import svm, datasets, metrics, preprocessing
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("student-etoh-por.csv", sep=';', header=0)  # !!!type is dataframe, not ndarray!!
# print 'shape of data: ', df.shape

# preprocessing, change string to int
df = df.apply(preprocessing.LabelEncoder().fit_transform)
df = np.array(df)  # type conversion needed to use slicing

all_column = np.arange(25)  # select feature for prediction
all_column = np.append(all_column, [28, 29, 30, 31])
X = df[:400, all_column]
y = df[:400, 26]
parameters = {'kernel': ('linear', 'rbf', 'poly'), 'gamma': [0.001, 0.0001], 'C': [1, 10]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(X, y)
D=clf.cv_results_
print D


predicted = clf.predict(df[400:, all_column])
expected = df[400:, 26]

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))


