from sklearn import metrics, preprocessing

from helpers import dtclf_pruned
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

df = pd.read_csv("student-prf.csv", sep=';', header=0)


df = df.apply(preprocessing.LabelEncoder().fit_transform)
df = np.array(df)  # type conversion needed to use slicing

all_column = np.arange(25)  # select feature for prediction
all_column = np.append(all_column, [28, 29, 30, 31])
X = df[:400, all_column]
y = df[:400, 26]
#for j,alpha in enumerate([-99999, -1,-0.01,-0.0001, 0,0.01,0.25]):
for j,alpha in enumerate([-1,-0.01,0,0.01,0.1]):
    boost = AdaBoostClassifier(dtclf_pruned(alpha=alpha),n_estimators=5)
    boost.fit(X, y)
    predicted = boost.predict(df[400:, all_column])
    expected = df[400:, 26]
    print('Booster number {}'.format(j))
    for i,dt in enumerate(boost.estimators_):
        print('pruned tree {}. Alpha is {}. There are {} nodes'.format(i+1,dt.alpha,dt.numNodes()))
    print("Classification report for classifier %s:\n%s\n"
          % (boost, metrics.classification_report(expected, predicted)))