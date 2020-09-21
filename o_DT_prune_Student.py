from sklearn import metrics, preprocessing
from helpers import dtclf_pruned
import pandas as pd
import numpy as np
df = pd.read_csv("student-prf.csv", sep=';', header=0) 

df = df.apply(preprocessing.LabelEncoder().fit_transform)
df = np.array(df)

all_column = np.arange(25)
all_column = np.append(all_column, [28, 29, 30, 31])
X = df[:400, all_column]
y = df[:400, 26]
for j,alpha in enumerate([-1000,-0.1,-0.01,-0.001,-0.0001,0,0.0001, 0.01, 0.01,0.1]):
    boost = dtclf_pruned(alpha=alpha)
    boost.fit(X, y)
    predicted = boost.predict(df[400:, all_column])
    expected = df[400:, 26]
    print('Booster number {}'.format(j))
    print('There are {} nodes'.format(boost.numNodes()))
    print("Classification report for classifier %s:\n%s\n"
          % (boost, metrics.classification_report(expected, predicted)))