from sklearn import metrics

from helpers import dtclf_pruned
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
df = pd.read_csv("alpha-recognition.csv")

df = np.array(df)

X = df[0:16000, 1:]
y = df[0:16000, 0]
for j,alpha in enumerate([-1,-0.01,0,0.01,0.1]):
#for j,alpha in enumerate([-1e-3,-1e-4,-1e-5,-1e-6,0,1e-6,1e-5,1e-4]):
    boost = AdaBoostClassifier(dtclf_pruned(alpha=alpha),n_estimators=5)
    boost.fit(X, y)
    predicted = boost.predict(df[16000:, 1:])
    expected = df[16000:, 0]
    print('Booster number {}'.format(j))
    for i,dt in enumerate(boost.estimators_):
        print('pruned tree {}. Alpha is {}. There are {} nodes'.format(i+1,dt.alpha,dt.numNodes()))
    print("Classification report for classifier %s:\n%s\n"
          % (boost, metrics.classification_report(expected, predicted)))