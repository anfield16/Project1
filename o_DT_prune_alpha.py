from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

from helpers import dtclf_pruned
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
df = pd.read_csv("alpha-recognition.csv") # !!!type is dataframe, not ndarray!!
# print 'shape of data: ', df.shape
df = np.array(df)  # type conversion needed to use slicing
# print type(df2)
# print df2[:1,:]
X = df[0:16000, 1:]
y = df[0:16000, 0]
for j,alpha in enumerate([-1000,-0.1,-0.01,-0.001,-0.0001,0,0.0001, 0.01,0.1,10]):
#for j, alpha in enumerate([-9999, -0.1, -0.01, -0.001, -0.0001, 0, 0.0001, 0.01, 0.01, 0.25]):
    boost = dtclf_pruned(alpha=alpha)
    boost.fit(X, y)
    predicted = boost.predict(df[16000:, 1:])
    expected = df[16000:, 0]
    print('Booster number {}'.format(j))
    print('There are {} nodes'.format(boost.numNodes()))
    print("Classification report for classifier %s:\n%s\n"
          % (boost, metrics.classification_report(expected, predicted)))