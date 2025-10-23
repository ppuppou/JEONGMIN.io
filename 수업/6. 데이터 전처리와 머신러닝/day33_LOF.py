import pandas as pd 
import numpy as np 
import janitor
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

train = pd.read_csv("../data/credit_fraud/train.csv")
valid = pd.read_csv("../data/credit_fraud/val.csv")
train = train.clean_names()
valid = valid.clean_names()
train = train.drop(['id'], axis = 1)
valid_x = valid.drop(['id', 'class'], axis = 1)
valid_y = valid['class']

minpts = np.round(np.log(train.shape[0])).astype(int)
clf = LocalOutlierFactor(n_neighbors = minpts, contamination = 0.001, novelty = True)
clf.fit(train)

from sklearn.metrics import confusion_matrix, classification_report
from sklearn import set_config

pred_val = clf.predict(valid_x)

valid_y.replace(1, -1, inplace = True)
valid_y.replace(0, 1, inplace = True)
result = pd.DataFrame({'real' : valid_y, 'pred' : pred_val})
confusion_matrix(result.real, result.pred)