import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklift.metrics import uplift_at_k
from sklift.models import SoloModel
from sklift.models import TwoModels
from xgboost import XGBClassifier
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
d2 = pd.read_csv('c:/AI2024/train_for_1211/for_e/test.csv')
d2['target'] = [1]*len(d2)


d1 = pd.read_csv('c:/AI2024/train_for_1211/for_e/train.csv').drop(['treatment','successful_utilization'],axis=1)

d1 = d1.sample(frac=1)

d1 = d1[:len(d2)]

# print(d1)
d1['target'] = [0]*len(d1)

# print(d1)





d_train = pd.concat([d1,d2])

d_train.reset_index(drop=True, inplace=True)

X = d_train.drop('target',axis=1)
y = d_train['target']

cat_features = []
j1 = []
for i in X.keys():
    if not i in ['treatment','successful_utilization']:
        if len(X[i].unique()) <= 100:
            cat_features.append(i)
        else:
            j1.append(i)
for i in cat_features:
    X[i] = X[i].fillna(-99999)
    X[i] = X[i].astype('category')
    d1[i] = d1[i].fillna(-99999)
    d1[i] = d1[i].astype('category')
    d2[i] = d2[i].fillna(-99999)
    d2[i] = d2[i].astype('category')
# X_train = X_train.drop(cat_features,axis=1)
X = X.fillna(0)
d1 = d1.fillna(0)
d2 = d2.fillna(0)

x_val = X[:1000]
y_val = y[:1000]

X = X[1000:]
y = y[1000:]

mod1 = LGBMClassifier(n_estimators=100)
print(y)
mod1 = mod1.fit(X,y)

y1 = mod1.predict(x_val)

# y_n = np.array(d1['target'])
# y1 = np.array(mod1.predict(d1.drop('target',axis=1)))

y_n = np.array(y_val)
y1 = np.array(y1)

y2 = abs(y_n-y1)
print(y2)
print(1-y2.sum()/len(y1))




