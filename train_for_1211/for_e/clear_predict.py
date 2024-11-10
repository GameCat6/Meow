import pandas as pd
from sklift.models import ClassTransformation
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
import matplotlib.pyplot as plt
# def hampel(vals_orig):
#     vals = vals_orig.copy()    
#     difference = np.abs(vals.median()-vals)
#     median_abs_deviation = difference.mean()
#     threshold = 5 * median_abs_deviation
#     outlier_idx = difference > threshold
#     vals[outlier_idx] = np.nan
#     return(vals)

d1 = pd.read_csv('c:/AI2024/train_for_1211/for_e/train.csv')
X_train = d1
# X_train = X_train.drop(['treatment','successful_utilization'],axis=1)
# cat_features = []

# cnt_tret = 0
# cnt_target = 0

# def f2(x):
#     global cnt_tret
#     if x == 1:
#         cnt_tret+=1
# def f2_1(x):
#     global cnt_target
#     if x==1:
#         cnt_target +=1
# d1_targ_1 = d1[d1['successful_utilization']==1]
# d1_targ_0 = d1[d1['successful_utilization']==0]

# d1['successful_utilization'].apply(f2_1)
# d1['treatment'].apply(f2)
# print(cnt_target/len(d1),cnt_tret/len(d1))
# cnt_target = 0
# cnt_tret =0

# d1_targ_0['successful_utilization'].apply(f2_1)
# d1_targ_0['treatment'].apply(f2)
# print(cnt_target/len(d1_targ_0),cnt_tret/len(d1_targ_0))
# cnt_target = 0
# cnt_tret =0

# d1_targ_1['successful_utilization'].apply(f2_1)
# d1_targ_1['treatment'].apply(f2)

# print(len(d1_targ_0),len(d1_targ_1))

# print(cnt_target/len(d1_targ_1),cnt_tret/len(d1_targ_1))
# cnt_target = 0
# cnt_tret =0

# d1_s_1_t_1  = d1_targ_1[d1_targ_1['treatment']==1]
# d1_s_1_t_0  = d1_targ_1[d1_targ_1['treatment']==0]

# d1_s_0_t_1  = d1_targ_0[d1_targ_0['treatment']==1]
# d1_s_0_t_0  = d1_targ_0[d1_targ_0['treatment']==0]

# # d1_s_1_t_1 = d1_s_1_t_1.sample(frac=1).reset_index(drop=True)
# # d1_s_0_t_1 = d1_s_0_t_1.sample(frac=1).reset_index(drop=True)
# # d1_s_1_t_0 = d1_s_1_t_0.sample(frac=1).reset_index(drop=True)
# # d1_s_0_t_0 = d1_s_0_t_0.sample(frac=1).reset_index(drop=True)


# print(len(d1_s_0_t_0),len(d1_s_0_t_1),len(d1_s_1_t_0),len(d1_s_1_t_1))



# j1 = []

# # m1 = min(len(d1_s_0_t_0),len(d1_s_0_t_1),len(d1_s_1_t_0),len(d1_s_1_t_1))
# # X_train = pd.concat([d1_s_0_t_0[:m1],d1_s_0_t_1[:m1],d1_s_1_t_0[:m1],d1_s_1_t_1[:m1]])

# # m1 = 12163
# # X_train = pd.concat([d1_s_0_t_0[:m1],d1_s_0_t_1[:m1],d1_s_1_t_0,d1_s_1_t_1[:m1]])

# m1 = 68767
# X_train = pd.concat([d1_s_0_t_0,d1_s_0_t_1[:m1],d1_s_1_t_0,d1_s_1_t_1[:m1]])


# X_train = X_train.sample(frac=1).reset_index(drop=True)


cat_features = []
j1 = []

# X_train = pd.read_csv('c:/AI2024/train_for_1211/for_e/b_train.csv')

# del X_train['Unnamed: 0']
m1 = -1
nm1 = '' 
for i in X_train.keys():
    if not i in ['treatment','successful_utilization']:
        if len(X_train[i].unique()) <= 100:
            cat_features.append(i)
        else:
            j1.append(i)
            if m1 < len(X_train[i].unique()):
                nm1 = i
                m1 = len(X_train[i].unique())



for i in cat_features:
    X_train[i] = X_train[i].fillna(-99999)
    X_train[i] = X_train[i].astype('category')
# X_train = X_train.drop(cat_features,axis=1)
X_train = X_train.fillna(0)



print(len(X_train))
def f1(x,m1,q1,q3,iqr):
    if (x < q3 + iqr) & (x > q1 - iqr):
        return m1
    else:
        return x
t1 = len(X_train)


# import seaborn as sns
# print(X_train['successful_utilization'].dtype)
# print(X_train[['successful_utilization']+j1].corr())

# cor = X_train[['successful_utilization']+j1[:10]].corr()

# plt.hist(x=cor.keys(),y = cor['successful_utilization'] )

# # sns.heatmap()
# plt.show()

X_train['bki_5'].sort_values().hist(bins=500)
plt.show()

for i in j1:
    mean = X_train[i].mean()
    threshold = 4.5*X_train[i].std()
    X_train = X_train[(X_train[i] < mean + threshold) & (X_train[i] > mean - threshold)]
    # X_train[i] = hampel(X_train[i])
    # q1 = X_train[i].quantile(0.25)
    # q3 = X_train[i].quantile(0.75)
    # m1 = X_train[i].median()
    # iqr = (q3 - q1)*1.5
    # X_train[i] = X_train[i].apply(lambda x: f1(x,m1,q1,q3,iqr))



X_train = X_train.dropna()
X_train['bki_5'].sort_values().hist(bins=500)
plt.show()
# for i in j1:
#     X_train[i]=np.log1p(X_train[i])

h1 = X_train[j1].hist(bins=3)
# print(h1)
plt.show()






X_train = X_train.fillna(-9999)
print(1-len(X_train)/t1)

y_train = X_train['successful_utilization']
treat_train = X_train['treatment']
X_train = X_train.drop(['treatment','successful_utilization'],axis=1)
# ct = ClassTransformation(LGBMClassifier(n_estimators=100,max_depth=4)) - Для сбалансированного без выбросов
print('Start fit')
# ct = ClassTransformation(LGBMClassifier(n_estimators=1000,max_depth=1)) - несбалансированный без выбросов

ct = ClassTransformation(LGBMClassifier(n_estimators=1000,max_depth=1))
ct = ct.fit(X_train,y_train,treat_train)
print('End fit')
d_test = pd.read_csv('c:/AI2024/train_for_1211/for_e/test.csv')
X_train = X_train.dropna()
for i in cat_features:
    d_test[i] = d_test[i].fillna(-99999)
    d_test[i] = d_test[i].astype('category')
# d_test = d_test.drop(cat_features,axis=1)
d_test = d_test.fillna(0)
# d_test = np.array(d_test)
# d_test = pca.transform(d_test)

y = ct.predict(d_test)
d_res = pd.DataFrame({'successful_utilization': y})
d_res.to_csv('c:/AI2024/train_for_1211/for_e/res_4_log.csv')
# print('\n'*10)
print(r'\/'*30)
print('Predict Created')
print('[]' *30)