import pandas as pd
import pickle
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from sklift.metrics import uplift_at_k
from sklift.models import SoloModel
from sklift.models import TwoModels
from xgboost import XGBClassifier
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

inp_d = pd.read_parquet(r'c:\AI2024\train_for_1211\for_f\video_stat.parquet')
print(inp_d)
x_train = inp_d.drop(['description','v_pub_datetime','row_number','title'],axis=1)
print(len(inp_d))
# _train[i] < mean + threshold) & (x_train[i] > mean - threshold)]
print(len(x_train))
# print('abcdef'[-3:])
multi_feat = set()

dict1 = dict()

cnt1 = 0
for i in x_train['category_id'].unique():
    dict1[i] = cnt1
    cnt1 += 1
x_train['category_id'] = x_train['category_id'].apply(lambda x: dict1[x])

dict_items = dict()
cnt1 = 0
for i in x_train['video_id'].unique():
    dict_items[i] = cnt1
    cnt1 += 1
x_train['video_id'] = x_train['video_id'].apply(lambda x: dict_items[x])

dict_authors = dict()
cnt1 = 0
for i in x_train['author_id'].unique():
    dict_authors[i] = cnt1
    cnt1 += 1
x_train['author_id'] = x_train['author_id'].apply(lambda x: dict_authors[x])

for i in x_train.keys():
    t1 = i.split(sep='_')
    t2 = ''
    if t1[-1] == 'days':
        for i in t1[:-2]:
            t2 += i +'_'
        multi_feat.add(t2)
        print(t2)
multi_feat.remove('v_category_popularity_percent_')
print(multi_feat)

for i in multi_feat:
    x_train[i+'30_'+'days'] = x_train[i+'30_'+'days'] - x_train[i+'7_'+'days']
    x_train[i+'7_'+'days'] = x_train[i+'7_'+'days'] - x_train[i+'1_'+'days']
x_train['v_year_views'] = x_train['v_year_views'] - x_train['v_month_views']
x_train['v_month_views'] = x_train['v_month_views'] - x_train['v_week_views']
x_train['v_week_views'] = x_train['v_week_views'] - x_train['v_day_views']
print(x_train)
for id,row in x_train.iterrows():
    print(row)
    break

with open(r'c:\AI2024\train_for_1211\for_f\videos', 'wb') as f:
     pickle.dump(x_train, f)
with open(r'c:\AI2024\train_for_1211\for_f\dictvid', 'wb') as f:
     pickle.dump(dict_items, f)
with open(r'c:\AI2024\train_for_1211\for_f\dictauthors', 'wb') as f:
     pickle.dump(dict_authors, f)