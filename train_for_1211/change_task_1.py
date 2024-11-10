# Сжимает зависимые значения
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
col_nan = dict()




d_tr1 = pd.read_csv('c:/AI2024/train_for_1211/for_e/train.csv',nrows=1000)
# t1 = d_tr1.drop(['successful_utilization',  'treatment'],axis=1)
t1 = d_tr1
l_strs = []
for i in t1.dtypes.keys():
    if t1.dtypes[i]==pd.StringDtype:
        l_strs.append(i)
t1 = t1.drop(l_strs,axis=1)
nans = pd.isna(t1)
bins_cols = []
cat_cols = []
just_cols = []

params_col = dict()

def normal(mn,mx,val):
    return (val-mn)/(mx-mn)

min_nan_col = []
m1 = 100000000000
cat1 = []
j1 = []
for col in t1.keys():
    
    if True in nans[col].value_counts().keys():
        col_nan[col] = nans[col].value_counts()[True]
    else:
        col_nan[col] = 0
    if col_nan[col] > 10000:
        min_nan_col.append(col)
    if len(t1[col].unique()) < 100:
        t1[col] = t1[col].astype('category')
        cat1.append(col)
        dict1 = dict()
        cnt1 = 0
        for u in t1[col].unique():
            dict1[u] = cnt1
            cnt1+=1
        t1[col] = t1[col].apply(lambda x: dict1[x]) 
        if len(t1[col].unique())==2:
            params_col[col] = ['bin']
        else:
            
            params_col[col] = ['cat']
    else:
        params_col[col] = ['just']
        j1.append(col)
    params_col[col].append(col_nan[col]/len(nans))
names = []
cnts_nan = []
b1 = False
for i in params_col.keys():
    b1 = False
    for j in range(0,len(names)):
        if col_nan[i] <= cnts_nan[j]:
            b1 =True
            if col_nan[i] == cnts_nan[j]:
                names[j].append(i) 
                break
            cnts_nan.insert(j,col_nan[i])
            names.insert(j,[i])
            break
    if not b1:
        names.append([i])
        cnts_nan.append(col_nan[i])
# for i in range(len(names)):
#     print(i,names[i],':',cnts_nan[i])
n = 27
# for i in names[n]:
#     print(i,':',params_col[i][0])
# from sklearn.manifold import MDS
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns
# scaler = StandardScaler()
# d_tr11 = pd.DataFrame(data = scaler.fit_transform(d_tr1[j1]),
#  columns = j1)
# d_tr11 = d_tr11.fillna(0)
# kmeans = KMeans(n_clusters = 10)
# cluster = kmeans.fit_predict(d_tr11)
# mds2D = MDS(n_components=2)

# mds_data2D = mds2D.fit_transform(d_tr11)
# mds2D_df = pd.DataFrame(data =  mds_data2D, columns = ['x', 'y'])

# mds2D_df['cluster'] = cluster

# sns.scatterplot(x='x', y='y', hue='cluster', data=mds2D_df)
# plt.title("MDS")
# # plt.show()
addictions = []
names_feat = list(t1.keys())
d_feat_to_ind = dict()
for i in range(0,len(names_feat)):
    d_feat_to_ind[names_feat[i]] = i

d1 = pd.DataFrame()
x1 = pd.DataFrame()
y1 = pd.DataFrame()
m1 = LGBMRegressor()
# cnt1 = 0
# for i in t1.keys():
#     print(cnt1/len(t1.keys()))
#     cnt1+=1
#     if i in cat1:
#         m1 = LGBMClassifier(n_estimators=25)
#     else:
#         m1 = LGBMRegressor(n_estimators=25)
#     d1 = t1.dropna(subset=[i])
#     x1 = d1.drop(i,axis=1)
#     y1 = d1[i]
#     m1.fit(x1,y1) 
#     addictions.append([i]+[-10]*len(t1.keys()))
#     for j in range(0,len(m1.feature_importances_)):
#         # print(len(addictions[len(addictions)-1]))
#         # print(d_feat_to_ind[m1.feature_name_[j]])
#         addictions[len(addictions)-1][d_feat_to_ind[m1.feature_name_[j]]+1] = m1.feature_importances_[j]
# for i in addictions:
#     for j in i:
#         print(j,end=' ')
#     print()
# c1 = ['name']+ names_feat
# d_feature_imp = pd.DataFrame(addictions,columns=c1)
# d_feature_imp.to_excel(r'c:/AI2024/train_for_1211/for_e/features_addiction.xlsx')

bki_train = t1[names[n]].dropna()
# bki_x = bki_train.drop(names[n][0],axis=1)
# bki_y = bki_train[names[n][0]]
bki_x = bki_train.drop('bki_27',axis=1)
bki_y = bki_train['bki_27']
# mx1 = bki_y.max()
# mn1 = bki_y.min()

# bki_y = bki_y.apply(lambda x: normal(mn1,mx1,x))
# print('//////////')
# print(bki_train['bki_45'].unique())
# print('//////////')
a1 = np.array(bki_y.unique())
d1 = dict()
for i in range(0,len(a1)):
    d1[a1[i]] = i
bki_y = bki_y.apply(lambda x: d1[x])



x_t_b,x_v_b,y_t_b,y_v_b = train_test_split(bki_x,bki_y,train_size=0.1)

# print(bki_y)

# bki_model = LGBMClassifier(n_estimators=5)

# bki_model.fit(x_t_b,y_t_b)
# pr1 = bki_model.predict(x_v_b)
# sc1 = 0.0
# y_v_b = np.array(y_v_b)
# for i in range(0,len(pr1)):
#     if pr1[i]==y_v_b[i]:
#         sc1+=1
#     # sc1 += abs(pr1[i]-y_v_b[i])



# print(sc1/len(pr1),y_v_b.mean())
# print(list(bki_model.feature_importances_))

compact_train = pd.DataFrame()
compact_test = pd.DataFrame()
d_test = pd.read_csv('c:/AI2024/train_for_1211/for_e/test.csv')

for i in names:
    compact_train[i[0]] = d_tr1[i[0]]
for i in names:
    compact_test[i[0]] = d_test[i[0]]
print(len(compact_train.keys()),len(d_tr1.keys()))
t1 = d_tr1.drop(['successful_utilization',  'treatment'],axis=1)
compact_train['successful_utilization'] = d_tr1['successful_utilization']
compact_train['treatment']=d_tr1['treatment']
compact_train.to_csv('c:/AI2024/train_for_1211/for_e/compact_train.csv',index=False)
compact_test.to_csv('c:/AI2024/train_for_1211/for_e/compact_test.csv',index=False)
# print('cat')
# for i in range(0,len(bki_model.feature_importances_)):
#     print(bki_model.feature_name_[i],bki_model.feature_importances_[i])
    # print(col,':',int((col_nan[col]/len(nans))*100),'% unique_vals:',len(t1[col].unique()))
