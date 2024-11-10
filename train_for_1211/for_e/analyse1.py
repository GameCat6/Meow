import pandas as pd
import numpy as np
d_feature_imp=pd.read_excel(r'c:/AI2024/train_for_1211/for_e/features_addiction.xlsx')
del d_feature_imp['Unnamed: 0']

for i in d_feature_imp.keys():
    if i != 'name':
        print(i,d_feature_imp[i].sum()+10)

