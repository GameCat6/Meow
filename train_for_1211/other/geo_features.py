import pandas as pd
# from sklearn.datasets import load_boston
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.datasets import fetch_california_housing
# housing = fetch_california_housing()
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
# Загрузка данных
# boston = load_boston()
# df = pd.DataFrame(housing.data, columns=housing.feature_names)
df = pd.read_csv(r'c:\AI2024\train_for_1211\other\housing.csv')
print(df.keys(),df)
# print(df['AveBedrms'].max())
print(df['ocean_proximity'].unique())
df['total_rooms'] = df['total_rooms']-df['total_bedrooms']
df['av_rooms'] = df['total_rooms'] / df['households']
df['av_bedrooms'] = df['total_bedrooms'] / df['households']
df['av_po'] = df["population"] / df['households']

df = df.drop(['total_rooms',"population",'total_bedrooms'],axis=1)

dict1 = {'<1H OCEAN':1,'NEAR BAY':2,'INLAND':0,'NEAR OCEAN':3,'ISLAND':4}
df['ocean_proximity'] = df['ocean_proximity'].apply(lambda x: dict1[x])
for i in ['av_rooms','av_bedrooms','av_po',"housing_median_age",'median_house_value']:
    mean = df[i].mean()
    threshold = 4.5*df[i].std()
    df = df[(df[i] < mean + threshold) & (df[i] > mean - threshold)]


# df['bogat']=df['median_house_value'].apply(lambda x:x if x >df['median_house_value'].median() else 0)
# df.plot(kind="scatter",x="longitude", y='latitude',s=(df['bogat']/df['bogat'].max()),alpha=0.1)
# df[df['median_house_value']>df['median_house_value'].median()]['median_house_value'].hist(bins=100)
x,y,x1,y1 = -118.36,34.15,-121.84,37.31
df['to1'] = ((df['longitude']-x)**2+(df['latitude']-y)**2)**0.5
df['to2'] = ((df['longitude']-x1)**2+(df['latitude']-y1)**2)**0.5
df.plot(kind="scatter",x="longitude", y='latitude',c=(df['ocean_proximity']/4)*100,s=(df['households']/df['households'].max())*10,alpha=0.4)

plt.show()
print(df,df.keys())
X = df.drop(["housing_median_age",'ocean_proximity'], axis=1)
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)
l1 = LGBMRegressor()
l1 = l1.fit(X_train,y_train)
s1 = 0
s1 = (abs(l1.predict(X_test)-y_test)/y_test)
print(s1.sum()/len(X_test))