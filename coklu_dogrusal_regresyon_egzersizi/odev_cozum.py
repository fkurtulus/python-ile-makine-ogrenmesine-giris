#kutuphanelerin yuklenmesi

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#veri yukleme

veriler = pd.read_csv("odev_tenis.csv")

#veri onisleme
#encoder: kategorik, numeric donusumu

veriler_numeric = veriler.apply(LabelEncoder().fit_transform)

ilkkolon_ohe = veriler_numeric.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ilkkolon_ohe=ohe.fit_transform(ilkkolon_ohe).toarray()

havadurumu = pd.DataFrame(data=ilkkolon_ohe, index=range(14), columns=["overcast","rainy","sunny"])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]], axis = 1)
sonveriler = pd.concat([veriler_numeric.iloc[:,-2:], sonveriler], axis = 1)

#verilerin egitim ve test kumeleri icin bolunmesi

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state = 0)

#nem oraninini tahmin ettirtme

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
tahmin = regressor.predict(x_test)

#backward elimination (geri eleme) yontemi

import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

#p degeri en yuksek olan ilk kolonu atiyoruz

sonveriler = sonveriler.iloc[:,1:]

import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

#windy kolonunu egitim ve test kumemizden cikartiyoruz

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

#backward elimination yontemi ile tahmin ettirtme

regressor.fit(x_train,y_train)
tahmin_be = regressor.predict(x_test)