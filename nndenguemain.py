import numpy as np
import pandas as pd
import matplotlib as plt
dataset=pd.read_csv('dengue_features_train.csv')
datas=pd.read_csv('dengue_labels_train.csv')

del dataset['week_start_date']


df_sj = dataset[dataset['city'] == 'sj']
df_iq = dataset[dataset['city'] == 'iq']

df_sjlb=datas[datas['city'] == 'sj']
df_iqlb=datas[datas['city'] == 'iq']

#for spiliting the data in dependent and independent form
X1=df_sj.iloc[:,0:24].values
X2=df_iq.iloc[:,0:24].values

Y1=df_sjlb.iloc[:,0:4].values
Y2=df_iqlb.iloc[:,0:4].values

y1=Y1[:,3:4]
y2=Y2[:,3:4]

# SL = 0.05 and eliminating those features which have p > SL


#labelencodig to the training dataset
from sklearn.preprocessing import LabelEncoder
lb_X1=LabelEncoder()
X1[:,0]=lb_X1.fit_transform(X1[:,0])

#for removing the null values
from sklearn.preprocessing import Imputer
impute=Imputer(missing_values="NaN" ,strategy="mean" ,axis=0)
X1[:,3:24]=impute.fit_transform(X1[:,3:24])

from sklearn.preprocessing import StandardScaler
sc_X1=StandardScaler()
X1=sc_X1.fit_transform(X1)

#labelencodig to the training dataset
from sklearn.preprocessing import LabelEncoder
lb_X2=LabelEncoder()
X2[:,0]=lb_X2.fit_transform(X2[:,0])

#for removing the null values
from sklearn.preprocessing import Imputer
impute=Imputer(missing_values="NaN" ,strategy="mean" ,axis=0)
X2[:,3:24]=impute.fit_transform(X2[:,3:24])

from sklearn.preprocessing import StandardScaler
sc_X2=StandardScaler()
X2=sc_X2.fit_transform(X2)


#loading the testing dataset
test=pd.read_csv("dengue_features_test.csv")

del test['week_start_date']
df_testsj = test[test['city'] == 'sj']
df_testiq = test[test['city'] == 'iq']


x_test1=df_testsj.iloc[:,0:24].values
x_test2=df_testiq.iloc[:,0:24].values


#labelencodig to the training dataset
from sklearn.preprocessing import LabelEncoder
lb_test1=LabelEncoder()
x_test1[:,0]=lb_test1.fit_transform(x_test1[:,0])


#for removing the null values
from sklearn.preprocessing import Imputer
imput=Imputer(missing_values="NaN" ,strategy="mean" ,axis=0)
x_test1[:,3:24]=imput.fit_transform(x_test1[:,3:24])

from sklearn.preprocessing import StandardScaler
sc_X11=StandardScaler()
x_test1=sc_X11.fit_transform(x_test1)


#labelencodig to the training dataset
from sklearn.preprocessing import LabelEncoder
lb_test2=LabelEncoder()
x_test2[:,0]=lb_test2.fit_transform(x_test2[:,0])


#for removing the null values
from sklearn.preprocessing import Imputer
imput=Imputer(missing_values="NaN" ,strategy="mean" ,axis=0)
x_test2[:,3:24]=imput.fit_transform(x_test2[:,3:24])

from sklearn.preprocessing import StandardScaler
sc_X22=StandardScaler()
x_test2=sc_X22.fit_transform(x_test2)


#creating the ann model
import keras
from keras import Sequential
from keras.layers import Dense
def build_regressor1():
    regressor1 = Sequential()
    regressor1.add(Dense(units=256, input_dim=23, activation="relu"))
    regressor1.add(Dense(units=256, activation="relu"))
    regressor1.add(Dense(units=256, activation="relu"))
    regressor1.add(Dense(units=1))
    regressor1.compile(optimizer='Adam', loss='mean_absolute_error',  metrics=['mae','accuracy'])
    return regressor1

from keras.wrappers.scikit_learn import KerasRegressor
regressor1 = KerasRegressor(build_fn=build_regressor1, batch_size=32,epochs=300)

result1=regressor1.fit(X1,y1)

y_predsj=regressor1.predict(x_test1)

def build_regressor2():
    regressor2 = Sequential()
    regressor2.add(Dense(units=256, input_dim=23, activation="relu"))
    regressor2.add(Dense(units=256, activation="relu"))
    regressor2.add(Dense(units=256, activation="relu"))
    regressor2.add(Dense(units=1))
    regressor2.compile(optimizer='Adam', loss='mean_absolute_error',  metrics=['mae','accuracy'])
    return regressor2

from keras.wrappers.scikit_learn import KerasRegressor
regressor2 = KerasRegressor(build_fn=build_regressor2, batch_size=32,epochs=250)

result2=regressor2.fit(X2,y2)

y_prediq=regressor2.predict(x_test2)

ypredsj=y_predsj.round()

yprediq=y_prediq.round()

