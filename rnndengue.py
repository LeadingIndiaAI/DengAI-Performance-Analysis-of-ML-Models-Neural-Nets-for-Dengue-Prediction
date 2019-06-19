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

Y1=df_sjlb.iloc[:,0:4].values
Y2=df_iqlb.iloc[:,0:4].values

y1=Y1[:,3:4]
y2=Y2[:,3:4]

#for spiliting the data in dependent and independent form
X1=df_sj.iloc[:,0:24].values
X2=df_iq.iloc[:,0:24].values

#-------------------------------for sj city training dataset---------------------------- 
#labelencodig to the training dataset
from sklearn.preprocessing import LabelEncoder
lb_X1=LabelEncoder()
X1[:,0]=lb_X1.fit_transform(X1[:,0])

#for removing the null values
from sklearn.preprocessing import Imputer
impute=Imputer(missing_values="NaN" ,strategy="mean" ,axis=0)
X1[:,3:24]=impute.fit_transform(X1[:,3:24])

#----------------------------------for iq city training dataset---------------------------
#labelencodig to the training dataset
from sklearn.preprocessing import LabelEncoder
lb_X2=LabelEncoder()
X2[:,0]=lb_X2.fit_transform(X2[:,0])

#for removing the null values
from sklearn.preprocessing import Imputer
impute=Imputer(missing_values="NaN" ,strategy="mean" ,axis=0)
X2[:,3:24]=impute.fit_transform(X2[:,3:24])

#-----------------------------------testing dataset-------------------------------------
#loading the testing dataset
test=pd.read_csv("dengue_features_test.csv")

del test['week_start_date']
df_testsj = test[test['city'] == 'sj']
df_testiq = test[test['city'] == 'iq']


x_test1=df_testsj.iloc[:,0:24].values
x_test2=df_testiq.iloc[:,0:24].values

#------------------------------------testing for sj city dataset---------------------------
#labelencodig to the training dataset
from sklearn.preprocessing import LabelEncoder
lb_test1=LabelEncoder()
x_test1[:,0]=lb_test1.fit_transform(x_test1[:,0])


#for removing the null values
from sklearn.preprocessing import Imputer
imput=Imputer(missing_values="NaN" ,strategy="mean" ,axis=0)
x_test1[:,3:24]=imput.fit_transform(x_test1[:,3:24])

#--------------------------------------testing for iq city dataset-------------------------
#labelencodig to the training dataset
from sklearn.preprocessing import LabelEncoder
lb_test2=LabelEncoder()
x_test2[:,0]=lb_test2.fit_transform(x_test2[:,0])


#for removing the null values
from sklearn.preprocessing import Imputer
imput=Imputer(missing_values="NaN" ,strategy="mean" ,axis=0)
x_test2[:,3:24]=imput.fit_transform(x_test2[:,3:24])

def dataset_generate(data, step_size=1):
    dataX, dataY = [], []
    for i in range(len(data)- step_size -1):
       a = data[i:(i+ step_size), 0]
       dataX.append(a)
       dataY.append(data[i + step_size, 0])
    return np.array(dataX), np.array(dataY)


 #Reshape into X1=t and y1=t+1
 step_size = 1
 trainsj, trainysj = dataset_generate(X1, step_size)
 trainiq, trainYiq = dataset_generate(X2, step_size)

#--------------------------scaling sj train city dataset along with the labels-------------
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(X1)
label_scaled=sc.fit_transform(y1)

# Reshaping
X1 = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))

#------------------------building rnn model for sj city----------------------------------

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 150, return_sequences = True, input_shape = (X1.shape[1], 1)))
regressor.add(Dropout(0.15))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 150, return_sequences = True))
regressor.add(Dropout(0.15))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 150, return_sequences = True))
regressor.add(Dropout(0.15))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 150))
regressor.add(Dropout(0.15))

# Adding the output layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error')

# Fitting the RNN to the Training set
regressor.fit(X1, y1, epochs = 25, batch_size = 32)

x_test1= np.reshape(x_test1, (x_test1.shape[0], x_test1.shape[1], 1))

predicted_sj = regressor.predict(x_test1)
predicted_sj = sc.inverse_transform(predicted_sj)

predsj=predicted_sj.round()

#----------------------------Feature Scaling for iq city train dataset along with labels----------------------
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(X2)
label_scaled=sc.fit_transform(y2)

# Reshaping
X2 = np.reshape(X2, (X2.shape[0], X2.shape[1], 1))

#------------------------building rnn model for iq city----------------------------------
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor1 = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
regressor1.add(LSTM(units = 150, return_sequences = True, input_shape = (X2.shape[1], 1)))
regressor1.add(Dropout(0.15))
# Adding a second LSTM layer and some Dropout regularisation
regressor1.add(LSTM(units = 150, return_sequences = True))
regressor1.add(Dropout(0.15))

# Adding a third LSTM layer and some Dropout regularisation
regressor1.add(LSTM(units = 150, return_sequences = True))
regressor1.add(Dropout(0.15))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor1.add(LSTM(units = 150))
regressor1.add(Dropout(0.15))

# Adding the output layer
regressor1.add(Dense(units = 1))

# Compiling the RNN
regressor1.compile(optimizer = 'adam', loss = 'mean_absolute_error')

# Fitting the RNN to the Training set
regressor1.fit(X2, y2, epochs = 30, batch_size = 32)

x_test2=np.reshape(x_test2, (x_test2.shape[0], x_test2.shape[1], 1))

#------------------------------prediction for iq city testset data--------------------------
predicted_iq = regressor1.predict(x_test2)
predicted_iq = sc.inverse_transform(predicted_iq)

prediq=predicted_iq.round()
