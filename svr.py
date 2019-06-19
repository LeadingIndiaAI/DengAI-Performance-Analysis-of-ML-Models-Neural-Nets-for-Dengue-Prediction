#standard imports
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import time as time

from warnings import filterwarnings
filterwarnings('ignore')

#load dataset
df=pd.read_csv('train_edited12.csv') 
#labels = pd.read_csv('dengue_labels_train.csv')
test = pd.read_csv('test_edited12.csv')

df=df.iloc[:,1:]
test=test.iloc[:,1:]
#fill NaNs - ffill bc it is a timeseries
df.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)
#to datetime
df['week_start_date']=pd.to_datetime(df['week_start_date'])
test['week_start_date']=pd.to_datetime(test['week_start_date'])
#extract month to new column
df['month']=df.week_start_date.dt.month
test['month']=test.week_start_date.dt.month
#merge labels
#df=pd.merge(df, labels, on=['city', 'year', 'weekofyear'])

df.rename(columns={'total_cases_x': 'total_cases'}, inplace=True)

# separate san juan and iquitos
sj = df[df['city']=='sj']
iq = df[df['city']=='iq']

sj_test=test[test['city']=='sj']
iq_test=test[test['city']=='iq']

#value previous week
#train
sj['cases_prev_wk'] = sj['total_cases'].shift(1)
iq['cases_prev_wk'] = iq['total_cases'].shift(1)

#test
sj_test['cases_prev_wk'] = sj_test['total_cases'].shift(1)
iq_test['cases_prev_wk'] = iq_test['total_cases'].shift(1)

#need to make sure no NaNs added when creating moving avg or getting previous week values
sj.fillna(method='bfill', inplace=True)
iq.fillna(method='bfill', inplace=True)

sj_test.fillna(method='bfill', inplace=True)
sj_test.fillna(method='bfill', inplace=True)

#these features performed better
features2=['reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
       'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
       'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent',
       'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c',
       'station_max_temp_c', 'station_min_temp_c', 'total_cases'] 

#TRAIN
X_sj= sj[features2]
Y_sj = sj['total_cases']

X_iq= iq[features2]
Y_iq = iq['total_cases']

#TEST
X_sj_t= sj_test[features2]

X_iq_t= iq_test[features2]

#need to make sure no NaNs added when creating moving avg or getting previous week values
X_sj.fillna(method='bfill', inplace=True)
X_iq.fillna(method='bfill', inplace=True)

X_sj_t.fillna(method='bfill', inplace=True)
X_iq_t.fillna(method='bfill', inplace=True)

##SAN JUAN
train_size = 100
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X_sj,Y_sj)
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)


model_sj=svr.best_estimator_
model_sj

##IQUITOS
train_size = 100
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X_iq,Y_iq)
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

model_iq=svr.best_estimator_
model_iq

#further divide the training set to create a lineplot
sj_train_subtrain = sj.head(800)
sj_train_subtest = sj.tail(sj.shape[0] - 800)

iq_train_subtrain = iq.head(400)
iq_train_subtest = iq.tail(iq.shape[0] - 400)

#create preds
preds_sj= model_sj.predict(sj_train_subtest[features2]).astype(int)
preds_iq=model_iq.predict(iq_train_subtest[features2]).astype(int)
#add to the dataframes
sj_train_subtest['fitted'] = preds_sj
iq_train_subtest['fitted'] = preds_iq
### reset axis
sj_train_subtest.index = sj_train_subtest['week_start_date']
iq_train_subtest.index = iq_train_subtest['week_start_date']
## make plot
figs, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 9))
sj_train_subtest.total_cases.plot(ax=axes[0], label="Actual")
sj_train_subtest.fitted.plot(ax=axes[0], label="Predictions")

iq_train_subtest.total_cases.plot(ax=axes[1], label="Actual")
iq_train_subtest.fitted.plot(ax=axes[1], label="Predictions")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()

## FIT THE MODELS
model_sj.fit(X_sj,Y_sj)
model_iq.fit(X_iq,Y_iq)

#predict for each city using test set
sj_predictions = model_sj.predict(X_sj_t).astype(int)
iq_predictions = model_iq.predict(X_iq_t).astype(int)
