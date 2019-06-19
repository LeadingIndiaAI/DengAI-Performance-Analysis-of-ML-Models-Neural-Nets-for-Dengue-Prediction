#!/usr/bin/env python3
"""
# -*- coding: utf-8 -*-
Created on Sat Jun 15 13:17:18 2019

@author: rohitgupta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_features = pd.read_csv("dengue_features_train.csv")
dataset_labels = pd.read_csv("dengue_labels_train.csv")

sj_features = dataset_features[dataset_features.city == 'sj']
iq_features = dataset_features[dataset_features.city == 'iq']

sj_labels = dataset_labels[dataset_labels.city == 'sj']
iq_labels = dataset_labels[dataset_labels.city == 'iq']

del sj_features['week_start_date']
del iq_features['week_start_date']
del sj_features['city']
del iq_features['city']

sj_features[sj_features['weekofyear'] == 53]
sj_features.drop([139, 451, 763], axis=0, inplace=True)
sj_labels.drop([139, 451, 763], axis=0, inplace=True)

iq_features[iq_features['weekofyear'] == 53]
iq_features.drop([1170, 1430], axis=0, inplace=True)
iq_labels.drop([1170, 1430], axis=0, inplace=True)

iq_features.columns

import seaborn as sns

sns.boxplot(iq_features.ndvi_ne)

iq_features[iq_features.ndvi_ne > 0.5].index
iq_features.drop([1370, 1413], axis=0, inplace=True)
iq_labels.drop([1370, 1413], axis=0, inplace=True)


sns.boxplot(iq_features.ndvi_nw)

sns.boxplot(iq_features.ndvi_se)

iq_features[iq_features.ndvi_se > 0.46].index
iq_features.drop([1261, 1409], axis=0, inplace=True)
iq_labels.drop([1261, 1409], axis=0, inplace=True)

iq_features[iq_features.ndvi_se < 0.05].index
iq_features.drop([1130], axis=0, inplace=True)
iq_labels.drop([1130], axis=0, inplace=True)


sns.boxplot(iq_features.ndvi_sw)

iq_features[iq_features.ndvi_sw > 0.5].index
iq_features.drop([1002], axis=0, inplace=True)
iq_labels.drop([1002], axis=0, inplace=True)


sns.boxplot(iq_features.precipitation_amt_mm)

iq_features[iq_features.precipitation_amt_mm > 155].index
iq_features.drop([1180], axis=0, inplace=True)
iq_labels.drop([1180], axis=0, inplace=True)


sj_features.columns

sns.boxplot(iq_features.reanalysis_air_temp_k)

iq_features[iq_features.reanalysis_air_temp_k > 301].index
iq_features.drop([956, 1209], axis=0, inplace=True)
iq_labels.drop([956, 1209], axis=0, inplace=True)

iq_features[iq_features.reanalysis_air_temp_k < 294.7].index
iq_features.drop([1040], axis=0, inplace=True)
iq_labels.drop([1040], axis=0, inplace=True)


sns.boxplot(iq_features.reanalysis_avg_temp_k)

iq_features[iq_features.reanalysis_avg_temp_k < 295.3].index
iq_features.drop([1085], axis=0, inplace=True)
iq_labels.drop([1085], axis=0, inplace=True)



sns.boxplot(iq_features.reanalysis_dew_point_temp_k)

iq_features[iq_features.reanalysis_dew_point_temp_k < 291.8].index
iq_features.drop([1203], axis=0, inplace=True)
iq_labels.drop([1203], axis=0, inplace=True)



sns.boxplot(iq_features.reanalysis_max_air_temp_k)

sns.boxplot(iq_features.reanalysis_min_air_temp_k)

iq_features[iq_features.reanalysis_min_air_temp_k < 289].index
iq_features.drop([939, 1047, 1243, 1345], axis=0, inplace=True)
iq_labels.drop([939, 1047, 1243, 1345], axis=0, inplace=True)


sns.boxplot(iq_features.reanalysis_precip_amt_kg_per_m2)

iq_features[iq_features.reanalysis_precip_amt_kg_per_m2 > 123].index
iq_features.drop([1385], axis=0, inplace=True)
iq_labels.drop([1385], axis=0, inplace=True)


sns.boxplot(iq_features.reanalysis_relative_humidity_percent)

iq_features[iq_features.reanalysis_relative_humidity_percent < 70].index
iq_features.drop([959, 994], axis=0, inplace=True)
iq_labels.drop([959, 994], axis=0, inplace=True)



sns.boxplot(iq_features.reanalysis_sat_precip_amt_mm)
sns.boxplot(iq_features.reanalysis_specific_humidity_g_per_kg)



sns.boxplot(iq_features.reanalysis_tdtr_k)

sns.boxplot(iq_features.station_avg_temp_c)
iq_features[iq_features.station_avg_temp_c > 29.5].index
iq_features.drop([1078, 1366, 1422], axis=0, inplace=True)
iq_labels.drop([1078, 1366, 1422], axis=0, inplace=True)

iq_features[iq_features.station_avg_temp_c < 25.5].index
iq_features.drop([1042, 1454], axis=0, inplace=True)
iq_labels.drop([1042, 1454], axis=0, inplace=True)



sns.boxplot(iq_features.station_diur_temp_rng_c)

iq_features[iq_features.station_diur_temp_rng_c > 14.8].index
iq_features.drop([1210], axis=0, inplace=True)
iq_labels.drop([1210], axis=0, inplace=True)

iq_features[iq_features.station_diur_temp_rng_c < 7].index
iq_features.drop([1327], axis=0, inplace=True)
iq_labels.drop([1327], axis=0, inplace=True)



sns.boxplot(iq_features.station_max_temp_c)
iq_features[iq_features.station_max_temp_c > 37].index
iq_features.drop([1053, 1359, 1414, 1415], axis=0, inplace=True)
iq_labels.drop([1053, 1359, 1414, 1415], axis=0, inplace=True)


sns.boxplot(iq_features.station_min_temp_c)
iq_features[iq_features.station_min_temp_c > 24].index
iq_features.drop([1229], axis=0, inplace=True)
iq_labels.drop([1229], axis=0, inplace=True)

iq_features[iq_features.station_min_temp_c < 18.7].index
iq_features.drop([1039, 1252], axis=0, inplace=True)
iq_labels.drop([1039, 1252], axis=0, inplace=True)


sns.boxplot(iq_features.station_precip_mm)
iq_features[iq_features.station_precip_mm > 175].index
iq_features.drop([1117, 1129, 1153, 1163, 1173, 1236, 1257, 1276, 1293], axis=0, inplace=True)
iq_labels.drop([1117, 1129, 1153, 1163, 1173, 1236, 1257, 1276, 1293], axis=0, inplace=True)

sj_features.hist()

def fillMissingValues(city,column):
    index = city[column].index[city[column].apply(np.isnan)]
    
    for value in index:
        week = city['weekofyear'][value]
        mean = city[city['weekofyear'] == week][column].mean()
        city[column][value] = mean

sj_features.columns

iq_features.isnull().sum()

fillMissingValues(sj_features, 'ndvi_ne')
fillMissingValues(sj_features, 'ndvi_nw')
fillMissingValues(sj_features, 'ndvi_se')
fillMissingValues(sj_features, 'ndvi_sw')
fillMissingValues(sj_features, 'precipitation_amt_mm')
fillMissingValues(sj_features, 'reanalysis_air_temp_k')
fillMissingValues(sj_features, 'reanalysis_avg_temp_k')
fillMissingValues(sj_features, 'reanalysis_dew_point_temp_k')
fillMissingValues(sj_features, 'reanalysis_max_air_temp_k')
fillMissingValues(sj_features, 'reanalysis_min_air_temp_k')
fillMissingValues(sj_features, 'reanalysis_precip_amt_kg_per_m2')
fillMissingValues(sj_features, 'reanalysis_relative_humidity_percent')
fillMissingValues(sj_features, 'reanalysis_sat_precip_amt_mm')
fillMissingValues(sj_features, 'reanalysis_specific_humidity_g_per_kg')
fillMissingValues(sj_features, 'reanalysis_tdtr_k')
fillMissingValues(sj_features, 'station_avg_temp_c')
fillMissingValues(sj_features, 'station_diur_temp_rng_c')
fillMissingValues(sj_features, 'station_max_temp_c')
fillMissingValues(sj_features, 'station_min_temp_c')
fillMissingValues(sj_features, 'station_precip_mm')

fillMissingValues(iq_features, 'ndvi_ne')
fillMissingValues(iq_features, 'ndvi_nw')
fillMissingValues(iq_features, 'ndvi_se')
fillMissingValues(iq_features, 'ndvi_sw')
fillMissingValues(iq_features, 'precipitation_amt_mm')
fillMissingValues(iq_features, 'reanalysis_air_temp_k')
fillMissingValues(iq_features, 'reanalysis_avg_temp_k')
fillMissingValues(iq_features, 'reanalysis_dew_point_temp_k')
fillMissingValues(iq_features, 'reanalysis_max_air_temp_k')
fillMissingValues(iq_features, 'reanalysis_min_air_temp_k')
fillMissingValues(iq_features, 'reanalysis_precip_amt_kg_per_m2')
fillMissingValues(iq_features, 'reanalysis_relative_humidity_percent')
fillMissingValues(iq_features, 'reanalysis_sat_precip_amt_mm')
fillMissingValues(iq_features, 'reanalysis_specific_humidity_g_per_kg')
fillMissingValues(iq_features, 'reanalysis_tdtr_k')
fillMissingValues(iq_features, 'station_avg_temp_c')
fillMissingValues(iq_features, 'station_diur_temp_rng_c')
fillMissingValues(iq_features, 'station_max_temp_c')
fillMissingValues(iq_features, 'station_min_temp_c')
fillMissingValues(iq_features, 'station_precip_mm')

iq_features.isnull().sum()

sj_features.columns

sj_features['reanalysis_air_temp_k'] = sj_features['reanalysis_air_temp_k'] - 273.15
sj_features['reanalysis_avg_temp_k'] = sj_features['reanalysis_avg_temp_k'] - 273.15
sj_features['reanalysis_dew_point_temp_k'] = sj_features['reanalysis_dew_point_temp_k'] - 273.15
sj_features['reanalysis_max_air_temp_k'] = sj_features['reanalysis_max_air_temp_k'] - 273.15
sj_features['reanalysis_min_air_temp_k'] = sj_features['reanalysis_min_air_temp_k'] - 273.15
sj_features['reanalysis_tdtr_k'] = sj_features['reanalysis_tdtr_k'] - 273.15

iq_features['reanalysis_air_temp_k'] = iq_features['reanalysis_air_temp_k'] - 273.15
iq_features['reanalysis_avg_temp_k'] = iq_features['reanalysis_avg_temp_k'] - 273.15
iq_features['reanalysis_dew_point_temp_k'] = iq_features['reanalysis_dew_point_temp_k'] - 273.15
iq_features['reanalysis_max_air_temp_k'] = iq_features['reanalysis_max_air_temp_k'] - 273.15
iq_features['reanalysis_min_air_temp_k'] = iq_features['reanalysis_min_air_temp_k'] - 273.15
iq_features['reanalysis_tdtr_k'] = iq_features['reanalysis_tdtr_k'] - 273.15

iq_features.to_csv("IQUITOS FINAL.csv")
iq_labels.to_csv("IQUITOS LABELS FINAL.csv")

sj_correlation = sj_features.corr()
sj_features.drop(['reanalysis_sat_precip_amt_mm'], axis=1, inplace=True)
sj_features.drop(['reanalysis_specific_humidity_g_per_kg'], axis=1, inplace=True)

iq_correlations = iq_features.corr()
iq_features.drop(['reanalysis_sat_precip_amt_mm'], axis=1, inplace=True)
iq_features.drop(['reanalysis_specific_humidity_g_per_kg'], axis=1, inplace=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
sj_features = scaler.fit_transform(sj_features)
iq_features = scaler.fit_transform(iq_features)

from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
sj_features = pca.fit_transform(sj_features)
iq_features = pca.fit_transform(iq_features)

total = sum(pca.explained_variance_)
k = 0
cur_var = 0 
while cur_var / total < 0.999:
    cur_var += pca.explained_variance_[k]
    k = k+1

sj_features.shape

X_train_sj = sj_features[:-100, :]
X_train_iq = iq_features[:-50, :]
y_train_sj = sj_labels.iloc[:-100, 3].values
y_train_iq = iq_labels.iloc[:-50, 3].values

X_val_sj = sj_features[-100:, :]
X_val_iq = iq_features[-50:, :]
y_val_sj = sj_labels.iloc[-100:, 3].values
y_val_iq = iq_labels.iloc[-50:, 3].values

from sklearn.linear_model import BayesianRidge, LinearRegression
sj_model = RandomForestRegressor()
sj_model.fit(X_train_sj, y_train_sj)
sj_pred = sj_model.predict(X_val_sj)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_val_sj, sj_pred)

from sklearn.ensemble import RandomForestRegressor

iq_model = BayesianRidge()
iq_model.fit(X_train_iq, y_train_iq)
iq_pred = iq_model.predict(X_val_iq)

mean_absolute_error(y_val_iq, iq_pred)
