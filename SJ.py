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

sj_features.columns

import seaborn as sns

sns.boxplot(sj_features.ndvi_ne)

sj_features[sj_features.ndvi_ne > 0.255].index
sj_features.drop([196, 651, 867], axis=0, inplace=True)
sj_labels.drop([196, 651, 867], axis=0, inplace=True)

sj_features[sj_features.ndvi_ne < -0.145].index
sj_features.drop([349, 708], axis=0, inplace=True)
sj_labels.drop([349, 708], axis=0, inplace=True)



sns.boxplot(sj_features.ndvi_nw)

sj_features[sj_features.ndvi_nw > 0.24].index
sj_features.drop([3, 13, 39, 173, 257, 265, 538, 790], axis=0, inplace=True)
sj_labels.drop([3, 13, 39, 173, 257, 265, 538, 790], axis=0, inplace=True)

sj_features[sj_features.ndvi_nw < -0.11].index
sj_features.drop([195, 649, 687], axis=0, inplace=True)
sj_labels.drop([195, 649, 687], axis=0, inplace=True)




sns.boxplot(sj_features.ndvi_se)

sj_features[sj_features.ndvi_se > 0.315].index
sj_features.drop([26, 389], axis=0, inplace=True)
sj_labels.drop([26, 389], axis=0, inplace=True)

sj_features[sj_features.ndvi_se < 0.03].index
sj_features.drop([722, 875], axis=0, inplace=True)
sj_labels.drop([722, 875], axis=0, inplace=True)


sns.boxplot(sj_features.ndvi_sw)

sj_features[sj_features.ndvi_sw > 0.3].index
sj_features.drop([329, 20, 769], axis=0, inplace=True)
sj_labels.drop([329, 20, 769], axis=0, inplace=True)

sj_features[sj_features.ndvi_sw < 0].index
sj_features.drop([553, 554], axis=0, inplace=True)
sj_labels.drop([553, 554], axis=0, inplace=True)



sns.boxplot(sj_features.precipitation_amt_mm)

sj_features[sj_features.precipitation_amt_mm > 125].index
sj_features.drop([25, 485, 589], axis=0, inplace=True)
sj_labels.drop([25, 485, 589], axis=0, inplace=True)


sj_features.columns

sns.boxplot(sj_features.reanalysis_air_temp_k)
sns.boxplot(sj_features.reanalysis_avg_temp_k)
sns.boxplot(sj_features.reanalysis_dew_point_temp_k)
sns.boxplot(sj_features.reanalysis_max_air_temp_k)
sns.boxplot(sj_features.reanalysis_min_air_temp_k)


sns.boxplot(sj_features.reanalysis_precip_amt_kg_per_m2)

sj_features[sj_features.reanalysis_precip_amt_kg_per_m2 > 64.5].index
sj_features.drop([234, 490, 843], axis=0, inplace=True)
sj_labels.drop([234, 490, 843], axis=0, inplace=True)


sns.boxplot(sj_features.reanalysis_relative_humidity_percent)

sj_features[sj_features.reanalysis_relative_humidity_percent < 69].index
sj_features.drop([457], axis=0, inplace=True)
sj_labels.drop([457], axis=0, inplace=True)



sns.boxplot(sj_features.reanalysis_sat_precip_amt_mm)
sj_features[sj_features.reanalysis_sat_precip_amt_mm > 106.8].index
sj_features.drop([172], axis=0, inplace=True)
sj_labels.drop([172], axis=0, inplace=True)

sns.boxplot(sj_features.reanalysis_specific_humidity_g_per_kg)



sns.boxplot(sj_features.reanalysis_tdtr_k)
sj_features[sj_features.reanalysis_tdtr_k > 3.68].index
sj_features.drop([279], axis=0, inplace=True)
sj_labels.drop([279], axis=0, inplace=True)

sj_features.columns

sns.boxplot(sj_features.station_avg_temp_c)



sns.boxplot(sj_features.station_diur_temp_rng_c)

sj_features[sj_features.station_diur_temp_rng_c > 8.8].index
sj_features.drop([41, 45], axis=0, inplace=True)
sj_labels.drop([41, 45], axis=0, inplace=True)

sj_features[sj_features.station_diur_temp_rng_c < 4.8].index
sj_features.drop([345], axis=0, inplace=True)
sj_labels.drop([345], axis=0, inplace=True)



sns.boxplot(sj_features.station_max_temp_c)
sj_features[sj_features.station_max_temp_c < 27.5].index
sj_features.drop([770, 819, 924], axis=0, inplace=True)
sj_labels.drop([770, 819, 924], axis=0, inplace=True)


sns.boxplot(sj_features.station_min_temp_c)
sj_features[sj_features.station_min_temp_c < 18].index
sj_features.drop([767], axis=0, inplace=True)
sj_labels.drop([767], axis=0, inplace=True)


sns.boxplot(sj_features.station_precip_mm)
sj_features[sj_features.station_precip_mm > 59].index
sj_features.drop([419, 663], axis=0, inplace=True)
sj_labels.drop([419, 663], axis=0, inplace=True)

sj_features.hist()

def fillMissingValues(city,column):
    index = city[column].index[city[column].apply(np.isnan)]
    
    for value in index:
        week = city['weekofyear'][value]
        mean = city[city['weekofyear'] == week][column].mean()
        city[column][value] = mean

sj_features.columns

sj_features.isnull().sum()

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
