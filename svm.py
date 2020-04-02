#!/usr/bin/env python
# coding: utf-8

# This is a one class support vector machine (svm) model for anomaly detection in time series
# Author: Lei Zhang; Email: leizhang@ryerson.ca

from sklearn import preprocessing
from sklearn.svm import OneClassSVM
from pandas import read_csv, to_datetime, DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np

# read and understand data
df = read_csv("~/Projects/IBM/simulator/cpu_train.csv", index_col=0)
# set index as time
df = df.reset_index()
df = df.rename(columns={"index": "time"})
print(df.info())
df.plot(x=None, y='cpu')
plt.show()

# get test data
df_test = read_csv("~/Projects/IBM/simulator/cpu_test.csv", index_col=0)
# set index as time
df_test = df_test.reset_index()
df_test = df_test.rename(columns={"index": "time"})
print(df_test.info())
df_test.plot(x=None, y='cpu')
plt.show()

# standardize the data
data = df[['cpu']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = DataFrame(np_scaled)

# train the model
outliers_fraction = 0.01
# model = OneClassSVM(nu=0.95 * outliers_fraction)
model = OneClassSVM(nu=outliers_fraction)
model.fit(data)

# add the data to the main
#df['anomaly'] = Series(model.predict(data))
#df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
#print(df['anomaly'].value_counts())
data_test = df_test[['cpu']]
# standardize test data
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data_test)
data_test = DataFrame(np_scaled)
# test model
df_test['anomaly'] = Series(model.fit_predict(data_test))
#df_test['anomaly'] = Series(model.predict(data_test))
df_test['anomaly'] = df_test['anomaly'].map({1: 0, -1: 1})
print(df_test['anomaly'].value_counts())

# visualisation of anomaly throughout time
fig, ax = plt.subplots()
#a = df.loc[df['anomaly'] == 1, ['time', 'cpu']]
#ax.plot(df['time'], df['cpu'], color='blue')
#ax.scatter(a['time'], a['cpu'], color='red')
#plt.show()
b = df_test.loc[df_test['anomaly'] == 1, ['time', 'cpu']]
ax.plot(df_test['time'], df_test['cpu'], color='blue')
ax.scatter(b['time'], b['cpu'], color='red')
plt.show()