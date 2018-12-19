#!/usr/bin/env python
# coding: utf-8

# This is a CNN autoencoder model for anomaly detection in time series
# Author: Lei Zhang; Email: leizhang@ryerson.ca

from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten

time_window_size = 10

model = Sequential()
model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', input_shape=(time_window_size, 1)))
model.add(GlobalMaxPool1D())

model.add(Dense(units=time_window_size, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', 'mape'])
print(model.summary())