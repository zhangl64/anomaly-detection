#!/usr/bin/env python
# coding: utf-8

# This is an LSTM RNN model for anomaly detection in time series
# Author: Lei Zhang; Email: leizhang@ryerson.ca

# todo: CNN as the input layer; change time steps; change the label policy between max and last

from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, GlobalMaxPool1D
from keras.layers.core import Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


# create datasets
# x - time series (timestamp and variables)
# y - anomaly or normal (boolean)
def get_data(filename, col_val, col_bool, time_steps=None):
    dataframe = read_csv(filename, skiprows=0)
    x = dataframe[[col_val]].values
    y = dataframe[[col_bool]].values

    if time_steps is None:
        # reshape x to 3D: samples, time_steps, and features
        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))  # change 2D to 3D
        # y = np.reshape(y, (y.shape[0], 1, y.shape[1])) # change 2D to 3D
    else:
        x = np.reshape(x, (int(x.shape[0]/time_steps), time_steps, x.shape[1]))  # change 2D to 3D
        y = np.reshape(y, (int(y.shape[0]/time_steps), time_steps))
        # reduce y to the max value on each row only
        y = np.amax(y, axis=1)
        # reshape y to 3D for LSTM
        # y = np.reshape(y, (y.shape[0], 1, 1))

    return x, y


# combine multiple datasets
def get_multi_data(fileprefix, col_val, col_bool, start, end):
    for i in range(start, end):
        filename = fileprefix + str(i) + ".csv"
        dataframe = read_csv(filename, skiprows=0)
        x = dataframe[[col_val]].values
        y = dataframe[[col_bool]].values
        # reshape x to 3D: samples, timesteps, and features
        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
        if i == start:
            x_all = x
            y_all = y
        else:
            x_all = np.concatenate((x_all, x), axis=0)
            y_all = np.concatenate((y_all, y), axis=0)
    return x_all, y_all


# build the model
def build_model(sequence, time_steps, data_dim):
    model = Sequential()

    # return_sequences must be True for a stack of LSTM layers
    model.add(LSTM(sequence, input_shape=(time_steps, data_dim), return_sequences=True))
    model.add(Dropout(0.2))

    # hidden layers
    for i in range(2):
        model.add(LSTM(sequence, return_sequences=True))
        model.add(Dropout(0.2))

    # cnn
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
    model.add(GlobalMaxPool1D())

    # output layer
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])

    print(model.summary())
    return model


# generate sample weights based on y_train and class_weight_dict
def generate_sample_weights(training_data, class_weight_dictionary):
    #sample_weights = [class_weight_dictionary[np.where(one_hot_row == 1)[0][0]] for one_hot_row in training_data]
    sample_weights = np.zeros(training_data.shape[0])
    for i in range(training_data.shape[0]):
        if training_data[i] == 0:
            sample_weights[i] = class_weight_dictionary[0]
        else:
            sample_weights[i] = class_weight_dictionary[1]

    return np.asarray(sample_weights)


# plot function
def pro_plot(x_test, predictions, y_test, rounded, loss_fun, mse_fun):
    # plot loss
    plt.plot(loss_fun)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()
    # plot mse
    plt.plot(mse_fun)
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='lower right')
    plt.show()
    # plot x_test
    x_plt = np.squeeze(x_test)
    plt.plot(x_plt)
    plt.title('x_test')
    plt.ylabel('value')
    plt.xlabel('observation')
    plt.show()
    # plot estimated
    o_plt = np.squeeze(predictions)
    plt.plot(o_plt)
    plt.title('original prediction')
    plt.ylabel('value')
    plt.xlabel('observation')
    plt.show()
    # plot y_test
    plt.subplot(211)
    y_plt = np.squeeze(y_test)
    plt.plot(y_plt)
    plt.title('measured class')
    plt.ylabel('class')
    plt.xlabel('observation')
    # plot classed
    plt.subplot(212)
    p_plt = np.squeeze(rounded)
    plt.plot(p_plt)
    plt.title('predicted class')
    plt.ylabel('class')
    plt.xlabel('observation')
    plt.show()


# confusion matrix plot
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# main
def main():
    # create data
    #x_train, y_train = get_multi_data('../datasets/webscope/A3Benchmark/A3Benchmark-TS', 'value', 'anomaly', 1, 100)
    #x_test, y_test = get_data('../datasets/webscope/A3Benchmark/A3Benchmark-TS100.csv', 'value', 'anomaly')
    time_steps = 4
    x_train, y_train = get_data('~/Projects/IBM/simulator/mem_train.csv', 'mem', 'label', time_steps)
    x_test, y_test = get_data('~/Projects/IBM/simulator/mem_test.csv', 'mem', 'label', time_steps)

    # build the model
    sequence = 50
    time_steps = x_train.shape[1]
    # time_steps = 10
    data_dim = x_train.shape[2]
    model = build_model(sequence, time_steps, data_dim)

    # fit the model
    # class_weight does not support for 3+ dimensional targets
    # calculate class_weight_dict dynamically
    class_1 = np.count_nonzero(y_train)
    ratio = (y_train.shape[0] - class_1) / class_1
    class_weight_dict = {0: 1, 1: ratio}

    history = model.fit(x_train, y_train, epochs=100, batch_size=5, verbose=2,
                        sample_weight=generate_sample_weights(y_train, class_weight_dict), validation_split=0.1)

    # evaluate the model
    scores = model.evaluate(x_test, y_test)
    print(scores)
    # print('Test accuracy: %s%%' % (scores[1] * 100))

    # predict
    predictions = model.predict(x_test)
    #rounded = [int(round(x[0])) for x in predictions]
    rounded = np.round(predictions)
    # reshape
    rounded = np.array(rounded)
    # y_test = np.squeeze(y_test)

    loss_fun = history.history['loss']
    mse_fun = history.history['mean_squared_error']
    pro_plot(x_test, predictions, y_test, rounded, loss_fun, mse_fun)

    # plot the confusion matrix
    cm = confusion_matrix(y_test, rounded)
    target_names = ['Normal', 'Anomaly']
    plot_confusion_matrix(cm, classes=target_names,
                          title='Confusion matrix')

    print('succeed')


if __name__ == '__main__':
    main()

