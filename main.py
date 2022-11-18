from ass1_code import *
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import operator
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = tf.keras.datasets.mnist.load_data()
    X_train, X_test, y_train, y_test = data[0][0], data[1][0], data[0][1], data[1][1]
    layers = [784, 20, 7, 5, 10]

    X_test, y_test = split_preprocess_test_data(X_test, y_test)
    batch_size = 200
    learning_rate = 0.009
    number_iterations = 20000
    batch_norm = False
    start_time = time.time()

    params, costs, costs_for_graph, number_run = l_layer_model(X_train, y_train, layers,
                                                               learning_rate,
                                                               number_iterations, batch_size, batch_norm=True)

    time = (time.time() - start_time)/60
    time = "%.3f" % time
    hyper_param = "batch_" + str(batch_size) + "_learningRate_" + str(learning_rate) + "_numberIterations_" + \
                  str(number_iterations) + "_batchNorm_" + str(batch_norm) + "_NumberRun_" + str(number_run) + "_Time"\
                  + str(time)

    acc_test = predict(X_test, y_test, params, batch_norm)


