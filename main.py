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
    # preprocess data
    X_train_p, X_eval, y_train_p, y_eval = split_preprocess_train_data(X_train, y_train)
    X_test, y_test = split_preprocess_test_data(X_test, y_test)

    # define parameters
    layers = [784, 20, 7, 5, 10]
    batch_size = 200
    learning_rate = 0.009
    number_iterations = 20000
    batch_norm = False
    start_time = time.time()
    # create NN
    params, costs, costs_for_graph, number_run = l_layer_model(X_train, y_train, layers,
                                                               learning_rate,
                                                               number_iterations, batch_size, batch_norm=False)

    time = (time.time() - start_time) / 60
    time = "%.3f" % time
    hyper_param = "Hyper parameters: \nbatch=" + str(batch_size) + "\nlearningRate=" + str(learning_rate)\
                  + "\nnumberIterations=" + str(number_iterations) + "\nbatchNorm=" + str(batch_norm) + \
        "\nNumberRun=" + str(number_run) + "\nTime in minutes=" + str(time)

    accuracy_results = get_accuracy_results(X_train_p, X_eval, y_train_p, y_eval, X_test, y_test, params, batch_norm)
    save_results(costs_for_graph, params, accuracy_results, batch_size, batch_norm, "run_no_norm", hyper_param)
