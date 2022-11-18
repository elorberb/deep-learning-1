import numpy as np
import os
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import operator

seed = 1111
np.random.seed(seed)


def initialize_parameters(layer_dims: list) -> dict:
    """

    @param layer_dims:
    an array of the dimensions of each layer in the
    network (layer 0 is the size of the flattened input, layer L is the output softmax)
    @return:
    params: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL).
    """
    params = {}
    # starting from 1 because index 0 is input
    for i in range(1, len(layer_dims)):
        Wi = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2 / layer_dims[i])  # define weights
        b = np.zeros((layer_dims[i], 1))  # define biases
        params[i] = [Wi, b]
    return params


def linear_forward(A: np.array, W: np.array, b: np.array) -> (np.array, tuple):
    """
    Description:
    Implement the linear part of a layer's forward propagation.

    @param A: the activations of the previous layer
    @param W: the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    @param b: the bias vector of the current layer (of shape [size of current layer, 1])
    @return:
    Z: the linear component of the activation function (i.e., the value before applying the non-linear function)
    linear_cache: a dictionary containing A, W, b (stored for making the backpropagation easier to compute)

    """
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    return Z, linear_cache


def softmax(Z: np.array) -> (np.array, np.array):
    """

    @param Z: the linear component of the activation function
    @return:
    A: the activations of the layer
    activation_cache: returns Z, which will be useful for the backpropagation

    """
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    activation_cache = Z
    return A, activation_cache


def relu(Z: np.array) -> (np.array, tuple):
    """
    @param Z: the linear component of the activation function
    @return:
    A – the activations of the layer
    activation_cache – returns Z, which will be useful for the backpropagation
    """
    A = np.maximum(0, Z)
    activation_cache = Z
    return A, activation_cache


def linear_activation_forward(A_prev: np.array, W: np.array, B: np.array, activation: str) -> (np.array, list):
    """
    Description:
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    @param A_prev: activations of the previous layer
    @param W: the weights matrix of the current layer
    @param B: the bias vector of the current layer
    @param activation: the activation function to be used (a string, either “softmax” or “relu”)
    @return:
    A: the activations of the current layer
    cache: a joint dictionary containing both linear_cache and activation_cache

    """
    Z, linear_cache = linear_forward(A_prev, W, B)
    A, activation_cache = None, None
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'softmax':
        A, activation_cache = softmax(Z)
    cache = [linear_cache, activation_cache]
    return A, cache


def l_model_forward(X: np.array, parameters: dict, use_batchnorm: bool) -> (np.array, tuple):
    """
    Description:
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation

    @param X: the data, numpy array of shape (input size, number of examples)
    @param parameters: the initialized W and b parameters of each layer
    @param use_batchnorm: a boolean flag used to determine whether to apply batchnorm after
    the activation (note that this option needs to be set to “false” in Section 3 and “true” in Section 4).

    @return:
    AL – the last post-activation value
    caches – a list of all the cache objects generated by the linear_forward function

    """
    caches = []
    A = X
    layers_len = len(parameters)
    # Each middle layer - relu activations
    for i in range(1, layers_len):
        W, B = parameters[i][0], parameters[i][1]
        A, cache = linear_activation_forward(A_prev=A, W=W, B=B, activation="relu")
        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(cache)
    # Last layer - softmax activation
    last_W, last_B = parameters[layers_len][0], parameters[layers_len][1]
    AL, cache = linear_activation_forward(A_prev=A, W=last_W, B=last_B, activation="softmax")
    caches.append(cache)

    return AL, caches


def compute_cost(AL: np.array, Y: np.array) -> np.array:
    """
    Description:
    Implement the cost function defined by equation. The requested cost function is categorical cross-entropy loss.

    @param AL: probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
    @param Y: the labels vector (i.e. the ground truth)
    @return:
    cost: the cross-entropy cost
    """
    cost = 0
    num_of_classes = AL.shape[0]
    num_of_examples = AL.shape[1]
    for e in range(num_of_examples):
        for c in range(num_of_classes):
            if Y[c][e] == 1:  # saves unnecessary computation
                cost -= Y[c][e] * np.log(AL[c][e])
    cost = cost / num_of_examples
    return cost


def apply_batchnorm(A: np.array) -> np.array:
    """
    Description:
    performs batchnorm on the received activation values of a given layer.

    @param A: the activation values of a given layer
    @return:
    NA: the normalized activation values, based on the formula learned in class
    """

    mean_values = np.mean(A)
    var_values = np.var(A)
    eps = np.finfo(float).eps
    NA = (A - mean_values) / np.sqrt((var_values + eps))
    return NA


def linear_backward(dZ: np.array, cache: tuple) -> tuple:
    """
    description:
    Implements the linear part of the backward propagation process for a single layer

    @param dZ: the gradient of the cost with respect to the linear output of the current layer (layer l)
    @param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    @return:
    dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW: Gradient of the cost with respect to W (current layer l), same shape as W
    db: Gradient of the cost with respect to b (current layer l), same shape as b

    """

    A_prev, W, b = cache
    len_examples = A_prev.shape[1]

    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_prev.T) / len_examples
    db = np.sum(dZ, axis=1, keepdims=True) / len_examples

    return dA_prev, dW, db


def linear_activation_backward(dA: np.array, cache: tuple, activation: str) -> tuple:
    """
    Description:
    Implements the backward propagation for the LINEAR->ACTIVATION layer.
    The function first computes dZ and then applies the linear_backward function.

    @param dA: post activation gradient of the current layer
    @param cache: contains both the linear cache and the activations cache
    @param activation: which activation function used to compute the A
    @return:
    dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW – Gradient of the cost with respect to W (current layer l), same shape as W
    db – Gradient of the cost with respect to b (current layer l), same shape as b

    """
    dZ = None
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'softmax':
        dZ = softmax_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def relu_backward(dA: np.array, activation_cache: np.array) -> np.array:
    """
    Description:
    Implements backward propagation for a ReLU unit

    @param dA: the post-activation gradient
    @param activation_cache: contains Z (stored during the forward propagation)
    @return:
    dZ: gradient of the cost with respect to Z
    """

    dZ = np.array(dA, copy=True)
    dZ[activation_cache <= 0] = 0
    return dZ


def softmax_backward(dA: np.array, activation_cache: np.array) -> np.array:
    """
    Description:
    Implements backward propagation for a softmax unit

    @param dA: the post-activation gradient - p - output of the softmax forward propagation
    @param activation_cache: contains Z (stored during the forward propagation) - Y - true value
    @return:
    dZ: gradient of the cost with respect to Z
    """
    # correct params
    Y = activation_cache
    p = dA
    dZ = p - Y
    return dZ


def l_model_backward(AL: np.array, Y: np.array, caches: list[tuple]) -> dict:
    """
    Description:
    Implement the backward propagation process for the entire network.

    @param AL: the probabilities vector, the output of the forward propagation (L_model_forward)
    @param Y: the true labels vector (the "ground truth" - true classifications)
    @param caches: list of caches containing for each layer: a) the linear cache; b) the activation cache
    @return:
    Grads - a dictionary with the gradients
    """
    grads = {}
    num_layers = len(caches)
    last_cache = caches[-1]
    linear_last_cache, activation_last_cache = last_cache
    dZ = softmax_backward(dA=AL, activation_cache=Y.reshape(AL.shape))
    dA, dW, db = linear_backward(dZ=dZ, cache=linear_last_cache)
    grads[f"dA{num_layers}"], grads[f"dW{num_layers}"], grads[f"db{num_layers}"] = dA, dW, db

    for curr_layer in reversed(range(num_layers - 1)):
        dA, dW, db = linear_activation_backward(grads[f"dA{curr_layer + 2}"], caches[curr_layer], "relu")
        grads[f"dA{curr_layer + 1}"], grads[f"dW{curr_layer + 1}"], grads[f"db{curr_layer + 1}"] = dA, dW, db

    return grads


def update_parameters(parameters: dict, grads: dict, learning_rate: float) -> dict:
    """
    Description:
    Updates parameters using gradient descent

    @param parameters: a python dictionary containing the DNN architecture’s parameters
    @param grads: a python dictionary containing the gradients (generated by L_model_backward)
    @param learning_rate: the learning rate used to update the parameters (the “alpha”)
    @return:
    parameters – the updated values of the parameters object provided as input
    """
    num_layers = len(parameters)
    for i in range(1, num_layers + 1):
        parameters[i][0] -= learning_rate * grads[f"dW{i}"]
        parameters[i][1] -= learning_rate * grads[f"db{i}"]

    return parameters


def l_layer_model(X: np.array, Y: np.array, layers_dims: list, learning_rate: float, num_iterations: int,
                  batch_size: int, batch_norm: bool) -> tuple:
    """
    Description:
    Implements a L-layer neural network.All layers but the last should have
    the ReLU activation function, and the final layer will apply the softmax activation function.
    The size of the output layer should be equal to the number of labels in the data.
    Please select a batch size that enables your
    code to run well (i.e. no memory overflows while still running relatively fast).

    @param X: the input data, a numpy array of shape (height*width , number_of_examples)
    Comment: since the input is in grayscale we only have height and width, otherwise it would have been height*width*3
    @param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    @param layers_dims: a list containing the dimensions of each layer, including the input
    @param learning_rate: the learning rate used to update the parameters (the “alpha”)
    @param num_iterations: number of epoches we iterate over the data.
    @param batch_size: the number of examples in a single training batch.
    @param batch_norm: bool to indicate if user wants to normalize the data.

    @return:
    parameters – the parameters learnt by the system during
    the training (the same parameters that were updated in the update_parameters function).
    costs – the values of the cost function (calculated by the compute_cost function).
    One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values).

    """
    X_train, X_test, y_train, y_test = split_preprocess_train_data(X, Y)
    parameters = initialize_parameters(layers_dims)
    train_iter, epoch = 0, 0
    stop_early = False
    costs, graph_costs = [], [[], []]
    indices = np.arange(len(y_train[0, :]))  # define indices for batch data

    while train_iter < num_iterations and not stop_early:
        print(f'Current epoch = {epoch}')
        batch_begin = 0
        batch_finish = batch_size
        shuffled_ind = np.copy(indices)
        np.random.shuffle(shuffled_ind)
        accuracy_before, accuracy = 0, 0

        while batch_begin < len(shuffled_ind) and not stop_early:
            # split data to batch
            X_batch, y_batch = create_batch_data(batch_begin, batch_finish, shuffled_ind, X_train, y_train)
            # perform forward propagation
            AL, caches = l_model_forward(X_batch, parameters, batch_norm)
            #  values needs to be saved each 100 training iterations (e.g. 3000 iterations -> 30 values).
            if train_iter % 100 == 0:
                cost = compute_cost(AL, y_batch)
                costs.append(cost)
                graph_costs[0].append(train_iter)
                graph_costs[1].append(cost)

                accuracy = predict(X_test, y_test, parameters, batch_norm)
                print(f"Current iteration = {train_iter} , Cost = {cost}, Accuracy = {accuracy}")
                if check_stopping_criterion(accuracy_before, accuracy):
                    stop_early = True
                    continue
                accuracy_before = accuracy
            grads = l_model_backward(AL, y_batch, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
            # update amount of data trained
            batch_begin, batch_finish = batch_begin + batch_size, batch_finish + batch_size
            train_iter += 1
        epoch += 1
    return parameters, costs, graph_costs, train_iter


def check_stopping_criterion(accuracy_before: float, current_accuracy: float) -> bool:
    """

    @param accuracy_before: Accuracy level last iteration
    @param current_accuracy: Accuracy level current iteration
    @return:
    bool representing whether to stop training
    """
    return accuracy_before > current_accuracy + 0.1


def create_batch_data(batch_begin: int, batch_finish: int, shuffled_ind: list, X: np.array, Y: np.array) -> tuple:
    """

    @param batch_begin: start ind for batch
    @param batch_finish: end idx for bach
    @param shuffled_ind:  list represnting shuffled indices
    @param X: X data
    @param Y: Y data
    @return:
    data split into batch
    """
    if batch_finish > len(shuffled_ind):
        batch_indices = shuffled_ind[batch_begin:len(shuffled_ind)]
    else:
        batch_indices = shuffled_ind[batch_begin:batch_finish]
    X_batch = X[:, batch_indices]
    y_batch = Y[:, batch_indices]

    return X_batch, y_batch


def split_preprocess_train_data(X: np.array, Y: np.array):
    """

    @param X: the X_train data from keras
    @param Y: the y_train data from keras
    @return:
    data preprocessed and split into training and validation
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    X_train = X_train.reshape(X_train.shape[0], -1).T  # The "-1" makes reshape flatten the remaining dimensions
    X_train = np.divide(X_train, 255)
    X_test = X_test.reshape(X_test.shape[0], -1).T
    X_test = np.divide(X_test, 255)

    y_train = np.squeeze(np.eye(y_train.max() + 1)[y_train.reshape(-1)]).T
    y_test = np.squeeze(np.eye(y_test.max() + 1)[y_test.reshape(-1)]).T

    return X_train, X_test, y_train, y_test


def split_preprocess_test_data(X_test: np.array, y_test: np.array):
    """

    @param X_test: X testing data from keras
    @param y_test: y testing data from keras
    @return:
    data preprocessed for testing model
    """
    X_test = X_test.reshape(X_test.shape[0], -1).T
    X_test = np.divide(X_test, 255)
    y_test = np.squeeze(np.eye(y_test.max() + 1)[y_test.reshape(-1)]).T

    return X_test, y_test


def predict(X: np.array, Y: np.array, parameters: dict, use_batchnorm: bool):
    """

    @param X: the input data, a numpy array of shape (height*width, number_of_examples)
    @param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    @param parameters: a python dictionary containing the DNN architecture’s parameters
    @param use_batchnorm:
    @return:
    return accuracy metric for the NN
    """
    count_match = 0
    prob, caches = l_model_forward(X, parameters, use_batchnorm)
    prob = softmax(prob)[0]
    predictions_length = prob.shape[1]
    for i in range(predictions_length):
        curr_Y = max(enumerate(Y[:, i]), key=operator.itemgetter(1))[0]
        curr_prob = max(enumerate(prob[:, i]), key=operator.itemgetter(1))[0]
        if curr_prob == curr_Y:
            count_match += 1
    accuracy = np.round((count_match / Y.shape[1]), 4)

    return accuracy


def get_accuracy_results(X_train_p, X_eval, y_train_p, y_eval, X_test, y_test, params, batch_norm):
    """
    Calculates accuracy results for training, evaluation, and testing data
    @return:
    all accuracy results
    """
    acc_train = predict(X_train_p, y_train_p, params, batch_norm)
    acc_eval = predict(X_eval, y_eval, params, batch_norm)
    acc_test = predict(X_test, y_test, params, batch_norm)
    return [acc_train, acc_eval, acc_test]


def save_results(costs, parameters, accuracy_results, batch_size, batch_norm, name, hyper_param):
    """ Save the results of the NN to new dir called output"""
    def write_info_to_file():
        f = open(f"output/results_{name}.txt", "w")
        f.write(f"Parameters: {parameters}\n")
        f.write(f"Accuracy Train: {accuracy_results[0]}\n")
        f.write(f"Accuracy Validation: {accuracy_results[1]}\n")
        f.write(f"Accuracy Test: {accuracy_results[2]}\n")
        f.write(f"Cost: {costs[1]}\n")
        f.write(hyper_param)
        f.close()

    def create_plot():
        plt.plot(costs[0], costs[1])
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title(f'Batch Size: {batch_size}, Batchnorm: {batch_norm}')
        plt.savefig(f"output/plot_{name}.png")

    print(hyper_param)
    if not os.path.exists("output"):
        os.mkdir("output")
    write_info_to_file()
    create_plot()


def create_NN(X_train_p, X_eval, y_train_p, y_eval, X_test, y_test, batch_norm, X_train, y_train, layers, learning_rate,
                                                        number_iterations, batch_size, name):
    """
    Create NN with specific parameters.
    Saving the results and plot for the cost values.

    """
    start_time = time.time()
    params, costs, costs_for_graph, number_run = l_layer_model(X_train, y_train, layers, learning_rate,
                                                               number_iterations, batch_size, batch_norm=False)

    finish_time = (time.time() - start_time) / 60
    finish_time = "%.3f" % finish_time
    hyper_param = "Hyper parameters: \nbatch=" + str(batch_size) + "\nlearningRate=" + str(learning_rate) \
                  + "\nnumberIterations=" + str(number_iterations) + "\nbatchNorm=" + str(batch_norm) + \
                  "\nNumberRun=" + str(number_run) + "\nTime in minutes=" + str(finish_time)

    accuracy_results = get_accuracy_results(X_train_p, X_eval, y_train_p, y_eval, X_test, y_test, params,
                                            batch_norm)
    save_results(costs_for_graph, params, accuracy_results, batch_size, batch_norm, name, hyper_param)
