import numpy as np
import os
import matplotlib.pyplot as plt
import time

np.random.seed(314977596)


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
        Wi = np.random.randn(layer_dims[i], layer_dims[i-1])  # define weights
        b = np.zeros((layer_dims[i], 1))  # define biases
        params[i] = [Wi, b]
    return params


def linear_forward(A: np.array, W: np.array, b: np.array) -> (float, dict):
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
    linear_cache = {"A": A, "W": W, "b": b}
    return Z, linear_cache


def softmax(Z: float) -> (float, dict):
    """

    @param Z: the linear component of the activation function
    @return:
    A: the activations of the layer
    activation_cache: returns Z, which will be useful for the backpropagation

    """
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    activation_cache = Z
    return A, activation_cache


def relu(Z: float) -> (float, dict):
    """
    @param Z: the linear component of the activation function
    @return:
    A – the activations of the layer
    activation_cache – returns Z, which will be useful for the backpropagation
    """
    print(Z)
    A = max(0, Z)
    activation_cache = Z
    return A, activation_cache


def linear_activation_forward(A_prev: float, W: np.array, B: np.array, activation: str) -> (float, dict):
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
    cache = {"linear_cache": linear_cache, "activation_cache": activation_cache}
    return A, cache


def l_model_forward(X: np.array, parameters: dict, use_batchnorm: bool):
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
    # Each middle layer - relu activations
    for i in range(1, len(parameters)-1):
        W, B = parameters[i][0], parameters[i][1]
        A, cache = linear_activation_forward(A_prev=A, W=W, B=B, activation="relu")
        if use_batchnorm:
            A = apply_batchnorm(A)

        caches.append(cache)
    # Last layer - softmax activation
    last_W, last_B = parameters[-1][0], parameters[-1][1]
    AL, cache = linear_activation_forward(A_prev=A, W=last_W, B=last_B, activation="softmax")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    Description:
    Implement the cost function defined by equation. The requested cost function is categorical cross-entropy loss. The formula is as follows:

    @param AL: probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
    @param Y: the labels vector (i.e. the ground truth)
    @return:
    cost: the cross-entropy cost
    """
    cost = 0
    num_of_classes = AL[0]
    num_of_examples = AL[1]
    for e in num_of_classes:
        for c in num_of_classes:
            cost += Y[c][e] * np.log(AL[c][e])
    cost = -1 * cost / num_of_examples
    return cost


def apply_batchnorm(A):
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

def linear_backward(dZ, cache):
    pass


def linear_activation_backward(dA, cache, activation):
    pass


def relu_backward(dA, activation_cache):
    pass


def softmax_backward(dA, activation_cache):
    pass


def l_model_backward(AL, Y, caches):
    pass


def update_parameters(parameters, grads, learning_rate):
    pass


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    pass


def predict(X, Y, parameters, use_batchnorm):
    pass