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

    @param A: the activations of the previous layer
    @param W: the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    @param b: the bias vector of the current layer (of shape [size of current layer, 1])
    @return:
    Z: the linear component of the activation function (i.e., the value before applying the non-linear function)
    linear_cache: a dictionary containing A, W, b (stored for making the backpropagation easier to compute)

    """

    Z = np.dot(W, A) + b
    linear_cache = {A: A, W: W, b: b}
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

    A = np.max(0, Z)
    activation_cache = Z
    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation):
    """

    @param A_prev: activations of the previous layer
    @param W: the weights matrix of the current layer
    @param B: the bias vector of the current layer
    @param activation: the activation function to be used (a string, either “softmax” or “relu”)
    @return:
    A – the activations of the current layer
    cache – a joint dictionary containing both linear_cache and activation_cache

    """
    Z, linear_cache = linear_forward(A_prev, W, B)
    A, activation_cache = None, None

    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'softmax':
        A, activation_cache = softmax(Z)
    cache = {linear_cache: linear_cache, activation_cache: activation_cache}
    return A, cache


def l_model_forward(X, parameters, use_batchnorm):
    pass


def compute_cost(AL, Y):
    pass


def apply_batchnorm(A):
    pass


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