
import numpy as np


def apply_batchnorm2(A):
    """
    performs batchnorm on the received activation values of a given layer.
    :param A: the activation values of a given layer
    :return:
    NA - the normalized activation values, based on the formula learned in class
    # """
    sum_value = np.sum(A, axis=1, keepdims=True)
    mean = sum_value/A.shape[1]
    var = np.sum(np.square(A-mean), axis=1, keepdims=True)/A.shape[1]
    eps = np.finfo(float).eps
    NA = np.divide(np.subtract(A, mean), np.sqrt(var+eps))
    return NA


def initialize_parameters(layer_dims):
    """
    :param layer_dims: an array of the dimensions of each layer in the network (layer 0 is the size of the flattened
    input, layer L is the output softmax)
    :return: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL).
    Hint: Use the randn and zeros functions of numpy to initialize W and b, respectively
    """
    parameters = {}
    for l in range(1, len(layer_dims)):  # first value is not a layer, its the inputs
        parameters[l] = [np.random.randn(layer_dims[l], layer_dims[l - 1]),
                         np.zeros((layer_dims[l], 1))]

    return parameters


if __name__ == '__main__':
    # f_a = np.array([[4, 1, -3]])
    # a_w1 = np.array([[1], [0], [1]])
    #
    # f_w1 = np.dot(a_w1, f_a)
    # print(a_w1.shape)
    # print(f_a.shape)
    # print(f_w1)
    #
    # f_w1_2 = np.dot(f_a.T, a_w1.T)
    # print(f_w1_2.T)
    s = 0
    l = [1,2,3,4,5]
    num_layers = len(l)
    for i in reversed(range(num_layers-1)):
        print(i+2)
