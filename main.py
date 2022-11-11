import numpy as np
from code import *


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
    # print(np.zeros((4,1)))
    print(initialize_parameters([2,2,1]))