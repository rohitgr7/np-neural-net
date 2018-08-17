import numpy as np


def relu_forward(Z):
    cache = Z > 0
    return Z * cache, cache


def relu_backward(dA, cache):
    return dA * cache


def sigmoid_forward(Z):
    sig_Z = 1. / (1 + np.exp(-Z))
    return sig_Z, sig_Z


def sigmoid_backward(dA, cache):
    return dA * cache * (1 - cache)


def tanh_forward(Z):
    tanh_Z = np.tanh(Z)
    return tanh_Z, tanh_Z


def tanh_backward(dA, cache):
    return dA * (1 - cache**2)


def lrelu_forward(Z, alpha):
    cache = np.float32(Z > 0)
    cache[cache == 0] += alpha

    return Z * cache, cache


def lrelu_backward(dA, cache):
    return dA * cache
