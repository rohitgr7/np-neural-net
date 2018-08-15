import numpy as np


def _init_hparams(params):
    h_params = {}

    for k in list(params.keys()):
        h_params[k] = np.zeros_like(params[k])

    return h_params


# SGD
class SGD:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, params, grads):

        for k in list(params.keys()):
            params[k] -= self.learning_rate * grads[k]

        return params


# SGD Momentum
class SGDMom:

    def __init__(self, learning_rate):
        self.init_hp = False
        self.learning_rate = learning_rate
        self.gamma = 0.9

    def optimize(self, params, grads):
        if not self.init_hp:
            self.momentum = _init_hparams(params)
            self.init_hp = True

        for k in list(params.keys()):
            self.momentum[k] = self.gamma * self.momentum[k] - self.learning_rate * grads[k]
            params[k] += self.momentum[k]

        return params


# AdaGrad
class AdaGrad:

    def __init__(self, learning_rate):
        self.init_hp = False
        self.learning_rate = learning_rate
        self.epsilon = 1e-8

    def optimize(self, params, grads):
        if not self.init_hp:
            self.cache = _init_hparams(params)
            self.init_hp = True

        for k in list(params.keys()):
            self.cache[k] += grads[k]**2
            params[k] -= (self.learning_rate * grads[k]) / (self.cache[k] ** 0.5 + self.epsilon)

        return params


# RMSProp
class RMSProp:

    def __init__(self, learning_rate):
        self.init_hp = False
        self.learning_rate = learning_rate
        self.beta = 0.99
        self.epsilon = 1e-8

    def optimize(self, params, grads):
        if not self.init_hp:
            self.cache = _init_hparams(params)
            self.init_hp = True

        for k in list(params.keys()):
            self.cache[k] = self.beta * self.cache[k] + (1 - self.beta) * (grads[k]**2)
            params[k] -= (self.learning_rate * grads[k]) / (self.cache[k] ** 0.5 + self.epsilon)

        return params


# Adam
class Adam:

    def __init__(self, learning_rate):
        self.init_hp = False
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epsilon = 1e-8

    def optimize(self, params, grads, itr):
        if not self.init_hp:
            self.first_momentum = _init_hparams(params)
            self.second_momentum = _init_hparams(params)

        for k in list(params.keys()):
            self.first_momentum[k] = self.beta1 * self.first_momentum[k] + self.beta2 * (grads[k])
            self.second_momentum[k] = self.beta1 * self.second_momentum[k] + self.beta2 * (grads[k]**2)

            corr_first_momentum = self.first_momentum[k] / (1 - self.beta1**itr)
            corr_second_momentum = self.second_momentum[k] / (1 - self.beta2**itr)

            params[k] -= (self.learning_rate * corr_first_momentum) / (corr_second_momentum**0.5 + self.epsilon)

        return params
