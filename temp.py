import numpy as np

from neural_net import NeuralNet


EPSILON = 1e-7


def convert_dict_to_vec(di, keys):
    ret = []

    for k in keys:
        new_vec = np.reshape(di[k], (-1, 1)).mean()
        ret.append(new_vec)

    return np.array(ret).reshape((-1, 1))


X = np.random.randn(2, 10)
y = np.random.randint(0, 5, size=(2))
one_hot = np.zeros_like(X)
one_hot[range(X.shape[0]), y] = 1
y = one_hot

net = NeuralNet([10, 10, 10, 10, 10], ['linear', 'lrelu', 'tanh', 'relu', 'sigmoid'], [1.0, 1.0, 1.0, 1.0, 1.0], input_dim=10, lalpha=[0.2, 0.2, 0.2])
net.compile(optimizer='sgd', loss='categorical_crossentropy', learning_rate=1e-2)
grads = net.train(X, y, return_grads=True, return_loss=False)

numeric_grads = np.zeros((len(net.parameters), 1))

keys = list(net.parameters.keys())

for i, k in enumerate(keys):
    net.update_parameter(k, EPSILON)
    loss_plus = net.train(X, y, return_grads=False, return_loss=True)
    net.update_parameter(k, -2 * EPSILON)
    loss_minus = net.train(X, y, return_grads=False, return_loss=True)
    net.update_parameter(k, EPSILON)

    numeric_grads[i] = (loss_plus - loss_minus) / (2 * EPSILON)


def check_gradient(grads, numeric_grads):
    print(numeric_grads)
    print(grads)
    numerator = np.linalg.norm(numeric_grads - grads)
    denominator = np.linalg.norm(numeric_grads) + np.linalg.norm(grads)
    print(numerator)
    print(denominator)
    print(numerator / denominator)


grads = convert_dict_to_vec(grads, keys)

check_gradient(np.float32(grads), np.float32(numeric_grads))
