import numpy as np

from activations import *
from losses import *
from optimizers import *
from utils import *


def _init_weights_and_bias(shape_in, shape_out):
    return np.random.randn(shape_in, shape_out) * np.sqrt(2 / (shape_in + shape_out)), np.zeros((1, shape_out))


class NeuralNet:

    def __init__(self, layers, input_dim, activations=None, keep_probs=None, lalpha=None):
        if keep_probs is None:
            keep_probs = [1.0] * len(layers)

        if lalpha is None:
            lalpha = [0.2] * len(layers)

        if activations is None:
            activations = ['linear'] * len(layers)

        assert len(layers) == len(activations) == len(keep_probs) == len(lalpha)

        self.layers = layers
        self.activations = activations
        self.keep_probs = keep_probs
        self.input_dim = input_dim
        self.lalpha = lalpha

    def compile(self, optimizer, loss, learning_rate):
        self.params = {}
        self.model_type = 'classification'

        shape_in = self.input_dim

        # Parameters
        for l in range(len(self.layers)):
            shape_out = self.layers[l]
            self.params[f'W{l}'], self.params[f'b{l}'] = _init_weights_and_bias(shape_in, shape_out)

            shape_in = shape_out

        # Optimizer
        if not isinstance(optimizer, str):
            self.optimizer = optimizer

        else:
            if optimizer == 'sgd':
                self.optimizer = SGD(learning_rate)

            elif optimizer == 'sgd_m':
                self.optimizer = SGDMom(learning_rate)

            elif optimizer == 'adagrad':
                self.optimizer = AdaGrad(learning_rate)

            elif optimizer == 'rmsprop':
                self.optimizer = RMSProp(learning_rate)

            elif optimizer == 'adam':
                self.optimizer = Adam(learning_rate)

        # Loss
        if loss == 'mean_squared_error':
            self.loss_fn = MSE()
            self.model_type = 'regression'
        elif loss == 'binary_crossentropy':
            self.loss_fn = BinaryCrossEntropy()
        elif loss == 'categorical_crossentropy':
            self.loss_fn = SoftmaxCrossEntropy()
        elif loss == 'sparse_categorical_crossentropy':
            self.loss_fn = SparseSoftmaxCrossEntropy()

    def _forward(self, inp_layer, dropout=False):
        caches = []

        for l in range(len(self.layers)):
            W, b = self.params[f'W{l}'], self.params[f'b{l}']
            activation = self.activations[l]
            keep_prob = 1.0

            if dropout:
                keep_prob = self.keep_probs[l]

            # Linear forward
            out_layer = inp_layer.dot(W)
            out_layer += b
            linear_cache = (inp_layer, W)
            activation_cache = None

            # Activation
            if activation == 'relu':
                out_layer, activation_cache = relu_forward(out_layer)

            elif activation == 'sigmoid':
                out_layer, activation_cache = sigmoid_forward(out_layer)

            elif activation == 'lrelu':
                out_layer, activation_cache = lrelu_forward(out_layer, self.lalpha[l])

            elif activation == 'tanh':
                out_layer, activation_cache = tanh_forward(out_layer)

            # Dropout
            dropout_mask = np.random.rand(out_layer.shape[0], out_layer.shape[1]) < keep_prob
            out_layer = out_layer * dropout_mask
            out_layer /= keep_prob

            inp_layer = out_layer
            caches.append((linear_cache, activation_cache, dropout_mask))

        return out_layer, caches

    def train(self, X_train, y_train, epochs=1, batch_size=1, validation_data=None):

        for epoch in range(epochs):
            loss = 0.0
            itr = 0

            for batch_X, batch_y in generate_batch(X_train, y_train, batch_size):

                # Forward Propagation
                out_layer, caches = self._forward(batch_X, dropout=True)
                loss += self.loss_fn.forward(out_layer, batch_y)
                itr += 1

                # Backward Propagation
                grads = {}
                dA = self.loss_fn.backward()

                for l in reversed(range(len(self.layers))):
                    linear_cache, activation_cache, dropout_cache = caches[l]
                    activation = self.activations[l]

                    # Dropout Backward
                    dA *= dropout_cache
                    dZ = dA

                    # Activation Backward
                    if activation == 'relu':
                        dZ = relu_backward(dA, activation_cache)
                    elif activation == 'sigmoid':
                        dZ = sigmoid_backward(dA, activation_cache)
                    elif activation == 'lrelu':
                        dZ = lrelu_backward(dA, activation_cache)
                    elif activation == 'tanh':
                        dZ = tanh_backward(dA, activation_cache)

                    X, W = linear_cache
                    grads[f'W{l}'] = X.T.dot(dZ) / X.shape[0]
                    grads[f'b{l}'] = np.sum(dZ, axis=0, keepdims=True) / X.shape[0]
                    dA = dZ.dot(W.T)

                self.params = self.optimizer.optimize(self.params, grads, itr)

            loss /= itr

            if self.model_type == 'regression':
                print(f'Epoch {epoch}/100 \t Loss: {loss:.4f}')

            else:
                train_preds = self.predict(X_train)
                train_acc = classification_accuracy(y_train, train_preds)

                if validation_data:
                    val_preds = self.predict(validation_data[0])
                    valid_acc = classification_accuracy(validation_data[1], val_preds)

                    print(f'Epoch {epoch}/100 \t Loss: {loss:.4f} \t Train_acc: {train_acc:.4f} \t Validation_acc: {valid_acc:.4f}')

                else:
                    print(f'Epoch {epoch}/100 \t Loss: {loss:.4f} \t Train_acc: {train_acc:.4f}')

    def predict(self, X_test):
        out_layer, _ = self._forward(X_test, dropout=False)

        return out_layer
