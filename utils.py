import numpy as np


def generate_batch(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    ix = 0

    if shuffle:
        shuff = np.random.permutation(N)
        X = X[shuff]
        y = y[shuff]

    while ix < N:
        if ix + batch_size <= N:
            yield X[ix:ix + batch_size], y[ix:ix + batch_size]
        else:
            yield X[ix:], y[ix:]

        ix += batch_size


def classification_accuracy(y_true, y_pred):
    if (y_true.shape[-1] == y_pred.shape[-1]):
        y_true = np.argmax(y_true, axis=-1)

    y_pred = np.argmax(y_pred, axis=-1)
    y_true = y_true.reshape(-1)

    return (np.float32(np.sum(y_pred == y_true)) / y_pred.shape[0]) * 100


def confusion_matrix(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    uniq = len(set(y_true))

    mat = np.zeros((uniq, uniq), dtype=np.int32)

    for t, p in zip(y_true, y_pred):
        mat[t][p] += 1

    return mat
