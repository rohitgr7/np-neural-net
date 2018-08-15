import numpy as np


class MSE:

    def forward(self, logits, targets):
        self.logits = logits
        self.targets = targets

        return np.mean((self.logits - self.targets) ** 2)

    def backward(self):
        return 2 * (self.logits - self.targets)


class BinaryCrossEntropy:
    def forward(self, logits, targets):
        self.logits = logits
        self.targets = targets

        return -np.mean(self.targets * np.log(self.logits) + (1 - self.targets) * np.log(1 - self.logits))

    def backward(self):
        return (self.logits - self.targets) / (self.logits * (1 - self.logits))


class SoftmaxCrossEntropy:

    def forward(self, logits, targets):
        self.logits = logits
        self.targets = targets
        self.probs = np.exp(self.logits) / np.sum(np.exp(self.logits), axis=1, keepdims=True)

        return np.mean(np.sum(np.log(self.probs[:, self.targets]), axis=1))

    def backward(self):
        return self.probs - self.targets


class SparseSoftmaxCrossEntropy:

    def forward(self, logits, targets):
        self.logits = logits
        one_hot_targets = np.zeros_like(logits)[:, targets] = 1
        self.targets = one_hot_targets

        self.softmax_loss = SoftmaxCrossEntropy()

        return self.softmax_loss.forward(self.logits, self.targets)

    def backward(self):
        return self.softmax_loss.backward()
