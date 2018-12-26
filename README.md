# Neural-Net

An implementation of neural-net with just numpy.

## Requirements
* [Numpy](http://www.numpy.org/)

Use `pip install -U numpy` to install numpy.

## Guide
### Activations
Activation functions available are:
* ReLU
* Sigmoid
* Tanh
* LReLU

### Losses
Loss functions available are:
* MSE
* BinaryCrossEntropy
* SoftmaxCrossEntropy
* SparseSoftmaxCrossEntropy

### Optimizers
Loss functions available are:
* SGD
* SGDMom
* AdaGrad
* RMSProp
* Adam

### Others
* You can pass keep_probs (list) which can initialize dropouts in layers.
* For activations pass a list of activation functions as strings (for eg. ['relu', 'lrelu', 'sigmoid', 'tanh', 'linear']) during initializations.
* Either pass a string for optimizer (for eg. 'adam') along with learning_rate or assign your own optimizer from optimizers.py and pass that.
* Similarly for loss function, either pass a string (for eg. 'mean_squared_error') or assign a loss function from losses.py.
* For training use model.fit() and pass the required arguments.
* For predicion use model.predict() and pass the required arguments.


