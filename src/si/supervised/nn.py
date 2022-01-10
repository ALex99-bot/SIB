import numpy as np
from abc import ABC, abstractmethod
from .model import Model
from ..util.metrics import mse, mse_prime


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_erros, learing_rate):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """Fully Connected layer"""
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.zeros((1, output_size))

    def set_weights(self, weights, bias):
        if weights.shape != self.weights.shape:
            raise ValueError(f"Shapes mismatch {weights.shape} and {self.weights.shape}")
        if bias.shape != self.bias.shape:
            raise ValueError(f"Shapes mismatch {weights.shape} and {self.weights.shape}")
        self.weights = weights
        self.bias = bias

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Computes dE/dW, dE/dB for a given output_error=dE/dY
        Returns input_erros=dE/dX to fedd the previous layer.
        """
        # Compute the weights erros dE/dW = X.T * dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # Compute the bias error dE/dB = dE/dY
        bias_error = np.sum(output_error, axis=0)
        # Error dE/dX to pass on to the previous layer
        input_error = np.dot(output_error, self.weights.T)
        # Update parameters
        self.weights -= learning_rate*weights_error
        self.bias -= learning_rate*bias_error
        return input_error


class Activation(Layer):
    def __init__(self, activation):
        self.activation = activation

    def foward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        # learning_rate is not used because thre is no "learnable" parameters.
        # Only passed the error do the previous layer
        return np.multiply(self.activation.prime(self.input), output_error)


class NN(Model):
    def __init__(self, epochs=1000, lr=0.001, verbose=True):
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

    def fit(self, dataset):
        X, y = dataset.getXy()
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            output = X

            # forward propagation
            for layer in self.layers:
                output = layer.forward(output)

            # backward propagation
            error = self.loss_prime(y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            # calculate average error
            err = self.loss(y, output)
            self.history[epoch] = err
            if self.verbose:
                print(f'epoch {epoch + 1}/{self.epochs} error={err}')
        if not self.verbose:
            print(f'error={err}')
        self.is_fitted = True

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        self.is_fitted = True
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def cost(self, X=None, y=None):
        assert self.is_fitted, 'Model must be fit before predict'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.Y
        output = self.predict(X)
        return self.loss(y, output)


class Conv2D:
    ...


class Pooling2D(Layer):
    def __init__(self, size=2,stride=2):
        self.size = size
        self.stride = stride

    def pool(self):
        pass

    def dpool(self):
        pass

    def forward(self, input):
        self.X_shape = input.shape
        n, h, w, d = input.shape
        h_out = (h.self.size)/self.stride+1
        w_out = (w.self.size) / self.stride + 1
        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invaid output dimension!')
        h_out, w_out = int(h_out), int(w_out)

        X_reshaped = input.reshape(n*d, h, w, 1)
        self.X_col = im2col(X_reshaped, self.size, padding=0, stride=self.stride)

        out, self.max_idx = self.pool(self.X_col)
        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(1, 2 ,3, 0)
        return out

    def backward(self, output_erros, learing_rate):
        n, w, h, d = self.X_shape

        dX_col = np.zeros_like(self.X_col)
        dout_col = output_error.transpose(1, 2, 3, 0).ravel()
        dX = self.dpool(dX_col, dout_col, self.max_idx)
        dX = self.col2im(dX, (n*d, h, w, 1),
                         self.size, self.size, padding=0, stride=self.stride)
        dX = dX.reshape(self.X_shape)
        return dX


class MaxPooling(Pooling2D):
    def pool(self):
        pass

    def dpool(self):
        pass
