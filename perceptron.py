from collections import Callable
import numpy as np


class Perceptron:
    def __init__(self, no_inputs: int, initial_weights: np.array = None, bias: float = 1):
        if initial_weights is None:
            initial_weights = np.random.rand(no_inputs, 1)
        elif (initial_weights is not None) and (len(initial_weights) != no_inputs):
            raise ValueError("The size of the passed initial weights does not match the specified number of inputs.")

        self.weights = initial_weights
        self.bias = bias
        self.trained = False

    def train(self, X: np.array, y: np.array, learning_rate: float, epochs: int) -> None:
        # TODO
        self.trained = True
        pass

    def predict(self, X: np.array, activation_function: Callable) -> np.array:
        if not self.trained:
            raise ValueError("Perceptron has not been trained yet.")
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError("Number of columns of X do not match number of weights.")

        predictions = X @ self.weights
        predictions += self.bias

        return activation_function(predictions)

    def loss(self, X: np.array) -> float:
        # TODO
        # self.weights
        loss = 0.0

        return loss
