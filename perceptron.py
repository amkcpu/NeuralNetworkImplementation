from collections import Callable
import numpy as np


class Perceptron:
    def __init__(self, no_inputs: int, initial_weights: np.array = None):
        self.weights = None
        if (initial_weights is not None) and (len(initial_weights) == no_inputs):
            self.weights = initial_weights
        elif (initial_weights is not None) and (len(initial_weights) != no_inputs):
            raise ValueError("The size of the passed initial weights does not match the specified number of inputs.")

        self.bias = 1

    def train(self, X: np.array, y: np.array, learning_rate: float, epochs: int) -> None:
        pass

    def predict(self, X: np.array, activation_function: Callable) -> np.array:
        if self.weights is None:
            raise ValueError("Perceptron has not been trained yet.")
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError("Number of columns of X do not match number of weights.")

        predictions = X @ self.weights
        predictions += self.bias

        return activation_function(predictions)
