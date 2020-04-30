import numpy as np
from perceptron import Perceptron
from collections import Callable

class NeuralNetwork:

    class Layer:
        def __init__(self, no_nodes: int):
            self.nodes = 

    def __init__(self, no_inputs: int, no_outputs: int,
                 no_hidden_layers: int, nodes_per_hidden_layer: np.array,
                 activation_function: Callable):
        weights_per_layer =
        self.nodes = [[Perceptron(no_inputs = ) for i in range(0, )]]

        # For output layer
        if no_outputs == 1:
            # use sigmoid activation
        else:
            # use softmax activation

    def train(self, X: np.array, y: np.array, learning_rate: float, epochs: int) -> None:
        # TODO
        pass

    def predict(self, X: np.array) -> np.array:
        # TODO
        pass

    def plot_network(self):
        # TODO
        pass