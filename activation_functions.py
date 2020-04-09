import numpy as np


def sigmoid(data: np.array) -> np.array:
    return 1 / (1 + np.exp(-data))


def tanh(data: np.array) -> np.array:
    return (2 / (1 + np.exp(-2*data))) - 1


def relu(data: np.array) -> np.array:
    return np.maximum(0, data)


def leaky_relu(data: np.array, alpha: float = 0.3) -> np.array:
    # Default for alpha as per Keras (https://keras.io/layers/advanced-activations/)
    return np.maximum(alpha * data, data)


def swish(data: np.array, beta: int = 1) -> np.array:
    # https://arxiv.org/pdf/1710.05941.pdf
    return data * sigmoid(beta * data)
