import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases


class ReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)


layer1 = LayerDense(2, 5)
act1 = ReLU()

layer1.forward(X)
act1.forward(layer1.output)

print(act1.output)
