import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()


# Dense Layer: First Layer in which you find the dot product of the inputs and weights then add the bias
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases


# Rectified Linear makes a value lower than 0 equal to 0, and any value that is higher than 0 stays the same
class ReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)


# Activation Function used on the output layer.
"""
First you expeneniate the inputs which is eulers number to the power of the input (To prevent overflow we take the max input 
and subtract it form every number before expentiating.)
Second you normalize the values which is just taking a input then dividing that by the sum of all inputs. 
"""


class Softmax:
    def forward(self, input):
        exp_value = np.exp(input - np.max(input, axis=1, keepdims=True))
        norm_value = input / np.sum(exp_value, axis=1, keepdims=True)
        self.output = norm_value


# Spiral Data Set
X, y = spiral_data(100, 3)

# Create first layer object with 2 inputs and 3 neurons
layer1 = LayerDense(2, 3)

# Each neuron needs to be activated
act1 = ReLU()

# 2nd layers has the same number of inputs as the previous layer's outputs, and is connected to the output layer with 3 neurons
layer2 = LayerDense(3, 3)

# Softmax will only activate the output layer
act2 = Softmax()

# Send data into first layer
layer1.forward(X)

# Activate the outputs
act1.forward(layer1.output)

# Input layer 1 outputs into layer 2
layer2.forward(act1.output)

# Use softmax to activate the output layer values
act2.forward(layer2.output)

# print first 5 values
print(act2.output[:5])
