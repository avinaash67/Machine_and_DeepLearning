# Description:
# Perceptron does not contain any hidden layers in a neural network

import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1-x)


training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T  # Measured outputs, this is used for training

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1  # random values are taken between 0 and 1
print("Random Synaptic weights: ")
print(synaptic_weights)


for iteration in range(20000):

    input_layer = training_inputs

    output_layer = sigmoid(np.dot(input_layer, synaptic_weights))  # This is yhat (predicted output)

    error = training_outputs - output_layer

    adjustments = error * sigmoid_derivative(output_layer)

    synaptic_weights += np.dot(input_layer.T, adjustments)


print("Synaptic weights after training")
print(synaptic_weights)

print("outputs after training")
print(output_layer)









