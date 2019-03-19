import numpy as np
from matplotlib import pyplot as plt

# XOR
x=np.array(([0,0],[0,1],[1,0],[1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)

class NN:
    def __init__(self, x, y):
        self.input = x 
        self.weights1 = np.random.rand(2,3)
        self.weights2 = np.random.rand(3,4)
        self.weights3 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(self.y.shape)
    
    def forward_prop(self):
        self.layer1 = logistic_function(np.dot(self.input, self.weights1))
        self.layer2 = logistic_function(np.dot(self.layer1, self.weights2))
        self.output = logistic_function(np.dot(self.layer2, self.weights3))
        return self.output

    def back_prop(self):
        delta = loss_function(self.y, self.output) * logistic_deriv(self.output)
        e_weights3 = np.dot(delta, self.weights3.T)
        d_weights3 = np.dot(e_weights3, logistic_deriv(self.weights3))

        delta2 = loss_function(self.y, self.output) * logistic_deriv(self.layer2)
        e_weights2 = np.dot(delta2, self.weights2.T)
        d_weights2 = np.dot(e_weights2, logistic_deriv(self.weights2))

        # Weight update
        # self.weights1 -= 0.1 * self.weights1 * d_weights1
        # self.weights2 -= 0.1 * self.weights2 * d_weights2
        self.weights3 -= 0.1 * np.dot(self.weights3.T, d_weights3)
        self.weights2 -= 0.1 * np.dot(self.weights2.T, d_weights2)
        print(self.weights2)

    def network_architecture(self):
        print("Input layer", self.input)
        print("Output layer", self.output)
        print("Target output", self.y)

# Activation function
def logistic_function(z):
    return 1 / (1 + np.exp(-z))

# Derivative function
def logistic_deriv(z):
    return logistic_function(z) * (1 - logistic_function(z))

# Squared loss function
def loss_function(target_y, output_y):
    return 0.5 * (target_y - output_y)**2

network = NN(x, y)

for i in range(1):    
    network.forward_prop()
    network.back_prop()

network.network_architecture()
