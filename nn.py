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
        # Error
        e_weights3 = loss_function(self.y, self.output) * logistic_deriv(self.output)
        d_weights3 = np.dot(self.layer2.T, e_weights3)        
        e_weights2 = np.dot(e_weights3, self.weights3.T) * logistic_deriv(self.layer2)
        d_weights2 = np.dot(self.layer1.T, e_weights2)        
        e_weights1 = np.dot(e_weights2, self.weights2.T) * logistic_deriv(self.layer1)
        d_weights1 = np.dot(self.input.T, e_weights1)        
        
        # print('En', d_weights1)
        # print('To', d_weights2)
        # print('Tre', d_weights3)

        # Weight update
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3

    def network_architecture(self):
        print("Input layer", self.input)
        print("Output layer", self.output)
        print("Target output", self.y)

# Activation function
def logistic_function(z):
    return 1 / (1 + np.exp(-z))

# Derivative function
def logistic_deriv(z):
    return z * (1 - z)

# Squared loss function
def loss_function(target_y, output_y):
    return 2 * (target_y - output_y)

network = NN(x, y)

for i in range(100):    
    network.forward_prop()
    network.back_prop()

network.network_architecture()
