import numpy as np
from matplotlib import pyplot as plt

# XOR
x=np.array(([0,0],[0,1],[1,0],[1,1]))
y=np.array(([0],[1],[1],[0]))
class NN:
    def __init__(self, x, y):
        self.input = x 
        self.weights1 = np.random.rand(2,3)
        self.weights2 = np.random.rand(3,4)
        self.weights3 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)
    
    def forward_prop(self,current_example):
        self.layer1 = logistic_function(np.dot(self.input[current_example], self.weights1))
        self.layer2 = logistic_function(np.dot(self.layer1, self.weights2))
        self.output = logistic_function(np.dot(self.layer2, self.weights3))
        return self.output

    def back_prop(self, current_example):
        delta = loss_function(self.y[current_example], self.output) * logistic_deriv(self.output)
        e_weights3 = np.dot(delta, self.weights3.T)
        d_weights3 = e_weights3 * logistic_deriv(self.layer2)

        #delta2 = loss_function(self.y, self.output) * logistic_deriv(self.layer1)
        e_weights2 = np.dot(d_weights3, self.weights2.T)
        d_weights2 = e_weights2 * logistic_deriv(self.layer1)

        # delta3 = loss_function(self.y, self.output) * logistic_deriv(self.layer1)
        e_weights1 = np.dot(d_weights2, self.weights1.T)
        d_weights1 = e_weights1 * logistic_deriv(self.input[current_example])

        self.weights3 = self.weights3 - 0.1 * self.layer2.T.dot(d_weights3)
        self.weights2 = self.weights2 - 0.1 * self.layer1.T.dot(d_weights2)
        self.weights1 = self.weights1 - 0.1 * self.input[current_example].T.dot(d_weights1)

    def network_architecture(self):
        print("Inputs", x)
        print("Weights1", self.weights1)
        print("Activations at layer 1", self.layer1)
        print("Weights2", self.weights2)
        print("Activations at layer 2", self.layer2)
        print("Weights3", self.weights3)
        print("Output layer", self.output)
        print("Target output", self.y)

def predict(network, x):
    for i in range(0, len(x)):
        print(x[i])
    # network.forward_prop()

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
training_data_counter = 0 # Hacky
for i in range(10000):
    #one vector of input at a time
    #e.g pass through [0, 0] then do forward prop then back prop
    # then switch to next training example
    # Keep doing this continually??
    network.forward_prop(training_data_counter)
    network.back_prop(training_data_counter)
    training_data_counter += 1
    if training_data_counter == 3:
        training_data_counter = 0

predict(network,x)
network.network_architecture()
