import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
#print(train_test_split(digits, shuffle=False))
# XOR
x=np.array(([0,0],[0,1],[1,0],[1,1]), dtype=np.float128)
y=np.array(([0],[1],[1],[0]), dtype=np.float128)

class NN:
    def __init__(self, x, y):
        self.weights1 = np.random.normal(loc=0, scale=1, size=(2,3))
        self.weights2 = np.random.normal(loc=0, scale=1, size=(3,4))
        self.weights3 = np.random.normal(loc=0, scale=1, size=(4,1))
    
    def forward_prop(self, training_data):
        self.layer1 = relu(np.dot(training_data, self.weights1))
        self.layer2 = relu(np.dot(self.layer1, self.weights2))
        self.output = relu(np.dot(self.layer2, self.weights3))
        return self.output

    def back_prop(self, training_data, y):
        self.delta = loss_function(y, self.output) * relu_derivate(self.output)
        self.e_weights3 = self.delta.dot(self.weights3.T)
        self.d_weights3 = self.e_weights3 * relu_derivate(self.layer2)

        self.e_weights2 = self.d_weights3.dot(self.weights2.T)
        self.d_weights2 = self.e_weights2 * relu_derivate(self.layer1)

        self.e_weights1 = self.d_weights2.dot(self.weights1.T)
        self.d_weights1 = self.e_weights1 * relu_derivate(training_data)

        self.weights1 -= 0.09 * training_data.T.dot(self.d_weights1)
        self.weights2 -= 0.09 * self.layer1.T.dot(self.d_weights2)
        self.weights3 -= 0.09 * self.layer2.T.dot(self.d_weights3)

    def network_architecture(self):
        print('-----------Architecture-----------')
        print("Inputs", x)
        print("Weights1", self.weights1)
        print("Activations at layer 1", self.layer1)
        print("Weights2", self.weights2)
        print("Activations at layer 2", self.layer2)
        print("Weights3", self.weights3)
        print('--------------------------------\n')

    def predict(self, network, input_data):
        print(input_data)
        print('Prediction:', network.forward_prop(input_data))

# Activation function
def logistic_function(z):
    return 1.0 / (1.0 + np.exp(-z))

# Derivative function
def logistic_deriv(z):
    return z * (1.0 - z)

def relu(x):
    return (1-np.exp(-2*x))/(1 + np.exp(-1*x))


def relu_derivate(x):
    return (1+ relu(x))*(1-relu(x))

# Squared loss function
def loss_function(target_y, output_y):
    loss = target_y - output_y
    return 0.5 * np.power(loss,2)

network = NN(x, y)
training_data_counter = 0
for i in range(1000):
    #one vector of input at a time
    #e.g pass through [0, 0] do forward prop then back prop
    # then switch to next training example
    # Keep doing this continually??
    
    for j in range(0, len(x)):
        network.forward_prop(x[j])
        network.back_prop(x[j],y[j])

network.network_architecture()
print('----------Predicition----------')
for j in range(0, len(x)):
    network.predict(network, x[j])
print('--------------------------------')
