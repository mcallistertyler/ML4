import numpy as np
from matplotlib import pyplot as plt

# XOR
x=np.array(([1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float) ##prepend inputs with bias
y=np.array(([0],[1],[1],[0]), dtype=float)

loss_values = []

class NN:
    def __init__(self):
        self.weights1 = 2*np.random.random((3,3)) - 1
        self.weights2 = 2*np.random.random((3,1)) - 1
        
    def forward_prop(self, training_data):
        self.layer1 = logistic_function(np.dot(training_data, self.weights1))
        self.output = logistic_function(np.dot(self.layer1, self.weights2))

    def back_prop(self, training_data, test_data):
        error = loss_function(test_data[0], self.output[0])
        self.delta = [np.multiply(error,logistic_deriv(self.output[0]))]
        self.delta = np.asarray(self.delta)
        self.delta = self.delta[np.newaxis,:]
        self.delta2 = self.delta.dot(self.weights2.T) * logistic_deriv(self.layer1)
        # self.delta1 = self.delta2.dot(self.weights1.T) *  logistic_deriv(training_data) # left this here as a reminder to NOT do this
        self.weights2 = self.weights2 + np.multiply(0.2, self.layer1.T[:,np.newaxis].dot(self.delta))
        self.weights1 = self.weights1 + np.multiply(0.2, training_data.T[:,np.newaxis].dot(self.delta2))

# Activation function
def logistic_function(z):
    return 1.0 / (1.0 + np.exp(-z))

# Derivative function
def logistic_deriv(z):
    return np.multiply(z,(1.0 - z))

# Squared loss function
def squared_loss_function(target_y, output_y):
    loss = target_y - output_y
    return 0.5 * (loss**2)

def loss_function(target_y, output_y):
    loss = target_y - output_y
    loss_values.append(loss)
    return loss

network = NN()
for i in range(10000):
    for j in range(0,len(x)):
        network.forward_prop(x[j])
        network.back_prop(x[j],y[j])

abs_loss = [ abs(x) for x in loss_values ]
# print(abs_loss)
epochs_trained = range(1, len(loss_values) + 1)
plt.plot(abs_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
network.forward_prop(x[0])
print(network.output)
network.forward_prop(x[1])
print(network.output)
network.forward_prop(x[2])
print(network.output)
network.forward_prop(x[3])
print(network.output)
plt.show()
