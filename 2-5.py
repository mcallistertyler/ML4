# Develop a fully-connected neural network for solving the problem of classifying handwritten digits.
from random import randint
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=2017)

loss_values = []
# XOR
# x=np.array(([1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=np.float128)
# y=np.array(([0],[1],[1],[0]), dtype=np.float128)

class NN:
    def __init__(self, x, y, lr):
        self.x = x
        self.weights1 = 2*np.random.random((x.shape[1],32)) - 1
        self.weights2 = 2*np.random.random((32,32)) - 1
        self.weights3 = 2*np.random.random((32, y.shape[1])) - 1
        self.learning_rate = lr
        self.y = y 

    def forward_prop(self, training_data):
        self.layer1 = logistic_function(np.dot(training_data, self.weights1))
        self.layer2 = logistic_function(np.dot(self.layer1, self.weights2))
        self.output = logistic_function(np.dot(self.layer2, self.weights3))

    def back_prop(self, training_data):
        plotted_error = error(self.output, self.y)
        self.delta = loss_function(self.output, self.y)
        self.delta2 = self.delta.dot(self.weights3.T) * logistic_deriv(self.layer2)
        self.delta3 = self.delta2.dot(self.weights2.T) * logistic_deriv(self.layer1)
        self.weights3 = self.weights3 - np.multiply(self.learning_rate, self.layer2.T.dot(self.delta))
        self.weights2 = self.weights2 - np.multiply(self.learning_rate, self.layer1.T.dot(self.delta2))
        self.weights1 = self.weights1 - np.multiply(self.learning_rate, training_data.T.dot(self.delta3))

def one_hot(test_data):
    lr = np.arange(10)
    converted_data = []
    for label in test_data:
        one_hot = (lr==label).astype(np.int)
        converted_data.append(one_hot)
    return converted_data

def loss_function(output_y, target_y):
    n_examples = target_y.shape[0]
    res = output_y - target_y
    loss = res / n_examples
    return loss

def error(output_y, target_y):
    n_samples = target_y.shape[0]
    logp = -np.log(output_y[np.arange(n_samples), target_y.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    loss_values.append(loss)
    return loss

# Activation function
def logistic_function(z):
    return 1.0 / (1.0 + np.exp(-z))

# Derivative function
def logistic_deriv(z):
    return np.multiply(z,(1.0 - z))

## to get f-measure, precision, recall, accuracy
def confusion_matrix(data, test):
    cm = np.zeros((10,10), int)
    for x in range(0, len(X_test)):
        network.forward_prop(X_test[x])
        pred = np.argmax(network.output)
        target = test[x]
        cm[pred, int(target)] +=1
    return cm
    
def recall(test, cm):
    row = cm[test, :]
    return cm[test, test] / row.sum()

def precision(test, cm):
    col = cm[:, test]
    return cm[test, test] / col.sum()

def predict(X_data,y_data):
    len_test_examples = len(X_data)
    correct_classification = 0
    for i in range(0, len(X_data)):
        network.forward_prop(X_data[i])
        if y_data[i] == np.argmax(network.output):
            correct_classification += 1
    return (correct_classification / len_test_examples) * 100

ohy_train = np.array(one_hot(y_train))
network = NN(X_train, ohy_train, 0.4)
print('Starting training...')
for i in range(10000):
    network.forward_prop(X_train)
    network.back_prop(X_train)
print('Training Complete')

train_classification = 0
test_classification = 0
len_test_examples = len(X_test)
print('Train Accuracy', predict(X_train, y_train))
print('Test Accuracy:', predict(X_test, y_test))
cm = confusion_matrix(X_test, y_test)
print(cm)
#Recall - out of all the positive classes, how much we predicted correctly - tp/tp+fn
#Precision - out of all classes, how much we predicted correctly - tp/tp+fp
for x in range(10):
    print('---------------------')
    print('Digit:', x)
    print('Recall:', recall(x, cm))
    print('Precision', precision(x,cm))
    print('Fmeasure', (2*recall(x,cm)*precision(x,cm)) / (recall(x,cm) + precision(x,cm)))
    print('---------------------')
# network.forward_prop(x[2])
# print(network.output)
# network.forward_prop(x[3])
# print(network.output)
# plt.show()
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
