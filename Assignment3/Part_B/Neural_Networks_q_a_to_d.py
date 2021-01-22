#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Load Libs'''
import numpy as np
import pandas as pd
import os.path
from pathlib import Path
from collections import Counter
import random
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from time import time
import matplotlib.pyplot as plt


# In[2]:


'''Load Data And Identify Classes'''
PathTrain =  os.getcwd() + "/data/train.csv"
PathTest =  os.getcwd() + "/data/test.csv"
DFTrain = pd.read_csv(PathTrain, header=None)
DFTest = pd.read_csv(PathTest, header=None)
totalClasses = len(Counter(DFTrain.iloc[:,-1]))


# In[3]:


DTrain = DFTrain.iloc[:,:-1].to_numpy()
LabelsTrain = np.array(DFTrain.iloc[:,-1])
DTest = DFTest.iloc[:,:-1].to_numpy()
LabelsTest = np.array(DFTest.iloc[:,-1])


# In[4]:


'''One Hot Encoding for Train and Test Labels'''
oneHotTrainLables = np.zeros((len(DFTrain), totalClasses))

for i in range(len(DFTrain)):
    oneHotTrainLables[i][DFTrain.iloc[i, -1]] = 1


# In[5]:


train_accuracy_list = np.zeros(5)
test_accuracy_list = np.zeros(5)
train_accuracy_adaptive_lr = np.zeros(5)
test_accuracy_adaptive_lr = np.zeros(5)
time_fixed_lr = np.zeros(5)
time_adaptive_lr = np.zeros(5)


# In[6]:


def sigmoid(z):
    #Sigmoid Function
    return 1/(1+np.exp(-z))

def ReLU(z):
    #Rectified Linear Unit
    return np.maximum(0,z)

def delSigmoid(z):
    #Derivative of Sigmoid
    #oj = sigmoid(z)  This was a mistake
    return np.multiply(z, (1-z))

def delReLU(z):
    #Derivative of ReLU
    z = np.matrix(z)
    z[z > 0] = 1
    z[z <= 0] = 0    
    return z
    
    
class Layer:    
    '''
    Class: Layer
    
    Params: 
    perceptron_units = Number of perceptrons in the layer
    n_inputs = Number of inputs to the layer
    activation_function = Function to introduce Non Linearity
    layer_type = this is to keep track of whether the Layer is Hidden/Output
    stn_dev = standard deviation of weights with mean = 0
    
    Returns: A Layer Object
    '''
    def __init__(self, perceptron_units, n_inputs, activation_func, layer_type, stn_dev):
        self.type = layer_type
        self.activation = activation_func
        self.perceptron_units = perceptron_units
        self.inputs = None
        self.output = None
        self.weights = np.random.normal(0,stn_dev, (n_inputs) * perceptron_units).reshape(n_inputs, perceptron_units)
        self.bias = np.random.normal(0,stn_dev, perceptron_units)
        self.delta = None
        
    def __repr__(self):
        representation = (self.type, self.perceptron_units, self.weights.shape, self.bias.shape, self.activation)
        return "<%s Layer | Num_Perceptrons_Units = %d, Weights = %s, Bias = %s, Activation = %s>" % representation

    

class NeuralNetwork:
    '''
    Class: Neural Network
    
    HyperParameters:
    list_hidden_layers = takes a list of number of perceptrons in each hidden layer
    op_layer_activation = Non Linearity function for output layer. We have used Sigmoid. but can be varied
    hidden_layers_activation = Non Linearity function for output layer. Experimentation done with ReLU and Sigmoid
    weights_sd = standard deviation of weights on layers. with mean = 0
    
    '''
    
    def __init__(self, list_hidden_layers, op_layers_activation, hidden_layers_activation, weights_sd):
        np.random.seed(525)
        self.total_layers = len(list_hidden_layers) + 1
        self.nodes_hidden_layers = list_hidden_layers
        self.layers = []
        
        for i in range(len(list_hidden_layers)):
            if i == 0:
                layer = Layer(list_hidden_layers[i], n, hidden_layers_activation, "Hidden", weights_sd)
                self.layers.append(layer)
            else:
                layer = Layer(list_hidden_layers[i], list_hidden_layers[i-1], hidden_layers_activation, "Hidden", weights_sd)                
                self.layers.append(layer)
        
        layer = Layer(r, list_hidden_layers[-1], op_layers_activation, "Output", weights_sd)
        self.layers.append(layer)
        
        
    def __repr__(self):
        layers = self.layers
        rep = ""
        print("Neural Network:")
        for i in range(len(layers)):
            rep += "Layer %d: %s\n" % (i, layers[i])
        return rep
    
    def forwardFeed(self, ip_data):
        '''
        Forward pass of input data to the output through all the layers
        '''
        layer = self.layers[0]
        layer.inputs = np.matrix(ip_data)
        #print("FF", layer.inputs.shape, np.matrix(layer.weights).shape, layer.bias.shape)
        layer.netj = np.matmul(layer.inputs, np.matrix(layer.weights)) + layer.bias
        layer.output = self.activation(layer.netj, layer.activation)
        for i in range(1, len(self.layers), 1):
            layer = self.layers[i]
            last_layer = self.layers[i-1]
            layer.inputs = last_layer.output
            layer.netj = np.matmul(layer.inputs, np.matrix(layer.weights)) + layer.bias
            layer.output = self.activation(layer.netj, layer.activation)
    
    def backwardPropagation(self, labels):
        '''
        Back propagation algorithm implementation
        '''
        output_layer = self.layers[-1]
        error = labels - output_layer.output
        delOj = self.delActivation(output_layer.output, output_layer.activation)
        output_layer.delta = -1*np.multiply(error, delOj)
        
        for i in reversed(range(self.total_layers-1)):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            error = np.matmul(next_layer.delta, next_layer.weights.T)
            delOj = self.delActivation(current_layer.output, current_layer.activation)
            current_layer.delta = np.multiply(error, delOj)
            
    def activation(self, x, activation):
        '''
        call Non Linearity function based on layer activation param
        '''
        if activation == "Sigmoid":
            return sigmoid(x)
        elif activation == "ReLU":
            return ReLU(x)
    
    def delActivation(self, x, activation):
        '''
        call Derivation of Non Linearity function based on layer activation param
        '''
        if activation == "Sigmoid":            
            return delSigmoid(x)
        
        elif activation == "ReLU":
            return delReLU(x)
        
    def updateParams(self, lr, bSize):
        '''
        update the weights and bias term of all the layers
        '''
        layers = self.layers
        for layer in layers:
            #print(layer.inputs.shape, layer.delta.shape)
            gradient = np.matmul(layer.inputs.T, layer.delta)
            layer.weights = layer.weights - (lr/bSize)*gradient
            layer.bias = layer.bias - (lr/bSize)*np.sum(layer.delta, 0)
    
            
    def meanSquaredError(self, Y, avg=True):        
        '''
        Mean Squared Error Cost computation
        '''
        div = len(Y)
        
        op_layer_labels = self.layers[-1].output
        error = Y - op_layer_labels
        error = np.square(error)
        error = np.sum(error)/(2*div)
        return error
    
    
    def fit(self, X, Labels, eta = 0.1, batch_size=100, max_epoch = 10000, epsilon=1e-06, adaptive=False):
        '''
        Train Model
        '''
        lr = eta
        data = X
        labels = Labels
        
        epoch = 0
        self.forwardFeed(data)
        error_prev = self.meanSquaredError(labels)        
        epoch_error_list = [error_prev]
        t0 = time()
        while epoch < max_epoch:
            t1 = time()
            epoch += 1
            
            data, labels = shuffle(data, labels)
            
            if adaptive:
                lr = eta/np.sqrt(epoch)
                
            for batch_start in range(0, len(data), batch_size):
                batch_end = batch_start + batch_size
                Xb = data[batch_start : batch_end]
                Yb = labels[batch_start : batch_end]
                
                self.forwardFeed(Xb)
                self.backwardPropagation(Yb)
                
                self.updateParams(lr, batch_size)
                
            t2 = time()   
            self.forwardFeed(data)
            error = self.meanSquaredError(labels)
            deltaError = error - error_prev
            epoch_error_list.append(np.abs(deltaError))            
            print("$$ Epoch: {} | Error = {} | DeltaError = {} | LR = {} | Epoch Train Time = {}Sec"
                  .format(epoch, round(error,6), round(deltaError,20), round(lr,5), round(t2-t1,2)))
            avg_deltaError = np.mean(epoch_error_list[-10:])
            if np.abs(avg_deltaError) < epsilon[1] or error < epsilon[0]:
                break
            error_prev = error
        
        t4 = time()
        print("\n%% Total Epochs ={} | Epsilon = {} | Total Learning Time = {}Min"
              .format(epoch, epsilon, round((t4-t0)/60,2)))
        return round((t4-t0)/60,2)
        
    def predict(self, data, labels):
        '''
        Model Prediction
        '''
        self.forwardFeed(data)
        outputs = self.layers[-1].output
        prediction = []
        for x in outputs:
            prediction.append(np.argmax(x))
        print(len(prediction), labels.shape)
        accu_score = accuracy_score(labels, prediction)
        #print("Accuracy Score: ", 100*accu_score)
        return 100*accu_score


def plotAccuracies(lr_type="fixed"):
    fig = plt.figure(0)
    num_perceptrons = [1,5,10,50,100]
    
    if lr_type == "fixed":
        train_accu = train_accuracy_list
        test_accu = test_accuracy_list
    elif lr_type == "adaptive":
        train_accu = train_accuracy_adaptive_lr
        test_accu = test_accuracy_adaptive_lr
    
    plt.plot(num_perceptrons, train_accu, c="tab:green", marker="o", label="Train Accuracy")
    plt.plot(num_perceptrons, test_accu, c="tab:orange", marker="x", label="Test Accuracy")
    plt.title("Train & Test Accuracies with {} learning rate".format(lr_type))
    plt.xlabel("# Perceptrons in Hidden Layer")
    plt.ylabel("% Accuracies")
    plt.legend()
    plt.show()
    fig.savefig("plots/accuracies_{}_lr.png".format(lr_type), dpi= 300, pad_inches=0.1, format="png")
    
def plotAccuraciesComparision(data_type = "Train"):
    fig = plt.figure(0)
    num_perceptrons = [1,5,10,50,100]
    
    if data_type == "Train":
        accu_fixed = train_accuracy_list
        accu_adaptive = train_accuracy_adaptive_lr
    elif data_type == "Test":
        accu_fixed = test_accuracy_list
        accu_adaptive = test_accuracy_adaptive_lr
    
    plt.plot(num_perceptrons, accu_fixed, c="tab:green", marker="o", label="Fixed Learning Rate")
    plt.plot(num_perceptrons, accu_adaptive, c="tab:orange", marker="x", label="Adaptive Learning Rate")
    plt.title("{} Accuracies with Fixed & Adaptive learning rate".format(data_type))
    plt.xlabel("# Perceptrons in Hidden Layer")
    plt.ylabel("% Accuracies")
    plt.legend()
    plt.show()
    fig.savefig("plots/comparision_{}_accuracies_fixed_adaptive_lr.png".format(data_type.lower()), dpi= 300,
                pad_inches=0.1, format="png")
    
def plotTrainingTime():
    fig = plt.figure(0)
    num_perceptrons = [1,5,10,50,100]
        
    plt.plot(num_perceptrons, time_fixed_lr, c="tab:green", marker="o", label="Fixed Learning Rate")
    plt.plot(num_perceptrons, time_adaptive_lr, c="tab:orange", marker="x", label="Adaptive Learning Rate")
    plt.title("Model Train Time for Fixed and Adaptive Learning Rates")
    plt.xlabel("# Perceptrons in Hidden Layer")
    plt.ylabel("Train Time (Mins)")
    plt.legend()
    plt.show()
    fig.savefig("plots/time_fixed_adaptive_lr.png", dpi= 300, pad_inches=0.1, format="png")
    
    fig1 = plt.figure(1)
        
    plt.plot(num_perceptrons, time_fixed_lr, marker="o", label="Fixed Learning Rate")
    plt.title("Model Train Time for Fixed Learning Rate")
    plt.xlabel("# Perceptrons in Hidden Layer")
    plt.ylabel("Train Time (Mins)")
    plt.legend()
    plt.show()
    fig1.savefig("plots/time_fixed_lr.png", dpi= 300, pad_inches=0.1, format="png")
    
    fig2 = plt.figure(2)
        
    plt.plot(num_perceptrons, time_adaptive_lr, marker="o", label="Adaptive Learning Rate")
    plt.title("Model Train Time for Adaptive Learning Rate")
    plt.xlabel("# Perceptrons in Hidden Layer")
    plt.ylabel("Train Time (Mins)")
    plt.legend()
    plt.show()
    fig2.savefig("plots/time_adaptive_lr.png", dpi= 300, pad_inches=0.1, format="png")


# In[7]:


'''Global Params'''
M = 100 #MiniBatch Size
n = len(DTrain[0])
r = totalClasses


# In[21]:


'''Global Params'''
nodesInHiddenLayers = [100]
print("@@@@@@@@@@---------Q1_b---------@@@@@@@@@@")
print("######------HiddenLayer = {}-----######\n".format(nodesInHiddenLayers))
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "Sigmoid", 0.05)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.1, max_epoch=10000, epsilon=[0.02, 1e-06])

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
train_accuracy_list[4] = AccuTrain
test_accuracy_list[4] = AccuTest
time_fixed_lr[4] = train_time
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[10]:


nodesInHiddenLayers = [50]
print("@@@@@@@@@@---------Q1_b---------@@@@@@@@@@")
print("######------HiddenLayer = {}-----######\n".format(nodesInHiddenLayers))
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "Sigmoid", 0.05)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.1, max_epoch=10000, epsilon=[0.02, 1e-06])

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
train_accuracy_list[3] = AccuTrain
test_accuracy_list[3] = AccuTest
time_fixed_lr[3] = train_time
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[11]:


'''Global Params'''
nodesInHiddenLayers = [10]
print("@@@@@@@@@@---------Q1_b---------@@@@@@@@@@")
print("######------HiddenLayer = {}-----######\n".format(nodesInHiddenLayers))
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "Sigmoid", 0.05)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.1, max_epoch=10000, epsilon=[0.02, 1e-08])

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
train_accuracy_list[2] = AccuTrain
test_accuracy_list[2] = AccuTest
time_fixed_lr[2] = train_time
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[10]:


'''Global Params'''
nodesInHiddenLayers = [5]
print("@@@@@@@@@@---------Q1_b---------@@@@@@@@@@")
print("######------HiddenLayer = {}-----######\n".format(nodesInHiddenLayers))
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "Sigmoid", 0.5)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.1, max_epoch=10000, epsilon=[0.02, 1e-06])

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
train_accuracy_list[1] = AccuTrain
test_accuracy_list[1] = AccuTest
time_fixed_lr[1] = train_time
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[13]:


'''Global Params'''
nodesInHiddenLayers = [1]
print("@@@@@@@@@@---------Q1_b---------@@@@@@@@@@")
print("######------HiddenLayer = {}-----######\n".format(nodesInHiddenLayers))
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "Sigmoid", 0.05)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.1, max_epoch=10000, epsilon=[0.02, 1e-06])

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
train_accuracy_list[0] = AccuTrain
test_accuracy_list[0] = AccuTest
time_fixed_lr[0] = train_time
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[8]:


'''Global Params'''
nodesInHiddenLayers = [100]
print("@@@@@@@@@@---------Q1_c---------@@@@@@@@@@")
print("@@@@------Adaptive Learning Rate------@@@@")
print("######------HiddenLayer = {}-----######\n".format(nodesInHiddenLayers))
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "Sigmoid", 0.05)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.5, max_epoch=10000, epsilon=[0.02, 1e-06], adaptive=True)

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
train_accuracy_adaptive_lr[4] = AccuTrain
test_accuracy_adaptive_lr[4] = AccuTest
time_adaptive_lr[4] = train_time
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[14]:


'''Global Params'''
nodesInHiddenLayers = [50]
print("@@@@@@@@@@---------Q1_c---------@@@@@@@@@@")
print("@@@@------Adaptive Learning Rate------@@@@")
print("######------HiddenLayer = {}-----######\n".format(nodesInHiddenLayers))
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "Sigmoid", 0.05)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.5, max_epoch=10000, epsilon=[0.02, 1e-06], adaptive=True)

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
train_accuracy_adaptive_lr[3] = AccuTrain
test_accuracy_adaptive_lr[3] = AccuTest
time_adaptive_lr[3] = train_time
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[20]:


'''Global Params'''
nodesInHiddenLayers = [10]
print("@@@@@@@@@@---------Q1_c---------@@@@@@@@@@")
print("@@@@------Adaptive Learning Rate------@@@@")
print("######------HiddenLayer = {}-----######\n".format(nodesInHiddenLayers))
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "Sigmoid", 0.05)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.5, max_epoch=10000, epsilon=[0.02, 1e-09], adaptive=True)

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
train_accuracy_adaptive_lr[2] = AccuTrain
test_accuracy_adaptive_lr[2] = AccuTest
time_adaptive_lr[2] = train_time
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[9]:


'''Global Params'''
nodesInHiddenLayers = [5]
print("@@@@@@@@@@---------Q1_c---------@@@@@@@@@@")
print("@@@@------Adaptive Learning Rate------@@@@")
print("######------HiddenLayer = {}-----######\n".format(nodesInHiddenLayers))
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "Sigmoid", 0.5)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.5, max_epoch=10000, epsilon=[0.02, 1e-06], adaptive=True)

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
train_accuracy_adaptive_lr[1] = AccuTrain
test_accuracy_adaptive_lr[1] = AccuTest
time_adaptive_lr[1] = train_time
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[17]:


'''Global Params'''
nodesInHiddenLayers = [1]
print("@@@@@@@@@@---------Q1_c---------@@@@@@@@@@")
print("@@@@------Adaptive Learning Rate------@@@@")
print("######------HiddenLayer = {}-----######\n".format(nodesInHiddenLayers))
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "Sigmoid", 0.05)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.5, max_epoch=10000, epsilon=[0.02, 1e-06], adaptive=True)

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
train_accuracy_adaptive_lr[0] = AccuTrain
test_accuracy_adaptive_lr[0] = AccuTest
time_adaptive_lr[0] = train_time
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[8]:


'''Global Params'''
print("@@@@@@@@@@---------Q1_d---------@@@@@@@@@@")
print("@@@@@@@@@@---------ReLU---------@@@@@@@@@@")
nodesInHiddenLayers = [100,100]
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "ReLU", 0.000005)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.1, max_epoch=1000, epsilon=[0.01, 1e-06], adaptive=False)

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[19]:


'''Global Params'''
print("@@@@@@@@@@---------Q1_d---------@@@@@@@@@@")
print("@@@@@@@@@@---------SIGMOID---------@@@@@@@@@@")
nodesInHiddenLayers = [100,100]
neu_net = NeuralNetwork(nodesInHiddenLayers, "Sigmoid", "Sigmoid", 0.05)
print(neu_net)
train_time = neu_net.fit(DTrain, oneHotTrainLables, eta=0.1, max_epoch=1000, epsilon=[0.02, 1e-06], adaptive=False)

AccuTrain = neu_net.predict(DTrain, LabelsTrain)
AccuTest = neu_net.predict(DTest, LabelsTest)
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[11]:


print("\nTrain Accuracy (Fixed Learning Rate):")
print(["{} %".format(round(elem, 3)) for elem in train_accuracy_list])
print("\nTrain Accuracy (Adaptive Learning Rate):")
print(["{} %".format(round(elem, 3)) for elem in train_accuracy_adaptive_lr])
print("\nTest Accuracy (Fixed Learning Rate):")
print(["{} %".format(round(elem, 3)) for elem in test_accuracy_list])
print("\nTest Accuracy (Adaptive Learning Rate):")
print(["{} %".format(round(elem, 3)) for elem in test_accuracy_adaptive_lr])
print("\nTrain Time (Fixed Learning Rate):")
print(["{} Mins".format(round(elem, 3)) for elem in time_fixed_lr])
print("\nTrain Time (Adaptive Learning Rate):")
print(["{} Mins".format(round(elem, 3)) for elem in time_adaptive_lr])


# In[12]:


plotAccuracies(lr_type = "fixed")
plotAccuracies(lr_type = "adaptive")


# In[13]:


plotAccuraciesComparision(data_type = "Train")
plotAccuraciesComparision(data_type = "Test")


# In[14]:


plotTrainingTime()


# In[100]:


'''
def prediction(net, data, lbl):
    net.forwardFeed(data)
    outputs = net.layers[-1].output
    prediction = []
    for x in outputs:
        prediction.append(np.argmax(x))
    print(len(prediction), lbl.shape)
    accu_score = accuracy_score(lbl, prediction)
    print("Accuracy= ",accu_score)
prediction(neu_net, DTest, LabelsTest)'''

