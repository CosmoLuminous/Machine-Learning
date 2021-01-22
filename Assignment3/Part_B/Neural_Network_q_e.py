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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize


# In[2]:


'''Load Data And Identify Classes'''
PathTrain =  os.getcwd() + "/data/train.csv"
PathTest =  os.getcwd() + "/data/test.csv"
DFTrain = pd.read_csv(PathTrain, header=None)
DFTest = pd.read_csv(PathTest, header=None)
totalClasses = len(Counter(DFTrain.iloc[:,-1]))


# In[3]:


DTrain = DFTrain.iloc[:,:-1]
#DTrain = np.c_[np.ones(len(DTrain)), DTrain]
LabelsTrain = np.array(DFTrain.iloc[:,-1])
DTest = DFTest.iloc[:,:-1]
#DTest = np.c_[np.ones(len(DTest)), DTest]
LabelsTest = np.array(DFTest.iloc[:,-1])


# In[4]:


DTrain = normalize(DTrain)
DTest = normalize(DTest)


# In[5]:


'''One Hot Encoding for Train and Test Labels'''
oneHotTrainLables = np.zeros((len(DFTrain), totalClasses))

for i in range(len(DFTrain)):
    oneHotTrainLables[i][DFTrain.iloc[i, -1]] = 1


# In[12]:


print("@@@@@@@@@@---------Q1_e---------@@@@@@@@@@")
print("@@@@@@@@@@-----SKLEARN_ReLU-----@@@@@@@@@@")
'''For invscaling it is not working. therefore kept lr to 0.1 constant and 0.5 with adaptive'''
t0 = time()
nn = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', 
                     batch_size=100, learning_rate_init=0.1, learning_rate='constant', max_iter=400, verbose=True)

print(nn)
print("$$ Training Model...!!!")
nn.fit(DTrain, LabelsTrain)
t1 = time()
print("$$ Training Complete...!!!")
print("$$ Training Time = {}Min".format(round((t1-t0)/60, 2)))


# In[13]:


pred = nn.predict(DTest)
AccuTest = 100* accuracy_score(LabelsTest, pred)

pred = nn.predict(DTrain)
AccuTrain = 100* accuracy_score(LabelsTrain, pred)
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[19]:


print("@@@@@@@@@@---------Q1_e---------@@@@@@@@@@")
print("@@@@@@@@@@-----SKLEARN_ReLU-----@@@@@@@@@@")
t0 = time()
nn = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', 
                     batch_size=100, learning_rate_init=0.5, learning_rate='adaptive', max_iter=400, verbose=True)
print(nn)
print("$$ Training Model...!!!")
nn.fit(DTrain, LabelsTrain)
t1 = time()
print("$$ Training Complete...!!!")
print("$$ Training Time = {}Min".format(round((t1-t0)/60, 2)))


# In[20]:


pred = nn.predict(DTest)
AccuTest = 100* accuracy_score(LabelsTest, pred)

pred = nn.predict(DTrain)
AccuTrain = 100* accuracy_score(LabelsTrain, pred)
print("Train Accuracy = {}%".format(round(AccuTrain,3)))
print("Test Accuracy = {}%".format(round(AccuTest,3)))


# In[ ]:




