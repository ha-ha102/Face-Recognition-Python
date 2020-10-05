# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:44:56 2018

@author: Home
"""
import numpy as np

alpha = 0.01
hiddenSize = 32
 
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)

x = new[:,0:4356]
y = new[:,4356]

np.random.seed(1)

synapse_0 = 2*np.random.random((4356,hiddenSize)) - 1
synapse_1 = 2*np.random.random((hiddenSize,1)) - 1

for j in range(60000):
    layer_0 = x
    layer_1 = sigmoid(np.dot(layer_0,synapse_0))
    layer_2 = sigmoid(np.dot(layer_1,synapse_1))

    layer_2_error = layer_2 - y

    if (j% 10000) == 0:
        print( "Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))))

    layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)

    layer_1_error = layer_2_delta.dot(synapse_1.T)

    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

    synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
    synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))

