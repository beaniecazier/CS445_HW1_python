# Perceptron will be initialized with a specified number of inputs.
# Bias should not be included in input count argument. Bias will always be provided.

# Design Unit/Module: Multiclass Perceptron
# File Name: perceptron.py
# Description: 
# Limitations: None?
# System:
# Author: Preston Cazier
# Course: CS 445 Machine Learning (Winter 2019)
# Assignment: Homework 1
# Revision: 1.0 03/1/2019

import numpy as np

WEIGHT_MIN = -0.05
WEIGHT_MAX = 0.05

class Perceptron:
    # Initialize with 
    def __init__(self, num_inputs, num_outputs, lrate):
        self.inputlen = num_inputs
        self.weights = np.random.uniform(low=WEIGHT_MIN,high=WEIGHT_MAX,size=(num_inputs,num_outputs))
        print(self.weights)
        self.numclass = num_outputs
        self.learnrate = lrate
        self.confmat = np.zeros((self.numclass, self.numclass))

    # Evaluates inputs by summing the products of inputs and weights
    # Return -1 if size of inputs doesn't match initialized input size for Perceptron
    # Returns 1 if evaluates greater than 0
    # Returns 0 otherwise
    def predict(self,inputs):
        if len(inputs) != self.inputlen:
            return -1
        return np.inner(inputs, self.weights)
    
    # Update weights for inputs if output didn't match target
    # Change of weight += (learn_rate)*(target-output)*(input)
    def train(self,target,inputs):
        print('Now training on ')
        predicts = self.predict(inputs)
        targets = np.zeros(self.numclass)
        targets[target] = 1

        delta = np.multiply(self.learnrate, np.subtract(targets, predicts))
        delta_weights = np.outer(delta,inputs)
        self.weights = np.add(self.weights,delta_weights)

    def accuracy(self, data):
        self.confusionmatrix(self.predict(data))
        # sum diagonal to get accuracy
        total = 0
        for i in range(self.numclass):
            total += self.confmat[i][i]
        return total / len(data)

    def confusionmatrix(self, predictions):
        # make confmat
        self.confmat = np.zeros((self.numclass, self.numclass))
        # for each row in predictions
        return self.confmat
