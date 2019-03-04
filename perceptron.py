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
    def __init__(self, num_inputs, num_outputs, lrate, verbose):
        self.verbose = verbose
        if self.verbose:
            print('\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print('now initializing the network')
        self.inputlen = num_inputs
        self.weights = np.random.uniform(low=WEIGHT_MIN, high=WEIGHT_MAX, size=(num_outputs, num_inputs))
        self.numclass = num_outputs
        self.learnrate = lrate
        self.confmat = np.zeros((self.numclass, self.numclass))
        if self.verbose:
            print('the network was initialized with ', self.inputlen, 'number of inputs')
            print(self.numclass, 'number of outputs and a learning rate of',self.learnrate)
            print('the initial weights are:')
            print(self.weights)
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')

    # Evaluates inputs by summing the products of inputs and weights
    # Return -1 if size of inputs doesn't match initialized input size for Perceptron
    # Returns 1 if evaluates greater than 0
    # Returns 0 otherwise
    def predict(self, inputs):
        if self.verbose:
            print('\n******************************')
        inputs = np.array(inputs)
        if self.verbose:
            print('the length of inputs is ', inputs.shape)
        #if len(inputs[0]) != self.inputlen:
        #    return -1
        predictions = np.inner(inputs, self.weights)
        if self.verbose:
            print('by using the following input,')
            print(inputs)
            print('the network predicted the following:')
            print(predictions)
            print('******************************\n')
        return predictions
    
    # Update weights for inputs if output didn't match target
    # Change of weight += (learn_rate)*(target-output)*(input)
    def train(self, target, inputs):
        if self.verbose:
            print('\n+++++++++++++++++++++++++++++++')
            print('Now training on ')
            print(inputs)
        predicts = np.sign(self.predict(inputs))
        if (self.verbose):
            print(predicts)
            print(predicts.shape)
        targets = np.zeros(self.numclass)
        targets[target] = 1
        if self.verbose:
            print('compared to the target array:')
            print(targets)
            print(targets.shape)
            print('leads the network weights of:')
            print(self.weights)
            print(np.subtract(targets, predicts))
        delta = np.multiply(self.learnrate, np.subtract(targets, predicts))
        delta_weights = np.outer(delta,inputs)
        self.weights = np.add(self.weights, delta_weights)
        if self.verbose:
            print('to be updated to:')
            print(self.weights)
            print('+++++++++++++++++++++++++++++++\n')

    def accuracy(self, data, targets):
        if self.verbose:
            print('\n///////////////////////////////')
            print('Now calculating the accuracy of the network on the following data:')
            print(data)
            print('compared to this target array')
            print(targets)
        predictions = self.predict(data)
        self.confusionmatrix(predictions, targets)
        # sum diagonal to get accuracy
        total = 0
        for i in range(self.numclass):
            total += self.confmat[i][i]
        acc = total / len(data)
        if self.verbose:
            print('this gives tp count of ', total, 'out of ',
                len(data), 'for an accuracy of ', acc)
            print('///////////////////////////////\n')
        return acc

    def confusionmatrix(self, predictions, targets):
        if self.verbose:
            print('\n-----------------------------')
        # make confmat
        self.confmat = np.zeros((self.numclass, self.numclass))
        # for each row in predictions
        for index in range(len(targets)):
            predicted_index = predictions[index].argmax(axis=0)
            #predicted_index = 0
            target_index = targets[index]
            if self.verbose:
                print('incrementing target:', target_index, ' and predicted:', predictions[index][predicted_index], 'at index ', predicted_index)
            self.confmat[target_index][predicted_index] += 1
        if self.verbose:
            print('the generated confusion matrix is:')
            print(self.confmat)
            print('-----------------------------\n')
        return self.confmat
