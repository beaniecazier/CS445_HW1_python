# Design Unit/Module: Multiclass Perceptron
# File Name: main.py
# Description: Use the MNIST data set to train and test a network of 10 perceptrons.
# Assumptions: First entry in each row contains target class
# Limitations: None?
# System: any(Python)
# Author: Preston Cazier
# Course: CS 445 Machine Learning (Winter 2019)
# Assignment: Homework 1
# Revision: 1.2 03/7/2019

import perceptron
import pandas as pd
import numpy as np

# initialize hyperparameter variables
EPOCHS = 50
INPUT_MAX = 255
LRATE = 0.01
NUMCLASSES = 10
VERBOSE = False
TRAIN_FILE = "mnist_train.csv"
TEST_FILE = "mnist_test.csv"

# set up accuracy recording
accuracy = pd.DataFrame(0.0, index=range(0, EPOCHS+1), columns=['test', 'train'])

# set up confusion matrix: rows=actual, col=predicted
confmat_train = pd.DataFrame(0, index=range(0, 10), columns=range(0, 10))
confmat_test = pd.DataFrame(0, index=range(0, 10), columns=range(0, 10))

# load data
train_data = pd.read_csv(TRAIN_FILE, header=None)
test_data = pd.read_csv(TEST_FILE, header=None).values

# randomize training set
train_data = train_data.sample(frac=1).values
if VERBOSE:
    print('now randomizing the training data and separating out the targets')

# Save targets as a separate dataframe/array
train_targets = train_data[:, 0]
train_data = np.array(train_data)
train_data[:, 0] = INPUT_MAX
if VERBOSE:
    print('training data targets:')
    print(train_targets)
    print('training data:')
    print(train_data)
    print('the shape of this set of data is:')
    print(train_data.shape)
    print('now separating out the targets from the testing data')
test_targets = test_data[:, 0]
test_data = np.array(test_data)
test_data[:, 0] = INPUT_MAX
if VERBOSE:
    print('testing data targets:')
    print(test_targets)
    print('testing data:')
    print(test_data)
    print('the shape of this set of data is:')
    print(test_data.shape)

input_size = len(train_data[0])  # how many inputs are there
if VERBOSE:
    print('the number of inputs is:', input_size)

# Preprocess data
train_data = np.divide(train_data, INPUT_MAX)
if VERBOSE:
    print('training data after preprocessing')
    print(train_data)
test_data = np.divide(test_data, INPUT_MAX)
if VERBOSE:
    print('testing data after preprocessing')
    print(test_data)

# initialize perceptron network
network = perceptron.Perceptron(lrate=LRATE, num_inputs=input_size, num_outputs=NUMCLASSES, verbose=VERBOSE)

# find initial accuracy
if VERBOSE:
    print('finding the initial accuracy of the testing and training data')
accuracy['test'][0] = network.accuracy(test_data, test_targets)
accuracy['train'][0] = network.accuracy(train_data, train_targets)

# do epochs
# 1.train on each data point one at a time
# 2.find accuracy for epoch
# make confusion matrix
for i in range(1, EPOCHS+1):
    if not VERBOSE:
        print('starting epoch ', i)
    for j in range(0, len(train_data)):
        network.train(target=train_targets[j], inputs=train_data[j])
    print('finding accuracy')
    accuracy['test'][i] = network.accuracy(test_data, test_targets)
    accuracy['train'][i] = network.accuracy(train_data, train_targets)

confmat_test = network.confusionmatrix(network.predict(test_data), test_targets)
confmat_train = network.confusionmatrix(network.predict(train_data), train_targets)

# Output accuracy and confusion matrix to CSV file
accuracy['test'].to_csv('accuracy_rate_test'+str(LRATE)+'.csv')
accuracy['train'].to_csv('accuracy_rate_train'+str(LRATE)+'.csv')
pd.DataFrame(confmat_train).to_csv('confusionmatrix_train'+str(LRATE)+'.csv')
pd.DataFrame(confmat_test).to_csv('confusionmatrix_test'+str(LRATE)+'.csv')
