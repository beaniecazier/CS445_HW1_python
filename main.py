# Design Unit/Module: Multiclass Perceptron
# File Name: main.py
# Description:
# Limitations: None?
# System:
# Author: Preston Cazier
# Course: CS 445 Machine Learning (Winter 2019)
# Assignment: Homework 1
# Revision: 1.0 03/1/2019

# Using the MNIST data sets to train and test perceptrons.
# Assumes first column is target info and other columns are input data
#
# This assignment requires the following:
# - accuracy based off training and test data sets
# - accuracy before training and after each epoch
# - perceptron with greatest output is treated to be the prediction for network
# - confusion matrix for test data set after training is completed
# - perform the above with three different learning rates: 0.01, 0.1, 1

import perceptron
import pandas
import numpy

def sumup(matrix):
    accuracy = 0.0
    return accuracy

# initialize hyperparameter variables
EPOCH_MAX = 2
INPUT_MAX = 255
LRATE = 1
NUMCLASSES = 10

# set up accuracy recording
accuracy = pandas.DataFrame(0, index=range(0, EPOCH_MAX+1), columns=['test', 'train'])

# set up confusion matrix: rows=actual, col=predicted
confmat = pandas.DataFrame(0, index=range(0, 10), columns=range(0, 10))

# load data
TRAIN_FILE = "mnist_train.csv"
TEST_FILE = "mnist_test.csv"
train_data = pandas.read_csv(TRAIN_FILE, header=None)
test_data = pandas.read_csv(TEST_FILE, header=None)

# Save targets as a separate dataframe/array
target_train = train_data[0].values
train_data[:, 0] = INPUT_MAX
target_test = train_data[0].values
test_data[:, 0] = INPUT_MAX

input_size = len(train_data[0])  # how many inputs are there

# initialize perceptron network
network = perceptron.Perceptron(lrate=LRATE, num_inputs=input_size,
                     num_outputs=NUMCLASSES)

# randomize training set
train_data.sample(frac=1)      # shuffle training data

# Preprocess data
train_data = numpy.divide(train_data, INPUT_MAX)
test_data = numpy.divide(train_data, INPUT_MAX)

# find initial accuracy
confmat['test'] = network.accuracy(network.predict(test_data))
accuracy['test'][0] = sumup(confmat['test'])
confmat['train'] = network.accuracy(network.predict(train_data))
accuracy['train'][0] = sumup(['train'])

# do epochs
# 1.train on each data point one at a time
# 2.find accuracy for epoch
# make confusion matrix
for i in range(1, EPOCH_MAX+1):
    for j in range(0, len(train_data)):
        network.train(target=target_train[i], inputs=train_data[i])
    confmat['test'] = network.accuracy(network.predict(test_data))
    accuracy['test'] = sumup(confmat['test'])
    confmat['train'] = network.accuracy(network.predict(train_data))
    accuracy['train'][i] = sumup(confmat['train'])

# Output accuracy and confusion matrix to CSV file
accuracy_filepath = 'accuracy_rate='+str(LRATE)+'.csv'
confmat_filepath = 'confusionmatrix_rate='+str(LRATE)+'.csv'
accuracy.to_csv(accuracy_filepath)
confmat.to_csv(confmat_filepath)
