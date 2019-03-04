import pandas 
import numpy as np

# initialize hyperparameter variables
EPOCH_MAX = 2
INPUT_MAX = 255
LRATE = 1
NUMCLASSES = 10
WEIGHT_MIN = -0.05
WEIGHT_MAX = 0.05
TEST_FILE = "mnist_test.csv"
test_data = pandas.read_csv(TEST_FILE, header=None)

# randomize training set
test_data = test_data.sample(frac=1).values

# set up accuracy recording
accuracy = pandas.DataFrame(0, index=range(0, EPOCH_MAX+1), columns=['test', 'train'])

# Save targets as a separate dataframe/array
target_test = test_data[:,0]
test_data[:, 0] = INPUT_MAX
test_data = np.divide(test_data, INPUT_MAX)

weights = np.random.uniform(low=WEIGHT_MIN, high=WEIGHT_MAX, size=(10, 785))

#inner = np.inner(test_data, weights)
inner = np.inner(weights, test_data)
print(weights.shape)
print(test_data.shape)
print(inner.shape)

# set up confusion matrix: rows=actual, col=predicted
confmat_train = pandas.DataFrame(0, index=range(0, 10), columns=range(0, 10))
confmat_test = pandas.DataFrame(0, index=range(0, 10), columns=range(0, 10))

# Output accuracy and confusion matrix to CSV file
#accuracy['test'].to_csv('accuracy_rate_test'+str(LRATE)+'.csv')
#ccuracy['train'].to_csv('accuracy_rate_train'+str(LRATE)+'.csv')
#confmat_train .to_csv('confusionmatrix_train'+str(LRATE)+'.csv')
#confmat_test .to_csv('confusionmatrix_test'+str(LRATE)+'.csv')
