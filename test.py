import pandas

TRAIN_FILE = "mnist_train.csv"
TEST_FILE = "mnist_test.csv"
train_data = pandas.read_csv(TRAIN_FILE, header=None)
test_data = pandas.read_csv(TEST_FILE, header=None)
print(len(test_data))