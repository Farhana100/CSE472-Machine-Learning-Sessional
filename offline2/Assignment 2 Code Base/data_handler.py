import numpy as np


def load_dataset(filename):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :param filename: dataset filename   n x m (including header)
    :return: X, y
    """
    # todo: implement
    X = []
    y = []
    with open('data_banknote_authentication.csv', 'r') as f:
        for line in f:
            X.append(line.strip().split(','))

    X = np.array(X)
    n, m = X.shape

    y = np.array([int(i) for i in X[1:, -1]])
    X = np.array([[float(j) for j in i] for i in X[1:, :m - 1]])

    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """

    n, m = X.shape

    if shuffle:
        X = np.append(X, np.reshape(y, (len(y), 1)), axis=1)
        np.random.shuffle(X)

        y = np.array([int(i) for i in X[:, -1]])
        X = X[:, :m]

    test_size = 0 if test_size is None else test_size
    number_of_sample_points_in_test = int(n * test_size)

    X_train = X[:n - number_of_sample_points_in_test]
    y_train = y[:n - number_of_sample_points_in_test]
    X_test = X[n - number_of_sample_points_in_test:]
    y_test = y[n - number_of_sample_points_in_test:]

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X: input matrix (n, d)
    :param y: output vector (n,)
    :return: sampled new training set
    """
    sample_ind = np.random.choice(range(len(y)), len(y), replace=True)

    X_sample = np.array(X)[sample_ind]
    y_sample = np.array(y)[sample_ind]

    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
