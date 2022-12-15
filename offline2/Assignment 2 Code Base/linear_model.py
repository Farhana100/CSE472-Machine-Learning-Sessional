import numpy as np


def sigmoid(z):
    """
    returns sigmoid of z
    :param z:
    :return: sigmoid(z)
    """
    z = np.array(z)
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, params):
        """
        :param params: dictionary with
            d:  number of features to learn
            alpha: learning parameter
            number_of_iterations: number of iterations in gradient ascend
        """

        self.d = params['d']
        self.theta = np.zeros(self.d + 1)
        self.alpha = params['alpha']
        self.number_of_iterations = params['number_of_iterations']

    def h(self, X):
        """
        theta.T x X
        :param X: input (n, d)
        :return: theta.T x X
        """
        X = np.array(X)

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        assert self.theta.shape[0] == X.shape[1] + 1

        X = np.append(np.ones((X.shape[0], 1)), X, axis=1)  # add bias i.e., theta 0
        h_ = sigmoid(np.matmul(X, self.theta))

        return h_

    def likelihoodFunction(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Initialization
        n = y.size  # number of training examples

        # likelihood
        L = np.sum(y * np.log(self.h(X)) + (1 - y) * np.log(1 - self.h(X))) / n
        grad = np.matmul(y - self.h(X), np.append(np.ones((n, 1)), X, axis=1)) / n

        return L, grad

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        # initialize theta
        self.theta = np.full(self.d + 1, 0.5)

        # run gradient ascend
        for i in range(self.number_of_iterations):
            L, grad = self.likelihoodFunction(X, y)
            self.theta = self.theta + self.alpha * grad

            # print("iteration:", i, L)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X: X matrix of shape (n, d)
        :return: y_pred vector of shape (n,)
        """

        y_pred = self.h(X)
        y_pred = np.array([0 if i < 0.5 else 1 for i in y_pred])

        return y_pred
