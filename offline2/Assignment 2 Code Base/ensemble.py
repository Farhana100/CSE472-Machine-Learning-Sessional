from data_handler import bagging_sampler
import numpy as np


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator: logistic regression classifier
        :param n_estimator: number of estimators
        :return: self
        """
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.theta = np.full((n_estimator, base_estimator.d + 1), 0.5)

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        X = np.array(X)
        y = np.array(y)
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        for i in range(self.n_estimator):
            # create sample
            X_sample, y_sample = bagging_sampler(X, y)

            # fit
            self.base_estimator.fit(X_sample, y_sample)
            self.theta[i] = self.base_estimator.theta

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X: (n, d)
        :return: y prediction (n,)
        """

        X = np.array(X)

        y = []

        for i in range(self.n_estimator):
            self.base_estimator.theta = self.theta[i]
            y_pred = self.base_estimator.predict(X)
            y.append(y_pred)

        y = np.array(y)
        y = np.array([1 if i else 0 for i in y.sum(axis=0) > self.n_estimator / 2])

        return y
