"""
main code that you will run
"""

DATASET_FILENAME = 'data_banknote_authentication.csv'

from linear_model import LogisticRegression
from ensemble import BaggingClassifier
from data_handler import load_dataset, split_dataset, bagging_sampler
from metrics import precision_score, recall_score, f1_score, accuracy

if __name__ == '__main__':
    # data load
    X, y = load_dataset(DATASET_FILENAME)

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y, 0.4, True)

    X_sample, y_sample = bagging_sampler(X_train, y_train)

    # training
    # params = dict()
    params = {
        'd': X.shape[1],
        'alpha': 0.01,
        'number_of_iterations': 1000,
    }
    base_estimator = LogisticRegression(params)
    classifier = BaggingClassifier(base_estimator=base_estimator, n_estimator=9)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
