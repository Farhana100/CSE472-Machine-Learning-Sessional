"""
main code that you will run
"""

DATASET_FILENAME = 'data_banknote_authentication.csv'

from linear_model import LogisticRegression
from data_handler import load_dataset, split_dataset
from metrics import precision_score, recall_score, f1_score, accuracy

if __name__ == '__main__':
    # data load
    X, y = load_dataset(DATASET_FILENAME)
    # print(X, y)

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y, 0.2, True)


    # training
    # params = dict()
    params = {
        'd': X.shape[1],
        'alpha': 0.01,
        'number_of_iterations': 1000,
    }
    classifier = LogisticRegression(params)

    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)
    y_pred_train = classifier.predict(X_train)

    # print(y_pred)
    # print(y_pred_train)

    # performance on test set
    print('Test Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Train Accuracy ', accuracy(y_true=y_train, y_pred=y_pred_train))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
