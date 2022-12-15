"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""

import numpy as np


def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return: accuracy in percentage
    """
    assert len(y_true) == len(y_pred)
    acc = np.array([1 if y_pred[i] == y_true[i] else 0 for i in range(len(y_true))]).sum() * 100.0 / len(y_true)

    return acc


def precision_score(y_true, y_pred):
    """
    calculates precision score true_pos / (true_pos + false_pos)
    :param y_true:
    :param y_pred:
    :return: precision score
    """
    true_pos = np.array([1 if y_true[i] == 1 and y_pred[i] == 1 else 0 for i in range(len(y_true))]).sum()
    false_pos = np.array([1 if y_true[i] == 0 and y_pred[i] == 1 else 0 for i in range(len(y_true))]).sum()

    # print('true_pos', true_pos)
    # print('false_pos', false_pos)

    pres = true_pos/(true_pos + false_pos)
    return pres


def recall_score(y_true, y_pred):
    """
    calculates recall score true_pos / (true_pos + false_neg)
    :param y_true:
    :param y_pred:
    :return: recall score
    """
    true_pos = np.array([1 if y_true[i] == 1 and y_pred[i] == 1 else 0 for i in range(len(y_true))]).sum()
    false_neg = np.array([1 if y_true[i] == 1 and y_pred[i] == 0 else 0 for i in range(len(y_true))]).sum()

    # print('false_neg', false_neg)

    recall = true_pos/(true_pos + false_neg)
    return recall


def f1_score(y_true, y_pred):
    """
    calculates f1 score 2 x precision x recall / (precision + recall)
    :param y_true:
    :param y_pred:
    :return: f1 score
    """

    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)

    f1 = 2 * precision * recall / (precision + recall)
    return f1
