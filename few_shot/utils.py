'''
Helpful utility functions
'''
from keras import backend as keras_backend


def recall(y_true, y_pred):
    '''
    Recall metric.
    '''
    true_positives = keras_backend.sum(
        keras_backend.round(keras_backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras_backend.sum(
        keras_backend.round(keras_backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + keras_backend.epsilon())
    return recall


def precision(y_true, y_pred):
    '''
    Precision metric
    '''
    true_positives = keras_backend.sum(
        keras_backend.round(keras_backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = keras_backend.sum(
        keras_backend.round(keras_backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives +
                                  keras_backend.epsilon())
    return precision


def f1(y_true, y_pred):
    '''
    F1 score
    '''
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((pre * rec) /
                (pre + rec + keras_backend.epsilon()))
