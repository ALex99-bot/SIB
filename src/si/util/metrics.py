import numpy as np

def accuracy_score(y_true, y_pred):
    """"
    Classification performance metric that computes the accuracy
    """
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct +=1
    accuracy = correct/len(y_true)
    return accuracy


def mse(y_true, y_pred, squared=True):
    """
    Mean squared error loss function.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = np.average((y_true-y_pred)**2, axis=0)
    if not squared:
        errors = np.sqrt(errors)
    return np.average(errors)