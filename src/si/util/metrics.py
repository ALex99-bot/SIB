
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