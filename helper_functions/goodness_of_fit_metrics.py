from scipy.integrate import trapz
import numpy as np

def accuracy(y_true, y_pred):
    """ 
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Args:
        y_true (np.array): is a binary array of true labels (0 or 1).
        y_pred (np.array): is a binary array of predicted labels (0 or 1).
    """
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    """ 
    Precision (Positive Predictive Value)= TP/(TP+FP)
    Args:
        y_true (np.array): is a binary array of true labels (0 or 1).
        y_pred (np.array): is a binary array of predicted labels (0 or 1).
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred):
    """ 
    Recall (sensitivity or True Positive Rate) = TP/(TP+FN)
    Args:
        y_true (np.array): is a binary array of true labels (0 or 1).
        y_pred (np.array): is a binary array of predicted labels (0 or 1).
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def false_positive_rate(y_true, y_pred):
    """
    FPR = FP/(FP+TN)
    proportion of all actual negatives that were classified incorrectly as positives, also known as the probability of false alarm
    Args:
        y_true (np.array): is a binary array of true labels (0 or 1).
        y_pred (np.array): is a binary array of predicted labels (0 or 1).
    """
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return fp/(fp+tn) if fp+tn > 0 else 0

def f1_score(y_true, y_pred):
    """ 
    F1 score = 2 x Precision x Recall/(Precision + Recall)
    Args:
        y_true (np.array): is a binary array of true labels (0 or 1).
        y_pred (np.array): is a binary array of predicted labels (0 or 1).
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0


def auc_roc(y_true, y_probs):
    """ 
    Area under the ROC curve, where it is a graph f TPR vs FPR
    Model with greater area under the curve is generally the better one
    Args:
        y_true (np.array): is a binary array of true labels (0 or 1).
        y_probs (np.array): is an array of predicted scores or probabilities for the positive class (values between 0 and 1).
    """
    # Sort by the y_probs values
    desc_order = np.argsort(-y_probs)
    y_true_sorted = y_true[desc_order]
    y_probs_sorted = y_probs[desc_order]
    
    # Calculate TPR and FPR
    # at higher y_prob, only classes with high probabilites are classified as positive
    # as threshold tends towards lower y_prob, FPR will increase
    tprs = np.cumsum(y_true_sorted) / np.sum(y_true_sorted) # tpr
    fprs = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)
    
    # Append (0,0) to start the curve
    tprs = np.insert(tprs, 0, 0)
    fprs = np.insert(fprs, 0, 0)
    
    return trapz(tprs, fprs)

def log_loss(y_true, y_probs):
    """ 
    binary cross entropy loss
    Args:
        y_true (np.array): is a binary array of true labels (0 or 1).
        y_probs (np.array): typically represents the predicted probability that an instance belongs to the positive class (class 1) in binary classification.
    """
    epsilon = 1e-15  # To avoid log(0)
    y_probs = np.clip(y_probs, epsilon, 1 - epsilon) # y_probs are clipped to [eps, 1-eps]
    return -np.mean(y_true * np.log(y_probs) + (1 - y_true) * np.log(1 - y_probs))
