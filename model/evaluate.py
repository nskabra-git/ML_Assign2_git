import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report, matthews_corrcoef,
    roc_curve, precision_recall_curve
)

def compute_metrics(y_true, y_pred, y_proba=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }
    if y_proba is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_proba)
        m["avg_precision"] = average_precision_score(y_true, y_proba)
    return m

def curves(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    return (fpr, tpr), (prec, rec)

def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def report_text(y_true, y_pred, target_names=None):
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
