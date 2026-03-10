import sys
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException


def get_classification_score(y_true, y_pred, y_prob=None) -> ClassificationMetricArtifact:
    """
    Computes classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities for class 1 (needed for roc_auc)

    Returns:
        ClassificationMetricArtifact with f1, precision, recall, roc_auc
    """
    try:
        model_f1_score       = f1_score(y_true, y_pred, zero_division=0)
        model_recall_score   = recall_score(y_true, y_pred, zero_division=0)
        model_precision_score = precision_score(y_true, y_pred, zero_division=0)

        # FIX: added roc_auc — requires probability scores, falls back to 0.0
        if y_prob is not None:
            model_roc_auc = roc_auc_score(y_true, y_prob)
        else:
            model_roc_auc = 0.0

        return ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            roc_auc_score=model_roc_auc,
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)
