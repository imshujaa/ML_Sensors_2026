"""Training and evaluation metrics."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics."""
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred_prob: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics.
        
        Args:
            y_true: True labels.
            y_pred_prob: Predicted probabilities.
            threshold: Classification threshold.
            
        Returns:
            Dictionary of metrics.
        """
        y_pred = (y_pred_prob > threshold).astype(int).flatten()
        y_true = y_true.flatten()
        y_pred_prob = y_pred_prob.flatten()
        
        # Basic metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROC AUC
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            roc_auc = roc_auc_score(y_true, y_pred_prob)
            pr_auc = average_precision_score(y_true, y_pred_prob)
        except:
            roc_auc = 0.0
            pr_auc = 0.0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1_score),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }
