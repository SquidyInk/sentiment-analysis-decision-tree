"""
Model Evaluation Module for Sentiment Analysis
CSC 108 - Project 2

This module provides comprehensive evaluation capabilities for sentiment analysis models,
including metrics calculation, confusion matrix generation, and classification reports.

Author: SquidyInk
Date: 2026-01-06
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class ModelEvaluator:
    """
    A class for evaluating machine learning models for sentiment analysis.
    
    This class provides methods to calculate various performance metrics,
    generate confusion matrices, and produce detailed classification reports.
    """
    
    def __init__(self, y_true: List[Any], y_pred: List[Any], labels: Optional[List[str]] = None):
        """
        Initialize the ModelEvaluator with true and predicted labels.
        
        Args:
            y_true: List of true labels
            y_pred: List of predicted labels
            labels: Optional list of label names for reporting
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.labels = labels if labels else sorted(list(set(y_true)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate various evaluation metrics for the model.
        
        Returns:
            Dictionary containing accuracy, precision, recall, and F1-score
        """
        # Calculate accuracy
        accuracy = np.mean(self.y_true == self.y_pred)
        
        # Calculate per-class metrics
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for label in self.labels:
            # True Positives, False Positives, False Negatives
            tp = np.sum((self.y_true == label) & (self.y_pred == label))
            fp = np.sum((self.y_true != label) & (self.y_pred == label))
            fn = np.sum((self.y_true == label) & (self.y_pred != label))
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precision_scores.append(precision)
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_scores.append(recall)
            
            # F1-score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        
        # Calculate macro averages
        metrics = {
            'accuracy': accuracy,
            'precision_macro': np.mean(precision_scores),
            'recall_macro': np.mean(recall_scores),
            'f1_macro': np.mean(f1_scores),
            'precision_weighted': self._weighted_average(precision_scores),
            'recall_weighted': self._weighted_average(recall_scores),
            'f1_weighted': self._weighted_average(f1_scores)
        }
        
        return metrics
    
    def _weighted_average(self, scores: List[float]) -> float:
        """
        Calculate weighted average based on support for each class.
        
        Args:
            scores: List of scores for each class
            
        Returns:
            Weighted average score
        """
        weights = [np.sum(self.y_true == label) for label in self.labels]
        total = sum(weights)
        
        if total == 0:
            return 0.0
        
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        return weighted_sum / total
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Generate a confusion matrix for the predictions.
        
        Returns:
            Confusion matrix as a numpy array where rows represent true labels
            and columns represent predicted labels
        """
        n_labels = len(self.labels)
        confusion_matrix = np.zeros((n_labels, n_labels), dtype=int)
        
        for true_label, pred_label in zip(self.y_true, self.y_pred):
            true_idx = self.label_to_idx[true_label]
            pred_idx = self.label_to_idx[pred_label]
            confusion_matrix[true_idx][pred_idx] += 1
        
        return confusion_matrix
    
    def get_classification_report(self) -> Dict[str, Dict[str, float]]:
        """
        Generate a detailed classification report with per-class metrics.
        
        Returns:
            Dictionary containing precision, recall, F1-score, and support
            for each class, plus overall metrics
        """
        report = {}
        
        for label in self.labels:
            tp = np.sum((self.y_true == label) & (self.y_pred == label))
            fp = np.sum((self.y_true != label) & (self.y_pred == label))
            fn = np.sum((self.y_true == label) & (self.y_pred != label))
            support = np.sum(self.y_true == label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            report[str(label)] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': int(support)
            }
        
        # Add overall metrics
        metrics = self.calculate_metrics()
        report['accuracy'] = metrics['accuracy']
        report['macro avg'] = {
            'precision': metrics['precision_macro'],
            'recall': metrics['recall_macro'],
            'f1-score': metrics['f1_macro'],
            'support': len(self.y_true)
        }
        report['weighted avg'] = {
            'precision': metrics['precision_weighted'],
            'recall': metrics['recall_weighted'],
            'f1-score': metrics['f1_weighted'],
            'support': len(self.y_true)
        }
        
        return report
    
    def print_evaluation_summary(self) -> None:
        """
        Print a formatted summary of the model evaluation results.
        
        Displays confusion matrix, classification report, and overall metrics
        in a human-readable format.
        """
        print("=" * 70)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 70)
        print()
        
        # Print overall metrics
        metrics = self.calculate_metrics()
        print("OVERALL METRICS:")
        print("-" * 70)
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro):   {metrics['f1_macro']:.4f}")
        print()
        
        # Print confusion matrix
        print("CONFUSION MATRIX:")
        print("-" * 70)
        cm = self.get_confusion_matrix()
        
        # Header
        header = "True/Pred |"
        for label in self.labels:
            header += f" {str(label):>8} |"
        print(header)
        print("-" * len(header))
        
        # Rows
        for i, label in enumerate(self.labels):
            row = f"{str(label):>9} |"
            for j in range(len(self.labels)):
                row += f" {cm[i][j]:>8} |"
            print(row)
        print()
        
        # Print classification report
        print("CLASSIFICATION REPORT:")
        print("-" * 70)
        report = self.get_classification_report()
        
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 70)
        
        for label in self.labels:
            label_str = str(label)
            metrics = report[label_str]
            print(f"{label_str:<15} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1-score']:>10.4f} {metrics['support']:>10}")
        
        print("-" * 70)
        print(f"{'accuracy':<15} {'':>10} {'':>10} {report['accuracy']:>10.4f} {len(self.y_true):>10}")
        print(f"{'macro avg':<15} {report['macro avg']['precision']:>10.4f} "
              f"{report['macro avg']['recall']:>10.4f} {report['macro avg']['f1-score']:>10.4f} "
              f"{report['macro avg']['support']:>10}")
        print(f"{'weighted avg':<15} {report['weighted avg']['precision']:>10.4f} "
              f"{report['weighted avg']['recall']:>10.4f} {report['weighted avg']['f1-score']:>10.4f} "
              f"{report['weighted avg']['support']:>10}")
        print("=" * 70)


def evaluate_model(y_true: List[Any], y_pred: List[Any], 
                  labels: Optional[List[str]] = None,
                  verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model and optionally print results.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        labels: Optional list of label names
        verbose: If True, print evaluation summary
        
    Returns:
        Dictionary containing metrics, confusion matrix, and classification report
    """
    evaluator = ModelEvaluator(y_true, y_pred, labels)
    
    if verbose:
        evaluator.print_evaluation_summary()
    
    results = {
        'metrics': evaluator.calculate_metrics(),
        'confusion_matrix': evaluator.get_confusion_matrix(),
        'classification_report': evaluator.get_classification_report()
    }
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Model Evaluation Module")
    print()
    
    # Sample data for sentiment analysis (positive, negative, neutral)
    y_true = ['positive', 'negative', 'neutral', 'positive', 'negative', 
              'positive', 'neutral', 'negative', 'positive', 'negative']
    y_pred = ['positive', 'negative', 'neutral', 'negative', 'negative',
              'positive', 'positive', 'negative', 'positive', 'neutral']
    
    # Evaluate the model
    results = evaluate_model(y_true, y_pred, verbose=True)
    
    print("\nReturned results dictionary keys:", list(results.keys()))
