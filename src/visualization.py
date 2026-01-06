"""
Visualization Module for Sentiment Analysis
CSC 108 Project 2

This module provides visualization tools for sentiment analysis model evaluation
and data exploration using matplotlib and seaborn.

Author: SquidyInk
Date: 2026-01-06
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import plot_tree
from typing import List, Optional, Tuple
import os


class Visualizer:
    """
    Static class for creating various visualizations for sentiment analysis.
    All methods are static and can be called without instantiating the class.
    """
    
    # Set default style for all plots
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             classes: List[str],
                             title: str = 'Confusion Matrix',
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (8, 6),
                             cmap: str = 'Blues') -> None:
        """
        Plot confusion matrix using seaborn heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: List of class names
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size as (width, height)
            cmap: Color map for heatmap
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create heatmap with annotations
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Count'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_decision_tree(model,
                          feature_names: Optional[List[str]] = None,
                          class_names: Optional[List[str]] = None,
                          title: str = 'Decision Tree Visualization',
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (20, 10),
                          max_depth: Optional[int] = 3) -> None:
        """
        Plot decision tree using sklearn's plot_tree.
        
        Args:
            model: Trained decision tree model
            feature_names: List of feature names
            class_names: List of class names
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size as (width, height)
            max_depth: Maximum depth to display (for large trees)
        """
        plt.figure(figsize=figsize)
        
        plot_tree(model, 
                 feature_names=feature_names,
                 class_names=class_names,
                 filled=True,
                 rounded=True,
                 fontsize=10,
                 max_depth=max_depth)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Decision tree saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_names: List[str],
                               importances: np.ndarray,
                               title: str = 'Feature Importance',
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 6),
                               top_n: Optional[int] = None,
                               color: str = 'steelblue') -> None:
        """
        Plot feature importance as a bar chart.
        
        Args:
            feature_names: List of feature names
            importances: Array of feature importance values
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size as (width, height)
            top_n: Show only top N features (optional)
            color: Bar color
        """
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Select top N if specified
        if top_n:
            indices = indices[:top_n]
        
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create bar chart
        bars = plt.barh(range(len(sorted_features)), sorted_importances, color=color)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest importance at top
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{sorted_importances[i]:.4f}',
                    ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: dict,
                               title: str = 'Model Metrics Comparison',
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot comparison of multiple metrics.
        
        Args:
            metrics_dict: Dictionary with metric names as keys and values as values
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size as (width, height)
        """
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create bar chart with colors
        colors = sns.color_palette("husl", len(metrics))
        bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Metrics', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray,
                      y_prob: np.ndarray,
                      title: str = 'ROC Curve',
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (8, 6),
                      pos_label: int = 1) -> None:
        """
        Plot ROC curve for binary classification.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size as (width, height)
            pos_label: Label of positive class
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_class_distribution(labels: np.ndarray,
                               class_names: Optional[List[str]] = None,
                               title: str = 'Class Distribution',
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 6),
                               plot_type: str = 'bar') -> None:
        """
        Plot distribution of classes in dataset.
        
        Args:
            labels: Array of class labels
            class_names: List of class names (optional)
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size as (width, height)
            plot_type: Type of plot ('bar' or 'pie')
        """
        # Count occurrences of each class
        unique, counts = np.unique(labels, return_counts=True)
        
        # Use class names if provided, otherwise use label values
        if class_names is None:
            class_names = [str(label) for label in unique]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        if plot_type == 'bar':
            # Create bar chart
            colors = sns.color_palette("husl", len(unique))
            bars = plt.bar(class_names, counts, color=colors, alpha=0.8, edgecolor='black')
            
            plt.ylabel('Count', fontsize=12)
            plt.xlabel('Class', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count)}\n({count/sum(counts)*100:.1f}%)',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            
        elif plot_type == 'pie':
            # Create pie chart
            colors = sns.color_palette("husl", len(unique))
            plt.pie(counts, labels=class_names, autopct='%1.1f%%',
                   colors=colors, startangle=90, explode=[0.05]*len(unique))
            plt.title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the Visualizer class.
    Uncomment to test individual visualization methods.
    """
    
    # Example 1: Confusion Matrix
    # y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    # y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2])
    # classes = ['Negative', 'Neutral', 'Positive']
    # Visualizer.plot_confusion_matrix(y_true, y_pred, classes, 
    #                                  save_path='outputs/confusion_matrix.png')
    
    # Example 2: Feature Importance
    # features = ['word_count', 'avg_word_length', 'exclamation_count', 
    #            'question_count', 'capital_ratio']
    # importances = np.array([0.35, 0.15, 0.25, 0.10, 0.15])
    # Visualizer.plot_feature_importance(features, importances,
    #                                   save_path='outputs/feature_importance.png')
    
    # Example 3: Metrics Comparison
    # metrics = {
    #     'Accuracy': 0.85,
    #     'Precision': 0.83,
    #     'Recall': 0.87,
    #     'F1-Score': 0.85
    # }
    # Visualizer.plot_metrics_comparison(metrics,
    #                                   save_path='outputs/metrics_comparison.png')
    
    # Example 4: Class Distribution
    # labels = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    # classes = ['Negative', 'Neutral', 'Positive']
    # Visualizer.plot_class_distribution(labels, classes, plot_type='bar',
    #                                   save_path='outputs/class_distribution.png')
    
    # Example 5: ROC Curve
    # y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    # y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.3, 0.7])
    # Visualizer.plot_roc_curve(y_true, y_prob,
    #                          save_path='outputs/roc_curve.png')
    
    print("Visualization module loaded successfully!")
    print("Import this module and use Visualizer class methods for creating plots.")
