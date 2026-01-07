"""
Main Execution Script for Sentiment Analysis using Decision Tree
CSC 108 - Project 2

This script demonstrates the complete workflow:
1. Load and preprocess data
2. Train Decision Tree classifier
3. Evaluate model performance
4. Generate visualizations

Author: SquidyInk
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocessing import TextPreprocessor, load_and_split_data, vectorize_text
from model import SentimentDecisionTree
from evaluation import evaluate_model
from visualization import Visualizer


def main():
    """
    Main execution function for the sentiment analysis project.
    """
    print("=" * 70)
    print("SENTIMENT ANALYSIS USING DECISION TREE CLASSIFIER")
    print("CSC 108 - Project 2")
    print("=" * 70)
    print()
    
    # Step 1: Load Dataset
    print("Step 1: Loading Dataset...")
    print("-" * 70)
    
    # TODO: Replace with your actual dataset loading
    # Example: df = pd.read_csv('data/sentiment_dataset.csv')
    
    # For demonstration, using sample data
    sample_data = {
        'text': [
            "I love this product! It's amazing!",
            "This is the worst experience ever.",
            "It's okay, nothing special.",
            "Absolutely fantastic! Highly recommend!",
            "Terrible quality, very disappointed.",
            "Pretty good, meets expectations.",
            "Not bad, could be better.",
            "Excellent service and great quality!",
            "I hate this, waste of money.",
            "Average product, nothing to complain about."
        ],
        'sentiment': [
            'positive', 'negative', 'neutral',
            'positive', 'negative', 'neutral',
            'neutral', 'positive', 'negative', 'neutral'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Step 2: Preprocess Data
    print("Step 2: Preprocessing Data...")
    print("-" * 70)
    
    preprocessor = TextPreprocessor(
        use_stemming=False,
        use_lemmatization=True,
        remove_stopwords=True
    )
    
    df_processed = preprocessor.preprocess_dataset(df, 'text', 'sentiment')
    print(f"Preprocessing complete: {len(df_processed)} samples after cleaning")
    print()
    
    # Step 3: Split Data
    print("Step 3: Splitting Data...")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test = load_and_split_data(
        df_processed,
        'cleaned_text',
        'sentiment',
        # Use 0.3 so the tiny demo set keeps at least one sample per class when stratifying
        test_size=0.3,
        random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()
    
    # Step 4: Vectorize Text
    print("Step 4: Vectorizing Text...")
    print("-" * 70)
    
    X_train_vec, X_test_vec, vectorizer = vectorize_text(
        X_train,
        X_test,
        method='tfidf',
        max_features=1000
    )
    
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print()
    
    # Step 5: Train Model
    print("Step 5: Training Decision Tree Model...")
    print("-" * 70)
    
    model = SentimentDecisionTree(
        criterion='gini',
        max_depth=10,
        min_samples_split=2,
        random_state=42
    )
    
    model.train(X_train_vec, y_train)
    
    print(f"Tree depth: {model.get_tree_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    print()
    
    # Step 6: Make Predictions
    print("Step 6: Making Predictions...")
    print("-" * 70)
    
    y_pred = model.predict(X_test_vec)
    print(f"Predictions made for {len(y_pred)} samples")
    print()
    
    # Step 7: Evaluate Model
    print("Step 7: Evaluating Model...")
    print("-" * 70)
    
    # Get unique labels for evaluation
    labels = sorted(df_processed['sentiment'].unique())
    
    results = evaluate_model(y_test.tolist(), y_pred.tolist(), labels=labels, verbose=True)
    print()
    
    # Step 8: Generate Visualizations
    print("Step 8: Generating Visualizations...")
    print("-" * 70)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Confusion Matrix
    Visualizer.plot_confusion_matrix(
        y_test, y_pred, labels,
        save_path='results/confusion_matrix.png'
    )
    
    # Feature Importance
    feature_names = vectorizer.get_feature_names_out()
    importances = model.get_feature_importance()
    
    # Get top 20 features
    top_indices = np.argsort(importances)[-20:]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    Visualizer.plot_feature_importance(
        top_features, top_importances,
        title='Top 20 Most Important Features',
        save_path='results/feature_importance.png'
    )
    
    # Metrics Comparison
    metrics = results['metrics']
    metrics_to_plot = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision_weighted'],
        'Recall': metrics['recall_weighted'],
        'F1-Score': metrics['f1_weighted']
    }
    
    Visualizer.plot_metrics_comparison(
        metrics_to_plot,
        save_path='results/metrics_comparison.png'
    )
    
    # Class Distribution
    Visualizer.plot_class_distribution(
        y_train.values, labels,
        title='Training Set Class Distribution',
        save_path='results/class_distribution.png'
    )
    
    print("All visualizations saved to 'results/' directory")
    print()
    
    # Step 9: Save Model
    print("Step 9: Saving Model...")
    print("-" * 70)
    
    os.makedirs('models', exist_ok=True)
    model.save_model('models/sentiment_decision_tree.pkl')
    print()
    
    print("=" * 70)
    print("PROCESS COMPLETE!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review the results in the 'results/' directory")
    print("2. Analyze the model performance metrics")
    print("3. Document findings in your IEEE paper")
    print("4. Consider hyperparameter tuning for better performance")
    

if __name__ == "__main__":
    main()
