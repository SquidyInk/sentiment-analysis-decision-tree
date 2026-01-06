"""
Decision Tree Model Implementation for Sentiment Analysis
CSC 108 - Project 2
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import joblib


class SentimentDecisionTree:
    """
    Decision Tree Classifier for Sentiment Analysis.
    
    Time Complexity Analysis:
    - Training: O(n * m * log(m)) where n is features, m is samples
    - Prediction: O(log(m)) for a balanced tree
    - Space Complexity: O(m) for storing the tree structure
    """
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, random_state=42):
        """
        Initialize the Decision Tree model.
        
        Args:
            criterion (str): 'gini' or 'entropy'
            max_depth (int): Maximum depth of tree
            min_samples_split (int): Minimum samples required to split
            min_samples_leaf (int): Minimum samples required at leaf
            random_state (int): Random seed
        """
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.best_params = None
    
    def train(self, X_train, y_train):
        """
        Train the decision tree model.
        
        Time Complexity: O(n * m * log(m))
        - n = number of features
        - m = number of samples
        - log(m) = tree depth
        
        Space Complexity: O(m) for storing tree nodes
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Training Decision Tree Classifier...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Time Complexity: O(k * log(m))
        - k = number of samples to predict
        - log(m) = average tree depth
        
        Args:
            X: Features to predict
            
        Returns:
            array: Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates for predictions.
        
        Time Complexity: O(k * log(m))
        
        Args:
            X: Features to predict
            
        Returns:
            array: Probability estimates
        """
        return self.model.predict_proba(X)
    
    def optimize_hyperparameters(self, X_train, y_train, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Time Complexity: O(p * n * m * log(m) * cv)
        - p = number of parameter combinations
        - cv = number of cross-validation folds
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters found
        """
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 20, 30, 40, 50, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'splitter': ['best', 'random']
        }
        
        print("Optimizing hyperparameters...")
        print(f"Testing {np.prod([len(v) for v in param_grid.values()])} combinations...")
        
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.best_params
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Time Complexity: O(cv * n * m * log(m))
        
        Args:
            X: Features
            y: Labels
            cv (int): Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from the trained model.
        
        Time Complexity: O(n) where n is number of features
        
        Args:
            feature_names (list): Names of features
            
        Returns:
            dict: Feature importance scores
        """
        importances = self.model.feature_importances_
        
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        
        return importances
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self
    
    def get_tree_depth(self):
        """
        Get the depth of the trained tree.
        
        Returns:
            int: Tree depth
        """
        return self.model.get_depth()
    
    def get_n_leaves(self):
        """
        Get the number of leaves in the tree.
        
        Returns:
            int: Number of leaves
        """
        return self.model.get_n_leaves()


if __name__ == "__main__":
    print("Decision Tree Model - CSC 108 Project 2")
    print("=" * 50)
    print("\nComplexity Analysis:")
    print("Training: O(n * m * log(m))")
    print("Prediction: O(log(m))")
    print("Space: O(m)")
