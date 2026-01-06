# Sentiment Analysis Decision Tree - Quick Reference

This quick reference guide provides essential commands and code snippets for working with the sentiment analysis decision tree project.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Data Loading](#data-loading)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Visualization](#visualization)
- [Debugging Tips](#debugging-tips)
- [Common Workflows](#common-workflows)

---

## Environment Setup

### Initial Setup
```bash
# Clone the repository
git clone https://github.com/SquidyInk/sentiment-analysis-decision-tree.git
cd sentiment-analysis-decision-tree

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Data Loading

### Load CSV Data
```python
# Basic loading
df = pd.read_csv('data/sentiment_data.csv')

# Load with specific encoding
df = pd.read_csv('data/sentiment_data.csv', encoding='utf-8')

# Load with specific columns
df = pd.read_csv('data/sentiment_data.csv', usecols=['text', 'sentiment'])

# Quick data inspection
print(df.head())
print(df.info())
print(df.describe())
print(df['sentiment'].value_counts())
```

### Handle Missing Data
```python
# Check for missing values
print(df.isnull().sum())

# Drop missing values
df = df.dropna()

# Fill missing values
df['text'] = df['text'].fillna('')
```

---

## Data Preprocessing

### Text Cleaning
```python
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def clean_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)
```

### Remove Stopwords
```python
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    """Remove stopwords from text"""
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

df['clean_text'] = df['clean_text'].apply(remove_stopwords)
```

### Stemming and Lemmatization
```python
# Stemming
stemmer = PorterStemmer()
def stem_text(text):
    words = text.split()
    return ' '.join([stemmer.stem(word) for word in words])

df['stemmed_text'] = df['clean_text'].apply(stem_text)

# Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

df['lemmatized_text'] = df['clean_text'].apply(lemmatize_text)
```

---

## Feature Engineering

### TF-IDF Vectorization
```python
# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

# Fit and transform
X_tfidf = tfidf.fit_transform(df['clean_text'])

# Convert to array for inspection
X_array = X_tfidf.toarray()
feature_names = tfidf.get_feature_names_out()
```

### Count Vectorization (Bag of Words)
```python
# Initialize Count vectorizer
count_vec = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

# Fit and transform
X_count = count_vec.fit_transform(df['clean_text'])
```

### Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

# Encode sentiment labels
le = LabelEncoder()
y = le.fit_transform(df['sentiment'])

# View mapping
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Label mapping: {label_mapping}")
```

---

## Model Training

### Train-Test Split
```python
# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Check split sizes
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
```

### Decision Tree Training
```python
# Basic Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Decision Tree with hyperparameters
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    criterion='gini',
    random_state=42
)
dt_model.fit(X_train, y_train)
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Grid search
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

---

## Model Evaluation

### Basic Metrics
```python
# Make predictions
y_pred = dt_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')

print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

### Feature Importance
```python
# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Important Features:")
print(feature_importance.head(10))
```

---

## Visualization

### Confusion Matrix Heatmap
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
```

### Feature Importance Plot
```python
# Plot top features
top_n = 20
top_features = feature_importance.head(top_n)

plt.figure(figsize=(10, 8))
plt.barh(range(top_n), top_features['importance'])
plt.yticks(range(top_n), top_features['feature'])
plt.xlabel('Importance')
plt.title(f'Top {top_n} Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()
```

### Decision Tree Visualization
```python
from sklearn.tree import plot_tree

# Plot decision tree (limited depth for clarity)
plt.figure(figsize=(20, 10))
plot_tree(dt_model, 
          max_depth=3,
          feature_names=feature_names,
          class_names=le.classes_,
          filled=True,
          fontsize=10)
plt.title('Decision Tree Visualization (max depth=3)')
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300)
plt.show()
```

### Learning Curves
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    dt_model, X_train, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300)
plt.show()
```

---

## Debugging Tips

### Check Data Issues
```python
# Check for imbalanced classes
print("Class distribution:")
print(df['sentiment'].value_counts())
print("\nClass percentages:")
print(df['sentiment'].value_counts(normalize=True))

# Check text length distribution
df['text_length'] = df['text'].str.len()
print(f"\nText length stats:")
print(df['text_length'].describe())

# Find empty or very short texts
short_texts = df[df['text_length'] < 10]
print(f"\nTexts shorter than 10 characters: {len(short_texts)}")
```

### Model Overfitting Check
```python
# Compare training vs test accuracy
train_pred = dt_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Difference: {abs(train_accuracy - test_accuracy):.4f}")

if abs(train_accuracy - test_accuracy) > 0.1:
    print("Warning: Model may be overfitting!")
```

### Memory Usage
```python
# Check memory usage
print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Feature matrix shape: {X_tfidf.shape}")
print(f"Feature matrix size: {X_tfidf.data.nbytes / 1024**2:.2f} MB")
```

### Debugging Predictions
```python
# Examine misclassified examples
misclassified_idx = np.where(y_test != y_pred)[0]
print(f"\nNumber of misclassified examples: {len(misclassified_idx)}")

# Show some misclassified examples
X_test_df = df.iloc[X_test.indices] if hasattr(X_test, 'indices') else df
for idx in misclassified_idx[:5]:
    print(f"\nText: {X_test_df.iloc[idx]['text'][:100]}...")
    print(f"True: {le.inverse_transform([y_test[idx]])[0]}")
    print(f"Predicted: {le.inverse_transform([y_pred[idx]])[0]}")
```

---

## Common Workflows

### Complete Training Pipeline
```python
def train_sentiment_model(data_path, test_size=0.2):
    """Complete pipeline for training sentiment analysis model"""
    
    # 1. Load data
    df = pd.read_csv(data_path)
    
    # 2. Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # 3. Vectorize
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['clean_text'])
    
    # 4. Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    
    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 6. Train model
    model = DecisionTreeClassifier(
        max_depth=15, min_samples_split=20, random_state=42
    )
    model.fit(X_train, y_train)
    
    # 7. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return model, tfidf, le
```

### Predict New Text
```python
def predict_sentiment(text, model, vectorizer, label_encoder):
    """Predict sentiment for new text"""
    
    # Clean text
    clean = clean_text(text)
    
    # Vectorize
    X = vectorizer.transform([clean])
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    # Decode
    sentiment = label_encoder.inverse_transform([prediction])[0]
    
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Confidence: {max(probability):.4f}")
    
    return sentiment

# Usage
predict_sentiment("This product is amazing!", dt_model, tfidf, le)
```

### Save and Load Model
```python
import pickle

# Save model and components
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump({
        'model': dt_model,
        'vectorizer': tfidf,
        'label_encoder': le
    }, f)

# Load model
with open('sentiment_model.pkl', 'rb') as f:
    saved_objects = pickle.load(f)
    loaded_model = saved_objects['model']
    loaded_vectorizer = saved_objects['vectorizer']
    loaded_le = saved_objects['label_encoder']
```

---

## Quick Tips

1. **Always check data balance** - Imbalanced datasets can lead to biased models
2. **Start simple** - Begin with basic parameters, then tune
3. **Monitor overfitting** - Use cross-validation and check train/test gap
4. **Feature engineering matters** - Try different vectorization techniques
5. **Visualize results** - Confusion matrices and feature importance are insightful
6. **Save your work** - Pickle trained models for later use
7. **Document hyperparameters** - Keep track of what works best
8. **Use random_state** - For reproducible results
9. **Clean text thoroughly** - Preprocessing significantly impacts performance
10. **Test on real examples** - Always validate with actual text samples

---

## Additional Resources

- Scikit-learn Documentation: https://scikit-learn.org/
- NLTK Documentation: https://www.nltk.org/
- Pandas Documentation: https://pandas.pydata.org/

---

*Last Updated: 2026-01-06*
