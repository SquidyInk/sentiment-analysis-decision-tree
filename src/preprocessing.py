"""
Data Preprocessing Module for Sentiment Analysis
CSC 108 - Project 2
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """
    A class to handle text preprocessing for sentiment analysis.
    
    Time Complexity Analysis:
    - clean_text(): O(n) where n is the length of text
    - preprocess_dataset(): O(m*n) where m is number of texts, n is avg text length
    """
    
    def __init__(self, use_stemming=False, use_lemmatization=True, remove_stopwords=True):
        """
        Initialize the preprocessor.
        
        Args:
            use_stemming (bool): Whether to apply stemming
            use_lemmatization (bool): Whether to apply lemmatization
            remove_stopwords (bool): Whether to remove stopwords
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def clean_text(self, text):
        """
        Clean and preprocess a single text string.
        
        Time Complexity: O(n) where n is the length of the text
        Space Complexity: O(n) for storing cleaned text
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Apply stemming or lemmatization
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(word) for word in tokens]
        elif self.use_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_dataset(self, df, text_column, label_column=None):
        """
        Preprocess an entire dataset.
        
        Time Complexity: O(m*n) where m is number of texts, n is avg text length
        Space Complexity: O(m*n) for storing processed dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of the column containing text
            label_column (str): Name of the column containing labels
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df_processed = df.copy()
        
        # Clean text
        print("Cleaning text data...")
        df_processed['cleaned_text'] = df_processed[text_column].apply(self.clean_text)
        
        # Remove empty texts
        df_processed = df_processed[df_processed['cleaned_text'].str.len() > 0]
        
        return df_processed


def load_and_split_data(df, text_column, label_column, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Time Complexity: O(m) where m is the number of samples
    Space Complexity: O(m) for storing train/test splits
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        text_column (str): Name of column with text data
        label_column (str): Name of column with labels
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df[text_column]
    y = df[label_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test, method='tfidf', max_features=5000):
    """
    Convert text to numerical features using TF-IDF or Count Vectorization.
    
    Time Complexity: O(m*n*f) where m is samples, n is avg text length, f is features
    Space Complexity: O(m*f) for feature matrix
    
    Args:
        X_train: Training text data
        X_test: Testing text data
        method (str): 'tfidf' or 'count'
        max_features (int): Maximum number of features
        
    Returns:
        tuple: Vectorized X_train, X_test, and vectorizer object
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    else:
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    
    print(f"Vectorizing text using {method.upper()}...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    
    return X_train_vec, X_test_vec, vectorizer


if __name__ == "__main__":
    # Example usage
    print("Text Preprocessing Module - CSC 108 Project 2")
    print("=" * 50)
    
    # Example text
    sample_text = "This is an AMAZING product! I love it so much!!! 😊 #happy"
    
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean_text(sample_text)
    
    print(f"\nOriginal: {sample_text}")
    print(f"Cleaned:  {cleaned}")
