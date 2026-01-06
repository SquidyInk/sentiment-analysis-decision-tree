# Dataset Guide for Sentiment Analysis

This guide provides comprehensive information about recommended datasets for training and testing sentiment analysis models using decision trees. Each dataset includes download instructions, preprocessing tips, and use case recommendations.

## Table of Contents

1. [Twitter Sentiment140](#1-twitter-sentiment140)
2. [Amazon Product Reviews](#2-amazon-product-reviews)
3. [IMDB Movie Reviews](#3-imdb-movie-reviews)
4. [Restaurant Reviews](#4-restaurant-reviews)
5. [Financial News Sentiment](#5-financial-news-sentiment)
6. [Steam Game Reviews](#6-steam-game-reviews)
7. [COVID-19 Tweets](#7-covid-19-tweets)
8. [Apple App Store Reviews](#8-apple-app-store-reviews)
9. [General Preprocessing Tips](#general-preprocessing-tips)
10. [Dataset Comparison](#dataset-comparison)

---

## 1. Twitter Sentiment140

### Overview
- **Size**: 1.6 million tweets
- **Classes**: Binary (positive, negative)
- **Domain**: Social media, general topics
- **Language**: English

### Description
The Sentiment140 dataset contains tweets extracted using the Twitter API. The tweets are annotated (0 = negative, 4 = positive) and were originally labeled based on emoticons present in the text.

### Download Instructions

**Option 1: Direct Download**
```bash
wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip
```

**Option 2: Kaggle**
```bash
kaggle datasets download -d kazanova/sentiment140
```

### Dataset Format
- **Columns**: target, ids, date, flag, user, text
- **Target values**: 0 (negative), 4 (positive)

### Preprocessing Tips
```python
import pandas as pd
import re

# Load dataset
df = pd.read_csv('training.1600000.processed.noemoticon.csv', 
                 encoding='latin-1', 
                 names=['target', 'ids', 'date', 'flag', 'user', 'text'])

# Convert target: 0 -> 0 (negative), 4 -> 1 (positive)
df['target'] = df['target'].replace(4, 1)

# Clean text
def clean_tweet(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove special characters
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

df['cleaned_text'] = df['text'].apply(clean_tweet)
```

### Use Cases
- General sentiment analysis
- Social media monitoring
- Brand reputation tracking

---

## 2. Amazon Product Reviews

### Overview
- **Size**: 142.8 million reviews (various subsets available)
- **Classes**: 5-star rating (can be converted to binary/multi-class)
- **Domain**: E-commerce, product reviews
- **Language**: Primarily English

### Description
Amazon product reviews spanning multiple categories including electronics, books, clothing, and more. Reviews include ratings, helpfulness votes, and review text.

### Download Instructions

**Option 1: Full Dataset**
```bash
# Visit: https://nijianmo.github.io/amazon/index.html
# Select category and download JSON files
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Electronics_5.json.gz
gunzip Electronics_5.json.gz
```

**Option 2: Kaggle Subset**
```bash
kaggle datasets download -d bittlingmayer/amazonreviews
```

### Dataset Format
- **Fields**: reviewerID, asin, reviewText, overall, summary, unixReviewTime

### Preprocessing Tips
```python
import pandas as pd
import json

# Load JSON data
data = []
with open('Electronics_5.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Convert ratings to sentiment classes
# Binary classification
df['sentiment'] = df['overall'].apply(lambda x: 1 if x >= 4 else 0)

# Three-class classification
def rating_to_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment_multi'] = df['overall'].apply(rating_to_sentiment)

# Combine review text and summary
df['full_text'] = df['summary'].fillna('') + ' ' + df['reviewText'].fillna('')

# Remove short reviews
df = df[df['full_text'].str.len() > 20]
```

### Use Cases
- Product sentiment analysis
- Customer feedback analysis
- E-commerce recommendation systems

---

## 3. IMDB Movie Reviews

### Overview
- **Size**: 50,000 reviews
- **Classes**: Binary (positive, negative)
- **Domain**: Entertainment, movies
- **Language**: English

### Description
A balanced dataset of movie reviews from IMDB, evenly split between positive and negative reviews. Widely used as a benchmark for sentiment analysis.

### Download Instructions

**Option 1: Official Source**
```bash
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

**Option 2: Keras/TensorFlow**
```python
from tensorflow.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data()
```

**Option 3: Hugging Face**
```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

### Dataset Format
- **Structure**: Separate folders for train/test and pos/neg
- **Files**: Individual text files per review

### Preprocessing Tips
```python
import os
import pandas as pd

def load_imdb_data(data_dir):
    reviews = []
    labels = []
    
    for sentiment in ['pos', 'neg']:
        path = os.path.join(data_dir, sentiment)
        label = 1 if sentiment == 'pos' else 0
        
        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    labels.append(label)
    
    return pd.DataFrame({'text': reviews, 'sentiment': labels})

# Load train data
train_df = load_imdb_data('aclImdb/train')

# Remove HTML tags
import re
def remove_html(text):
    return re.sub(r'<.*?>', '', text)

train_df['cleaned_text'] = train_df['text'].apply(remove_html)
```

### Use Cases
- Movie review analysis
- Entertainment industry insights
- Benchmark testing for sentiment models

---

## 4. Restaurant Reviews

### Overview
- **Size**: Various datasets (1,000 - 10,000+ reviews)
- **Classes**: Binary or multi-class
- **Domain**: Food service, hospitality
- **Language**: English

### Description
Restaurant reviews from various sources including Yelp and OpenTable, containing customer feedback about dining experiences.

### Download Instructions

**Option 1: Yelp Dataset**
```bash
# Register at https://www.yelp.com/dataset
# Download the complete dataset (JSON format)
```

**Option 2: Kaggle Restaurant Reviews**
```bash
kaggle datasets download -d vigneshwarsofficial/reviews
```

**Option 3: UCI Restaurant Reviews**
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip
unzip sentiment\ labelled\ sentences.zip
```

### Dataset Format
- **Yelp Format**: JSON with business_id, stars, text, date
- **Simple Format**: Tab-separated (review_text, label)

### Preprocessing Tips
```python
import pandas as pd

# For simple format
df = pd.read_csv('restaurant_reviews.tsv', 
                 delimiter='\t', 
                 names=['text', 'sentiment'])

# For Yelp JSON
import json
reviews = []
with open('yelp_academic_dataset_review.json', 'r') as f:
    for line in f:
        reviews.append(json.loads(line))

df = pd.DataFrame(reviews)

# Filter for restaurant reviews only
# Merge with business data to filter by category
business_df = pd.read_json('yelp_academic_dataset_business.json', lines=True)
restaurants = business_df[business_df['categories'].str.contains('Restaurant', na=False)]
df = df[df['business_id'].isin(restaurants['business_id'])]

# Convert stars to binary sentiment
df['sentiment'] = df['stars'].apply(lambda x: 1 if x >= 4 else 0)

# Handle common restaurant-specific terms
def normalize_food_terms(text):
    text = text.lower()
    # Normalize common variations
    text = text.replace("delicious", "good")
    text = text.replace("yummy", "good")
    text = text.replace("terrible", "bad")
    return text

df['normalized_text'] = df['text'].apply(normalize_food_terms)
```

### Use Cases
- Restaurant reputation management
- Food service quality analysis
- Customer experience tracking

---

## 5. Financial News Sentiment

### Overview
- **Size**: 4,000+ financial news headlines
- **Classes**: Binary or three-class (positive, neutral, negative)
- **Domain**: Finance, business news
- **Language**: English

### Description
Financial news headlines and articles labeled with sentiment, useful for stock market prediction and financial analysis.

### Download Instructions

**Option 1: Financial Phrasebank**
```bash
# Visit: https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10
# Download manually or use:
wget https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/download
```

**Option 2: Kaggle**
```bash
kaggle datasets download -d ankurzing/sentiment-analysis-for-financial-news
```

### Dataset Format
- **Columns**: Sentiment, News Headline
- **Sentiment values**: positive, negative, neutral

### Preprocessing Tips
```python
import pandas as pd

# Load dataset
df = pd.read_csv('financial_news_sentiment.csv', encoding='latin-1')

# Encode sentiment
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df['sentiment_encoded'] = df['Sentiment'].map(sentiment_map)

# For binary classification (remove neutral)
df_binary = df[df['Sentiment'] != 'neutral'].copy()
df_binary['sentiment_binary'] = df_binary['Sentiment'].apply(
    lambda x: 1 if x == 'positive' else 0
)

# Financial domain-specific preprocessing
def preprocess_financial_text(text):
    # Keep numbers and percentages (important in finance)
    text = text.lower()
    # Normalize financial terms
    text = text.replace('$', 'dollar ')
    text = text.replace('%', ' percent')
    return text

df['processed_text'] = df['News Headline'].apply(preprocess_financial_text)

# Handle financial entities
# Keep company names, stock symbols intact
```

### Use Cases
- Stock market sentiment analysis
- Financial news monitoring
- Investment decision support

---

## 6. Steam Game Reviews

### Overview
- **Size**: Millions of reviews available
- **Classes**: Binary (recommended/not recommended)
- **Domain**: Gaming
- **Language**: Multiple languages (primarily English)

### Description
User reviews from the Steam gaming platform, including play time, helpfulness votes, and recommendation status.

### Download Instructions

**Option 1: Kaggle Steam Reviews**
```bash
kaggle datasets download -d luthfim/steam-reviews-dataset
```

**Option 2: Steam API** (for custom collection)
```python
import requests

def get_steam_reviews(app_id, num_reviews=100):
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        'json': 1,
        'num_per_page': 100,
        'language': 'english',
        'purchase_type': 'all'
    }
    response = requests.get(url, params=params)
    return response.json()
```

### Dataset Format
- **Fields**: app_id, review_text, voted_up, playtime_forever, timestamp_created

### Preprocessing Tips
```python
import pandas as pd

# Load dataset
df = pd.read_csv('steam_reviews.csv')

# Binary sentiment from recommendation
df['sentiment'] = df['voted_up'].astype(int)

# Filter by playtime (remove reviews from users with very little playtime)
df = df[df['playtime_forever'] > 60]  # At least 1 hour

# Handle gaming-specific slang
def clean_gaming_text(text):
    text = text.lower()
    # Common gaming abbreviations
    replacements = {
        'op': 'overpowered',
        'nerf': 'weaken',
        'buff': 'strengthen',
        'fps': 'frames per second',
        'dlc': 'downloadable content',
        'ez': 'easy',
        'gg': 'good game'
    }
    for abbr, full in replacements.items():
        text = text.replace(abbr, full)
    return text

df['cleaned_review'] = df['review_text'].apply(clean_gaming_text)

# Remove very short reviews
df = df[df['review_text'].str.len() > 50]
```

### Use Cases
- Game development feedback
- Gaming community sentiment
- Product recommendation systems

---

## 7. COVID-19 Tweets

### Overview
- **Size**: Millions of tweets (varies by collection period)
- **Classes**: Typically requires manual labeling or weak supervision
- **Domain**: Public health, social issues
- **Language**: Multiple languages

### Description
Tweets related to COVID-19, containing public sentiment about the pandemic, vaccines, lockdowns, and related topics.

### Download Instructions

**Option 1: Kaggle COVID-19 Tweets**
```bash
kaggle datasets download -d datatattle/covid-19-nlp-text-classification
```

**Option 2: IEEE Dataport**
```bash
# Visit: https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset
# Register and download
```

**Option 3: GitHub Repositories**
```bash
git clone https://github.com/thepanacealab/covid19_twitter
```

### Dataset Format
- **Fields**: tweet_id, date, text, user_location, hashtags
- **Note**: Many datasets only provide tweet IDs due to Twitter's terms of service

### Preprocessing Tips
```python
import pandas as pd
import re

# Load dataset
df = pd.read_csv('covid19_tweets.csv')

# COVID-specific preprocessing
def preprocess_covid_tweet(text):
    text = text.lower()
    
    # Normalize COVID-related terms
    covid_terms = ['covid-19', 'covid19', 'coronavirus', 'corona', 'pandemic']
    for term in covid_terms:
        text = text.replace(term, 'covid')
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    
    return text.strip()

df['cleaned_text'] = df['text'].apply(preprocess_covid_tweet)

# Filter by language if needed
df = df[df['lang'] == 'en']

# Remove retweets for original content only
df = df[~df['text'].str.startswith('RT')]

# Labeling approaches (if not pre-labeled)
# 1. Keyword-based weak labeling
positive_keywords = ['hope', 'recover', 'safe', 'vaccinated', 'protected']
negative_keywords = ['死', 'death', 'fear', 'anxiety', 'lockdown', 'scared']

def weak_label(text):
    text_lower = text.lower()
    pos_score = sum(1 for kw in positive_keywords if kw in text_lower)
    neg_score = sum(1 for kw in negative_keywords if kw in text_lower)
    
    if pos_score > neg_score:
        return 'positive'
    elif neg_score > pos_score:
        return 'negative'
    else:
        return 'neutral'

df['weak_sentiment'] = df['cleaned_text'].apply(weak_label)
```

### Use Cases
- Public health sentiment monitoring
- Crisis communication analysis
- Social media trend analysis

---

## 8. Apple App Store Reviews

### Overview
- **Size**: Varies by app (thousands to millions of reviews)
- **Classes**: 5-star rating (convertible to binary/multi-class)
- **Domain**: Mobile applications
- **Language**: Multiple languages

### Description
User reviews from the Apple App Store, including ratings, review text, version information, and helpfulness votes.

### Download Instructions

**Option 1: Kaggle Dataset**
```bash
kaggle datasets download -d snap/amazon-fine-food-reviews
# Or search for specific app reviews on Kaggle
```

**Option 2: iTunes API** (for custom collection)
```python
import requests

def get_app_reviews(app_id, country='us', page=1):
    url = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostRecent/page={page}/json"
    response = requests.get(url)
    return response.json()
```

**Option 3: App Store Scraper**
```bash
pip install app-store-scraper

# Python usage
from app_store_scraper import AppStore
app = AppStore(country='us', app_name='your-app-name', app_id='12345')
app.review(how_many=1000)
reviews_df = pd.DataFrame(app.reviews)
```

### Dataset Format
- **Fields**: review_id, userName, rating, title, review, date, version

### Preprocessing Tips
```python
import pandas as pd
from datetime import datetime

# Load dataset
df = pd.read_csv('app_store_reviews.csv')

# Convert rating to sentiment
# Binary classification
df['sentiment_binary'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Three-class classification
def rating_to_sentiment_3class(rating):
    if rating >= 4:
        return 2  # positive
    elif rating == 3:
        return 1  # neutral
    else:
        return 0  # negative

df['sentiment_3class'] = df['rating'].apply(rating_to_sentiment_3class)

# Combine title and review
df['full_review'] = df['title'].fillna('') + '. ' + df['review'].fillna('')

# Filter by date (e.g., recent reviews only)
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2023-01-01']

# Remove empty reviews
df = df[df['full_review'].str.len() > 10]

# Handle app-specific preprocessing
def preprocess_app_review(text):
    text = text.lower()
    
    # Remove version mentions
    text = re.sub(r'version\s+\d+\.\d+', '', text)
    
    # Normalize common app-related terms
    text = text.replace('app', 'application')
    text = text.replace('ui', 'user interface')
    text = text.replace('ux', 'user experience')
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s\.\!\?]', '', text)
    
    return text.strip()

df['cleaned_review'] = df['full_review'].apply(preprocess_app_review)
```

### Use Cases
- App quality monitoring
- Feature request identification
- User experience analysis
- Competitive analysis

---

## General Preprocessing Tips

### 1. Text Cleaning Pipeline

```python
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, use_stemming=False, use_lemmatization=True):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags (for social media)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove numbers (optional - depends on use case)
        # text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_process(self, text):
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Stemming
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        # Lemmatization
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.tokenize_and_process(text)
        return text

# Usage
preprocessor = TextPreprocessor()
df['processed_text'] = df['text'].apply(preprocessor.preprocess)
```

### 2. Handling Imbalanced Data

```python
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from collections import Counter

# Check class distribution
print(Counter(df['sentiment']))

# Method 1: Undersampling majority class
df_majority = df[df['sentiment'] == 1]
df_minority = df[df['sentiment'] == 0]

df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=len(df_minority),
                                   random_state=42)

df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Method 2: Oversampling minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Method 3: SMOTE (for feature vectors)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['processed_text'])
y = df['sentiment']

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

### 3. Feature Extraction for Decision Trees

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Method 1: Bag of Words
bow_vectorizer = CountVectorizer(max_features=500, 
                                  ngram_range=(1, 2),
                                  min_df=5,
                                  max_df=0.8)
X_bow = bow_vectorizer.fit_transform(df['processed_text'])

# Method 2: TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=500,
                                    ngram_range=(1, 2),
                                    min_df=5,
                                    max_df=0.8)
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])

# Method 3: Custom features
def extract_features(text):
    features = {}
    
    # Length features
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)
    features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
    
    # Sentiment-related features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    # Positive/negative word counts
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'worst']
    
    features['positive_word_count'] = sum(1 for word in positive_words if word in text.lower())
    features['negative_word_count'] = sum(1 for word in negative_words if word in text.lower())
    
    return features

# Apply custom features
feature_df = pd.DataFrame([extract_features(text) for text in df['processed_text']])
```

### 4. Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Simple split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# With validation set
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
```

---

## Dataset Comparison

| Dataset | Size | Classes | Domain | Difficulty | Best For |
|---------|------|---------|--------|------------|----------|
| Sentiment140 | 1.6M | Binary | Social Media | Easy-Medium | Beginners, social media analysis |
| Amazon Reviews | 142M+ | 5-star | E-commerce | Medium | Product analysis, multi-class |
| IMDB Reviews | 50K | Binary | Entertainment | Medium | Benchmark testing |
| Restaurant Reviews | 10K+ | Binary/Multi | Food Service | Easy-Medium | Domain-specific analysis |
| Financial News | 4K+ | 3-class | Finance | Hard | Financial applications |
| Steam Reviews | Millions | Binary | Gaming | Medium | Gaming industry |
| COVID-19 Tweets | Millions | Varies | Public Health | Hard | Real-time monitoring |
| App Store Reviews | Varies | 5-star | Mobile Apps | Medium | App development |

### Recommendations by Experience Level

**Beginners:**
- Start with IMDB Movie Reviews (clean, balanced, well-structured)
- Restaurant Reviews (smaller size, straightforward)

**Intermediate:**
- Twitter Sentiment140 (larger dataset, social media preprocessing)
- Amazon Product Reviews (multi-class option, real-world data)

**Advanced:**
- Financial News Sentiment (complex domain knowledge required)
- COVID-19 Tweets (noisy data, requires sophisticated preprocessing)
- Multiple datasets combined for robust models

---

## Additional Resources

### Useful Libraries
```bash
pip install pandas numpy scikit-learn nltk textblob
pip install wordcloud matplotlib seaborn
pip install imbalanced-learn
```

### Further Reading
- [Sentiment Analysis Guide](https://www.nltk.org/howto/sentiment.html)
- [Text Preprocessing Techniques](https://scikit-learn.org/stable/modules/feature_extraction.html)
- [Handling Imbalanced Datasets](https://imbalanced-learn.org/)

### Contributing
Feel free to suggest additional datasets or preprocessing techniques by opening an issue or pull request!

---

**Last Updated**: January 2026
