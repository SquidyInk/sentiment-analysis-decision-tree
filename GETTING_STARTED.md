# Getting Started with Sentiment Analysis Decision Tree

Welcome to the Sentiment Analysis Decision Tree project! This guide will help you set up, understand, and extend this CSC 108 project.

## Table of Contents
- [Installation Instructions](#installation-instructions)
- [Dataset Selection](#dataset-selection)
- [Running the Project](#running-the-project)
- [Understanding the Code](#understanding-the-code)
- [Next Steps for Your CSC 108 Project](#next-steps-for-your-csc-108-project)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)
- [Student Checklist](#student-checklist)

---

## Installation Instructions

### Prerequisites
- Python 3.7 or higher installed on your computer
- Basic familiarity with the command line/terminal
- A text editor or IDE (VS Code, PyCharm, or IDLE)

### Step 1: Clone the Repository
```bash
git clone https://github.com/SquidyInk/sentiment-analysis-decision-tree.git
cd sentiment-analysis-decision-tree
```

If you don't have Git installed, you can download the repository as a ZIP file from GitHub.

### Step 2: Set Up a Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

If there's no `requirements.txt`, you may need to install packages individually:
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Step 4: Verify Installation
```bash
python --version
python -c "import numpy, pandas; print('All packages installed successfully!')"
```

---

## Dataset Selection

### Understanding the Dataset
This project uses text data for sentiment analysis. Common datasets include:

1. **Movie Reviews Dataset** - Reviews labeled as positive or negative
2. **Twitter Sentiment Dataset** - Tweets with sentiment labels
3. **Product Reviews** - Amazon or Yelp reviews with ratings

### Preparing Your Dataset
Your dataset should have:
- **Text column**: The review or comment
- **Label column**: Sentiment (positive/negative or 1/0)

Example CSV format:
```csv
text,sentiment
"This movie was amazing!",positive
"Terrible experience, would not recommend.",negative
```

### Where to Find Datasets
- **Built-in datasets**: Check if the project includes sample data in a `data/` folder
- **Kaggle**: [kaggle.com/datasets](https://www.kaggle.com/datasets)
- **UCI Machine Learning Repository**: [archive.ics.uci.edu/ml](https://archive.ics.uci.edu/ml)
- **Create your own**: Collect reviews from public sources (remember to respect terms of service!)

---

## Running the Project

### Basic Execution
```bash
python main.py
```

### Common Command-Line Arguments
```bash
# Specify a dataset
python main.py --dataset data/reviews.csv

# Adjust tree depth
python main.py --max-depth 5

# Enable visualization
python main.py --visualize
```

### Expected Output
You should see:
1. Data loading confirmation
2. Training progress
3. Model accuracy metrics
4. Decision tree structure (if visualization is enabled)
5. Sample predictions

### Example Run
```
Loading dataset...
Dataset loaded: 1000 samples
Training decision tree...
Training complete!

Accuracy: 85.3%
Precision: 0.87
Recall: 0.83

Sample predictions:
Text: "I loved this product!"
Predicted sentiment: Positive (Confidence: 92%)
```

---

## Understanding the Code

### Project Structure
```
sentiment-analysis-decision-tree/
├── main.py                 # Main execution script
├── decision_tree.py        # Decision tree implementation
├── preprocessing.py        # Text preprocessing functions
├── feature_extraction.py   # Feature extraction from text
├── evaluation.py           # Model evaluation metrics
├── data/                   # Dataset folder
├── models/                 # Saved models
└── visualizations/         # Generated plots
```

### Key Concepts

#### 1. **Text Preprocessing**
- **Tokenization**: Breaking text into words
- **Lowercasing**: Converting all text to lowercase
- **Stop word removal**: Removing common words (the, is, and)
- **Stemming/Lemmatization**: Reducing words to root form

```python
# Example preprocessing
text = "This movie is AMAZING!!!"
# After preprocessing: "movie amazing"
```

#### 2. **Feature Extraction**
- **Bag of Words**: Counting word occurrences
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **Word presence**: Binary features (word exists or not)

#### 3. **Decision Tree Algorithm**
- Splits data based on features
- Creates rules like: "If 'amazing' in text AND 'love' in text → Positive"
- Uses metrics like Gini impurity or entropy to determine best splits

#### 4. **Model Evaluation**
- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted positive, how many are actually positive
- **Recall**: Of actual positive, how many were predicted positive
- **F1-Score**: Harmonic mean of precision and recall

---

## Next Steps for Your CSC 108 Project

### Beginner Extensions
1. **Add more preprocessing steps**
   - Remove punctuation
   - Handle emojis
   - Normalize numbers

2. **Experiment with features**
   - Add word length features
   - Count exclamation marks
   - Detect capitalized words (shouting)

3. **Improve visualization**
   - Create confusion matrix
   - Plot word clouds for positive/negative sentiments
   - Show decision tree graphically

### Intermediate Extensions
4. **Implement cross-validation**
   - Split data into multiple folds
   - Test model stability

5. **Add command-line interface**
   - Allow users to input text for real-time prediction
   - Save and load trained models

6. **Compare with other algorithms**
   - Naive Bayes
   - Logistic Regression
   - Random Forest

### Advanced Extensions
7. **Handle multi-class sentiment**
   - Instead of just positive/negative, add neutral
   - Add intensity levels (very positive, slightly positive, etc.)

8. **Build a web interface**
   - Use Flask or Streamlit
   - Create an interactive demo

9. **Analyze model decisions**
   - Extract most important features
   - Explain why the model made specific predictions

---

## Troubleshooting Common Issues

### Issue 1: Module Not Found Error
```
ModuleNotFoundError: No module named 'numpy'
```
**Solution**: Install the missing package
```bash
pip install numpy
```

### Issue 2: Dataset File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/reviews.csv'
```
**Solution**: 
- Check the file path is correct
- Ensure the dataset file exists in the specified location
- Use absolute path if relative path doesn't work

### Issue 3: Low Accuracy (Below 60%)
**Possible causes**:
- Dataset too small (need more training examples)
- Poor feature extraction
- Overfitting (tree too deep) or underfitting (tree too shallow)

**Solutions**:
- Increase training data
- Adjust `max_depth` parameter
- Improve preprocessing
- Try different features

### Issue 4: Memory Error with Large Datasets
```
MemoryError: Unable to allocate array
```
**Solution**:
- Use a smaller subset of data for testing
- Reduce feature dimensionality
- Use sparse matrices for bag-of-words

### Issue 5: Encoding Errors
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```
**Solution**:
```python
# When reading CSV files
pd.read_csv('file.csv', encoding='latin-1')
# or
pd.read_csv('file.csv', encoding='ISO-8859-1')
```

---

## Student Checklist

### Setup Phase
- [ ] Python 3.7+ installed and verified
- [ ] Repository cloned or downloaded
- [ ] Virtual environment created (optional)
- [ ] All required packages installed
- [ ] Test run successful

### Understanding Phase
- [ ] Read through all code files
- [ ] Understand preprocessing steps
- [ ] Understand feature extraction
- [ ] Understand decision tree algorithm
- [ ] Understand evaluation metrics

### Dataset Phase
- [ ] Dataset selected or created
- [ ] Dataset properly formatted (CSV with text and labels)
- [ ] Dataset split into training and testing sets
- [ ] Data quality checked (no missing values, balanced classes)

### Experimentation Phase
- [ ] Baseline model trained and evaluated
- [ ] At least 3 different preprocessing approaches tested
- [ ] At least 3 different hyperparameters tested
- [ ] Results documented and compared

### Extension Phase
- [ ] At least one extension implemented
- [ ] Extension tested and working
- [ ] Extension documented in code comments

### Documentation Phase
- [ ] Code properly commented
- [ ] README updated with your changes
- [ ] Results and findings documented
- [ ] Challenges and solutions noted

### Presentation Phase (if applicable)
- [ ] Demo prepared
- [ ] Slides created explaining the project
- [ ] Examples ready to show
- [ ] Able to explain how the decision tree works

---

## Additional Resources

### Learning Materials
- **Python Documentation**: [docs.python.org](https://docs.python.org/3/)
- **Scikit-learn Tutorials**: [scikit-learn.org/stable/tutorial](https://scikit-learn.org/stable/tutorial/)
- **Decision Trees Explained**: Search for "Decision Tree Classification" on YouTube
- **NLP Basics**: Natural Language Processing tutorials

### Getting Help
1. **Check the Issues tab** on GitHub for common problems
2. **Ask your TA** during office hours
3. **CSC 108 Discussion Board** for course-specific questions
4. **Stack Overflow** for technical Python questions

### Tips for Success
- **Start early** - Don't wait until the deadline
- **Test incrementally** - Test each small change before moving on
- **Comment your code** - You'll thank yourself later
- **Save your work often** - Use Git commits regularly
- **Ask questions** - No question is too small!

---

## Good Luck! 🚀

Remember, the goal is to learn and understand the concepts, not just to get it working. Take time to experiment, break things (in a safe environment!), and see what happens. That's how real learning occurs!

If you make improvements or find bugs, consider contributing back to the project by creating a pull request.

**Happy coding!** 💻✨
