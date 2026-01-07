# CSC 108 Project Submission Guide

## Sentiment Analysis Decision Tree Project

**Last Updated:** January 7, 2026

---

## Table of Contents
1. [Submission Overview](#submission-overview)
2. [Required Components](#required-components)
3. [IEEE Paper Submission](#ieee-paper-submission)
4. [Source Code Submission](#source-code-submission)
5. [Dataset Submission](#dataset-submission)
6. [Submission Checklist](#submission-checklist)
7. [Important Deadlines](#important-deadlines)
8. [Troubleshooting](#troubleshooting)

---

## Submission Overview

This guide provides comprehensive instructions for submitting your CSC 108 Sentiment Analysis Decision Tree project. Your submission must include:

- **IEEE-formatted Research Paper**
- **Complete Source Code**
- **Dataset and Data Files**
- **Documentation**

All submissions must be completed through the designated course portal unless otherwise specified by your instructor.

---

## Required Components

### 1. IEEE Paper (Required)
- Format: PDF
- Conference template: IEEE Conference Paper Format
- Page limit: 6-8 pages (including references)
- File naming: `CSC108_LastName_FirstName_Paper.pdf`

### 2. Source Code (Required)
- Language: Python 3.x
- Format: `.py` files or Jupyter notebooks (`.ipynb`)
- File naming: Descriptive names (e.g., `decision_tree_classifier.py`, `data_preprocessing.py`)

### 3. Dataset (Required)
- Format: CSV, JSON, or appropriate data format
- Include both raw and preprocessed datasets
- File naming: `dataset_raw.csv`, `dataset_processed.csv`

### 4. Supporting Documentation (Required)
- README.md with setup instructions
- requirements.txt for Python dependencies
- Any additional configuration files

---

## IEEE Paper Submission

### Paper Structure

Your IEEE paper should follow this structure:

#### 1. **Title and Author Information**
- Descriptive title reflecting the sentiment analysis focus
- Your name and affiliation
- Abstract (150-250 words)

#### 2. **Introduction**
- Problem statement
- Motivation for sentiment analysis using decision trees
- Research objectives
- Paper organization

#### 3. **Related Work**
- Literature review of sentiment analysis techniques
- Decision tree applications in text classification
- Comparison with other machine learning approaches

#### 4. **Methodology**
- Data collection and preprocessing
- Feature extraction techniques
- Decision tree algorithm implementation
- Training and validation approach
- Evaluation metrics

#### 5. **Experimental Results**
- Performance metrics (accuracy, precision, recall, F1-score)
- Confusion matrix
- Comparison with baseline models
- Visualizations (decision tree diagram, performance graphs)

#### 6. **Discussion**
- Analysis of results
- Strengths and limitations
- Potential improvements

#### 7. **Conclusion**
- Summary of findings
- Future work

#### 8. **References**
- IEEE citation format
- Minimum 10-15 academic references

### Formatting Requirements

```
Page Setup:
- Paper size: US Letter (8.5" × 11")
- Margins: 0.75" all sides
- Columns: Two-column format
- Font: Times New Roman, 10pt
- Line spacing: Single

Sections:
- Section headings: Bold, numbered (e.g., I. INTRODUCTION)
- Subsections: Italic, lettered (e.g., A. Data Collection)

Figures and Tables:
- Centered with captions
- Referenced in text
- High resolution (minimum 300 DPI)

Equations:
- Centered and numbered
- Use IEEE equation editor or LaTeX
```

### IEEE Template Resources

- **Overleaf Template:** Search for "IEEE Conference Template"
- **Microsoft Word Template:** Available on IEEE website
- **LaTeX Template:** `\documentclass[conference]{IEEEtran}`

### Paper Submission Steps

1. **Prepare your manuscript** following the IEEE format
2. **Export to PDF** ensuring all fonts are embedded
3. **Verify file size** is under 10MB
4. **Check PDF for errors** (missing figures, formatting issues)
5. **Name file correctly:** `CSC108_LastName_FirstName_Paper.pdf`
6. **Upload to course portal** before the deadline

---

## Source Code Submission

### Code Organization

Structure your code repository as follows:

```
sentiment-analysis-decision-tree/
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── decision_tree_model.py
│   ├── evaluation.py
│   └── main.py
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── model_training.ipynb
│
├── data/
│   ├── raw/
│   │   └── dataset_raw.csv
│   └── processed/
│       └── dataset_processed.csv
│
├── results/
│   ├── model_performance.txt
│   ├── confusion_matrix.png
│   └── decision_tree_visualization.png
│
├── docs/
│   └── API_documentation.md
│
├── tests/
│   └── test_model.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

### Code Quality Standards

#### 1. **Documentation**
```python
"""
Module: decision_tree_model.py
Description: Implementation of decision tree classifier for sentiment analysis
Author: Your Name
Date: January 7, 2026
"""

def train_decision_tree(X_train, y_train, max_depth=10):
    """
    Train a decision tree classifier for sentiment analysis.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    max_depth : int, optional (default=10)
        Maximum depth of the decision tree
        
    Returns:
    --------
    model : DecisionTreeClassifier
        Trained decision tree model
    """
    # Implementation here
    pass
```

#### 2. **Code Style**
- Follow PEP 8 style guide
- Use meaningful variable names
- Keep functions focused and modular
- Maximum line length: 79-100 characters
- Use type hints where appropriate

#### 3. **Comments**
- Explain complex algorithms
- Document data transformations
- Clarify non-obvious decisions

### requirements.txt

Include all dependencies:

```txt
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
nltk==3.8.0
jupyter==1.0.0
```

### README.md

Your README should include:

```markdown
# Sentiment Analysis Decision Tree

## Project Description
Brief overview of the project and its objectives.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/SquidyInk/sentiment-analysis-decision-tree.git
cd sentiment-analysis-decision-tree

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python src/main.py --train --data data/processed/dataset_processed.csv
```

### Running Predictions
```bash
python src/main.py --predict --text "This product is amazing!"
```

## Dataset
Description of the dataset, including:
- Source
- Size (number of samples)
- Features
- Labels/classes

## Results
Summary of model performance metrics.

## License
Specify the license for your code.

## Contact
Your name and email for questions.
```

### Code Submission Steps

1. **Test your code** thoroughly
2. **Run linters** (e.g., `pylint`, `flake8`)
3. **Verify all dependencies** are in requirements.txt
4. **Create a ZIP archive:**
   ```bash
   zip -r CSC108_LastName_FirstName_Code.zip . -x "*.git*" "*.pyc" "__pycache__/*" "venv/*"
   ```
5. **Upload to course portal**

---

## Dataset Submission

### Dataset Requirements

#### 1. **Raw Dataset**
- Original, unprocessed data
- Include data source documentation
- Format: CSV, JSON, or TSV
- Minimum size: 1,000 samples recommended

#### 2. **Processed Dataset**
- Cleaned and preprocessed data
- Ready for model training
- Include preprocessing script

#### 3. **Data Description**

Create a `DATA_DESCRIPTION.md` file:

```markdown
# Dataset Description

## Overview
- **Name:** [Dataset Name]
- **Source:** [URL or citation]
- **Collection Date:** [Date]
- **Size:** [Number of samples]

## Features
1. **text**: Raw text of the review/comment
2. **sentiment**: Label (positive/negative/neutral)
3. **timestamp**: Date of the review
4. **additional_feature**: Description

## Preprocessing Steps
1. Text cleaning (removal of HTML tags, special characters)
2. Tokenization
3. Stopword removal
4. Lemmatization/Stemming
5. Feature vectorization (TF-IDF, Bag of Words, etc.)

## Train/Test Split
- Training set: 80% (X samples)
- Testing set: 20% (Y samples)
- Validation set: Optional

## Data Ethics
- Privacy considerations
- Consent and usage rights
- Bias mitigation efforts
```

### Dataset Formats

#### CSV Format Example:
```csv
id,text,sentiment,processed_text
1,"I love this product! It's amazing.",positive,"love product amazing"
2,"Terrible experience. Would not recommend.",negative,"terrible experience recommend"
3,"It's okay, nothing special.",neutral,"okay nothing special"
```

#### JSON Format Example:
```json
[
  {
    "id": 1,
    "text": "I love this product! It's amazing.",
    "sentiment": "positive",
    "processed_text": "love product amazing"
  },
  {
    "id": 2,
    "text": "Terrible experience. Would not recommend.",
    "sentiment": "negative",
    "processed_text": "terrible experience recommend"
  }
]
```

### Data Visualization

Include in your submission:
- Class distribution plot
- Word clouds for each sentiment class
- Feature importance visualization

### Dataset Submission Steps

1. **Verify data integrity** (no corrupted files)
2. **Check file sizes** (if large, use compression)
3. **Include data dictionary**
4. **Package datasets:**
   ```bash
   zip CSC108_LastName_FirstName_Data.zip data/ DATA_DESCRIPTION.md
   ```
5. **Upload to course portal** or provide download link

---

## Submission Checklist

Use this checklist before submitting:

### IEEE Paper
- [ ] Follows IEEE conference format
- [ ] 6-8 pages in length
- [ ] All sections complete (Abstract through Conclusion)
- [ ] References in IEEE format (minimum 10)
- [ ] Figures and tables properly captioned
- [ ] PDF exported correctly
- [ ] File named correctly
- [ ] Proofread for grammar and spelling

### Source Code
- [ ] Code runs without errors
- [ ] All dependencies in requirements.txt
- [ ] README.md included and complete
- [ ] Code follows PEP 8 style guidelines
- [ ] Functions and classes documented
- [ ] Comments explain complex logic
- [ ] Test cases included (if applicable)
- [ ] No hardcoded file paths
- [ ] .gitignore includes unnecessary files
- [ ] ZIP file created correctly

### Dataset
- [ ] Raw dataset included
- [ ] Processed dataset included
- [ ] DATA_DESCRIPTION.md complete
- [ ] Data source cited
- [ ] Preprocessing scripts included
- [ ] Data visualizations generated
- [ ] Ethical considerations documented
- [ ] File formats are standard (CSV/JSON)

### Overall
- [ ] All files named according to convention
- [ ] File sizes within limits
- [ ] Submission tested (can extract and run)
- [ ] Plagiarism check completed
- [ ] Academic integrity statement signed

---

## Important Deadlines

**Note:** Verify exact deadlines with your instructor. The following are typical:

- **Project Proposal:** Week 3 (if required)
- **Progress Report:** Week 6 (if required)
- **Final Paper Draft:** Week 10 (for peer review)
- **Complete Submission Deadline:** [INSERT EXACT DATE]
  - IEEE Paper: [DATE] by 11:59 PM
  - Source Code: [DATE] by 11:59 PM
  - Dataset: [DATE] by 11:59 PM

**Late Submission Policy:**
- Check your course syllabus for late penalties
- Request extensions well in advance if needed
- Technical issues are not valid excuses; submit early

---

## Troubleshooting

### Common Issues and Solutions

#### 1. **PDF Export Issues**
**Problem:** Figures missing or formatting broken in PDF

**Solution:**
- Embed all fonts when exporting
- Use "Print to PDF" rather than "Save as PDF"
- Check PDF before submitting using Adobe Reader

#### 2. **Code Won't Run**
**Problem:** Instructor cannot execute your code

**Solution:**
- Test in a fresh virtual environment
- Use relative paths, not absolute paths
- Document exact Python version used
- Include all configuration files

#### 3. **File Size Too Large**
**Problem:** Dataset or submission exceeds upload limit

**Solution:**
- Compress files using ZIP or tar.gz
- Use `.gitignore` to exclude unnecessary files
- Consider hosting large datasets on Google Drive/Dropbox with link
- Remove redundant data or models

#### 4. **Import Errors**
**Problem:** Missing dependencies when running code

**Solution:**
```bash
# Generate requirements.txt from current environment
pip freeze > requirements.txt

# Or manually specify versions
pip list --format=freeze > requirements.txt
```

#### 5. **Dataset Loading Errors**
**Problem:** Path errors when loading data

**Solution:**
```python
import os
from pathlib import Path

# Use relative paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'dataset.csv'

# Or use os.path
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'processed', 'dataset.csv')
```

#### 6. **IEEE Format Confusion**
**Problem:** Unsure about formatting details

**Solution:**
- Download official IEEE template
- Review sample papers from IEEE Xplore
- Use LaTeX for precise formatting control
- Consult IEEE Author Digital Toolbox

---

## Submission Platform Instructions

### Via Course Portal

1. Log in to the course management system
2. Navigate to "Assignments" → "CSC 108 Final Project"
3. Upload files in the designated sections:
   - **Paper Submission:** Upload PDF
   - **Code Submission:** Upload ZIP file
   - **Dataset Submission:** Upload ZIP or provide link
4. Verify uploads are complete
5. Submit for grading
6. Check for confirmation email

### Via GitHub (If Allowed)

1. Ensure your repository is properly organized
2. Tag your final submission:
   ```bash
   git tag -a v1.0 -m "CSC 108 Final Submission"
   git push origin v1.0
   ```
3. Make repository public or grant instructor access
4. Submit repository URL through course portal

---

## Academic Integrity

### Guidelines

- **Cite all sources** properly in IEEE format
- **Original work only:** Do not copy code or text from others
- **Collaboration:** Acknowledge any assistance received
- **AI Tools:** If allowed, disclose use of AI assistants (ChatGPT, Copilot)
- **Dataset usage:** Ensure you have rights to use the data

### Plagiarism

Plagiarism includes:
- Copying code without attribution
- Paraphrasing papers without citation
- Submitting someone else's work as your own
- Reusing your own previous work without permission

**Consequences:** Academic penalties per university policy

---

## Getting Help

### Resources

- **Instructor Office Hours:** [Times and Location]
- **Teaching Assistants:** [Contact Information]
- **Course Forum:** [Link to discussion board]
- **Technical Support:** [IT help desk contact]

### Questions to Ask Before the Deadline

- "Can you review my paper structure?"
- "Is my code organization acceptable?"
- "Are my dataset files in the correct format?"
- "Do I need to include X in my submission?"

---

## Final Tips for Success

1. **Start Early:** Don't wait until the last minute
2. **Test Everything:** Run code in a clean environment
3. **Backup Your Work:** Use version control (Git)
4. **Read Instructions Carefully:** Follow all requirements
5. **Proofread:** Check for typos and errors
6. **Organize Files:** Maintain a clean directory structure
7. **Document Thoroughly:** Future you will thank present you
8. **Ask Questions:** Clarify doubts before submitting
9. **Submit Early:** Avoid last-minute technical issues
10. **Keep Copies:** Maintain backups of all submissions

---

## Example File Naming Convention

```
Paper:    CSC108_Smith_John_Paper.pdf
Code:     CSC108_Smith_John_Code.zip
Dataset:  CSC108_Smith_John_Data.zip
```

Replace "Smith_John" with your actual last name and first name.

---

## Contact Information

**Course Instructor:** [Name]
- Email: [instructor@university.edu]
- Office: [Building and Room Number]
- Office Hours: [Days and Times]

**Teaching Assistants:** [Names]
- Email: [ta@university.edu]
- Lab Hours: [Days and Times]

---

**Good luck with your submission! 🎓**

*This guide is subject to updates. Check regularly for any changes.*

---

*Last Revision: January 7, 2026*
