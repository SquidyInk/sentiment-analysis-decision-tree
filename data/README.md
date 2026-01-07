# Financial News Sentiment Dataset

## Overview

This directory contains the **Financial News Sentiment Analysis** dataset from Kaggle, which is used for training and evaluating the sentiment analysis decision tree model.

**Dataset Source**: [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

## Dataset Details

- **Total Sentences**: 4,846
- **Classes**: 3 sentiment categories
  - **Positive**: Optimistic or favorable financial news
  - **Neutral**: Objective or balanced financial news
  - **Negative**: Pessimistic or unfavorable financial news
- **Domain**: Financial news articles and statements
- **Format**: CSV file
- **Language**: English

## How to Download

### Prerequisites
- A Kaggle account (free to create at [kaggle.com](https://www.kaggle.com))
- Kaggle API credentials (optional, for CLI download)

### Option 1: Manual Download (Recommended for Beginners)

1. Visit the dataset page: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
2. Click the **Download** button (you may need to sign in to Kaggle)
3. Extract the downloaded ZIP file
4. Copy the CSV file(s) to this `data/` directory

### Option 2: Using Kaggle API

1. Install the Kaggle API:
   ```bash
   pip install kaggle
   ```

2. Set up your Kaggle API credentials:
   - Go to your Kaggle account settings: https://www.kaggle.com/settings
   - Scroll to the "API" section and click "Create New API Token"
   - This downloads a `kaggle.json` file
   - Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<Username>\.kaggle\kaggle.json` (Windows)
   - Set permissions (Linux/Mac only): `chmod 600 ~/.kaggle/kaggle.json`

3. Download the dataset:
   ```bash
   kaggle datasets download -d ankurzing/sentiment-analysis-for-financial-news
   ```

4. Extract and move to the data directory:
   ```bash
   unzip sentiment-analysis-for-financial-news.zip -d data/
   ```

## File Structure

After downloading, your `data/` directory should contain:

```
data/
├── README.md (this file)
├── all-data.csv (or similar - the main dataset file)
└── [any additional files from the dataset]
```

**Note**: The exact filename may vary. Common names include:
- `all-data.csv`
- `financial_news_sentiment.csv`
- `data.csv`

Please verify the actual filename after downloading and update your code accordingly.

## Dataset Format

The dataset typically contains the following columns:

- **Sentiment**: The sentiment label (positive, neutral, negative)
- **News/Text**: The financial news sentence or statement

Example:
```csv
Sentiment,News
neutral,"According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing ."
positive,"Technopolis plans to develop in stages an area of no less than 100,000 square meters in order to host companies working in computer technologies and telecommunications , the statement said ."
negative,"The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility , as reported by the company's spokesman ."
```

## Usage in Project

To load the dataset in your Python code:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/all-data.csv', encoding='latin-1', names=['Sentiment', 'News'])

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"\nSentiment distribution:\n{df['Sentiment'].value_counts()}")
```

## Citation

If you use this dataset in your research or project, please cite:

```
Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).
Good debt or bad debt: Detecting semantic orientations in economic texts.
Journal of the Association for Information Science and Technology, 65(4), 782-796.
```

**Kaggle Dataset**: Ankur Zing - Sentiment Analysis for Financial News
https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

## Troubleshooting

### Issue: "File not found" error

**Solution**: 
- Verify the dataset file is in the `data/` directory
- Check the exact filename matches what's used in your code
- Ensure the file path is correct relative to your script location

### Issue: Encoding errors when loading CSV

**Solution**:
```python
# Try different encodings
df = pd.read_csv('data/all-data.csv', encoding='latin-1')
# or
df = pd.read_csv('data/all-data.csv', encoding='utf-8')
# or
df = pd.read_csv('data/all-data.csv', encoding='iso-8859-1')
```

### Issue: Kaggle API authentication fails

**Solution**:
- Ensure `kaggle.json` is in the correct location
- Verify file permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)
- Check that your Kaggle API token is valid and not expired

### Issue: "403 Forbidden" when downloading

**Solution**:
- Make sure you're logged into Kaggle
- Accept the dataset's terms and conditions on the Kaggle website
- Verify your Kaggle account is verified (email confirmation)

### Issue: Dataset structure different than expected

**Solution**:
- Check the column names after loading: `print(df.columns)`
- Inspect the first few rows: `print(df.head())`
- Adjust your code to match the actual column names and structure

### Issue: Imbalanced classes affecting model performance

**Solution**:
- Use stratified train-test split to maintain class distribution
- Consider class weighting in your model
- Apply oversampling (SMOTE) or undersampling techniques if needed

## License

Please refer to the dataset page on Kaggle for license information and usage terms:
https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

## Additional Resources

- [Kaggle Dataset Page](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- [Original Research Paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.23062)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)

---

**Last Updated**: 2026-01-07

For questions or issues related to this dataset, please open an issue in the repository or refer to the Kaggle dataset discussion section.
