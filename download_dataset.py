"""
Download Twitter Sentiment140 Dataset

This script downloads the Twitter Sentiment140 dataset from Hugging Face.
Note: This is a large dataset (approximately 238MB) containing 1.6 million tweets.

The dataset contains tweets labeled with sentiment:
- 0 = negative
- 4 = positive (will be converted to 1)

Original dataset: https://huggingface.co/datasets/kazanova/sentiment140
"""

from datasets import load_dataset
import os

def download_twitter_sentiment140():
    """
    Download the Twitter Sentiment140 dataset from Hugging Face.
    
    This is a large dataset (~238MB) with 1.6 million tweets labeled for sentiment analysis.
    The download may take several minutes depending on your internet connection.
    """
    print("Downloading Twitter Sentiment140 dataset...")
    print("Note: This is a large dataset (238MB) with 1.6 million tweets.")
    print("Download may take a few minutes...")
    
    # Dataset name on Hugging Face
    dataset_name = "kazanova/sentiment140"
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset(dataset_name)
        
        print(f"\nDataset '{dataset_name}' downloaded successfully!")
        print(f"Dataset info: {dataset}")
        
        # Get the cache directory where the dataset is stored
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
        print(f"\nDataset cached in: {cache_dir}")
        
        # Check if the original CSV file exists and rename it
        sentiment140_dir = os.path.join(cache_dir, "kazanova___sentiment140")
        if os.path.exists(sentiment140_dir):
            # Search for the original CSV file
            for root, dirs, files in os.walk(sentiment140_dir):
                if "training.1600000.processed.noemoticon.csv" in files:
                    old_path = os.path.join(root, "training.1600000.processed.noemoticon.csv")
                    new_path = os.path.join(root, "twitter_sentiment140.csv")
                    
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"\nRenamed dataset file to: twitter_sentiment140.csv")
                    break
        
        return dataset
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return None

if __name__ == "__main__":
    dataset = download_twitter_sentiment140()
    
    if dataset:
        print("\n" + "="*50)
        print("Twitter Sentiment140 Dataset Download Complete!")
        print("="*50)
        print("\nDataset contains 1.6 million tweets for sentiment analysis.")
        print("You can now proceed with data preprocessing and model training.")
