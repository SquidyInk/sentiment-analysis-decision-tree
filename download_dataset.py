"""
Kaggle Dataset Downloader for Financial News Sentiment Analysis
This script automatically downloads the dataset from Kaggle and places it in the correct folder.

Requirements:
- Kaggle account
- Kaggle API credentials (kaggle.json)

Setup Instructions:
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)
5. Run this script: python download_dataset.py
"""

import os
import sys
import zipfile
from pathlib import Path

def check_kaggle_installed():
    """Check if kaggle package is installed"""
    try:
        import kaggle
        return True
    except ImportError:
        return False

def install_kaggle():
    """Install kaggle package"""
    print("Installing kaggle package...")
    os.system(f"{sys.executable} -m pip install kaggle")

def download_dataset():
    """Download the Financial News Sentiment dataset from Kaggle"""
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("=" * 60)
        print("Financial News Sentiment Dataset Downloader")
        print("=" * 60)
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        print("\n✓ Kaggle API authenticated successfully")
        print(f"✓ Data directory created: {data_dir.absolute()}")
        
        # Dataset information
        dataset_name = "ankurzing/sentiment-analysis-for-financial-news"
        
        print(f"\n📥 Downloading dataset: {dataset_name}")
        print("This may take a few minutes depending on your internet speed...")
        
        # Download dataset
        api.dataset_download_files(
            dataset_name,
            path=str(data_dir),
            unzip=True
        )
        
        print("\n✓ Dataset downloaded successfully!")
        
        # List downloaded files
        print(f"\n📁 Files in {data_dir}:")
        for file in data_dir.iterdir():
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({size_mb:.2f} MB)")
        
        print("\n" + "=" * 60)
        print("✅ DOWNLOAD COMPLETE!")
        print("=" * 60)
        print("\nYour dataset is ready in the 'data/' folder.")
        print("You can now run: python main.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a Kaggle account")
        print("2. Check that kaggle.json is in the correct location:")
        print("   - Linux/Mac: ~/.kaggle/kaggle.json")
        print("   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print("3. Verify your Kaggle API credentials are valid")
        print("4. Accept the dataset terms at:")
        print("   https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news")
        return False

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("CSC 108 - Sentiment Analysis Dataset Setup")
    print("=" * 60)
    
    # Check if kaggle is installed
    if not check_kaggle_installed():
        print("\n⚠ Kaggle package not found.")
        response = input("Would you like to install it now? (y/n): ")
        if response.lower() == 'y':
            install_kaggle()
        else:
            print("\n❌ Please install kaggle manually:")
            print("   pip install kaggle")
            sys.exit(1)
    
    # Download dataset
    success = download_dataset()
    
    if success:
        print("\n🎉 Setup complete! You're ready to start analyzing sentiment!")
    else:
        print("\n❌ Setup failed. Please follow the troubleshooting steps above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
