# setup.py

import nltk

def download_nltk_data():
    nltk.download('vader_lexicon')
    nltk.download('punkt')

if __name__ == "__main__":
    download_nltk_data()
