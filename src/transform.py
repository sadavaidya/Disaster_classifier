import pandas as pd
import re
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from src.config import TRAIN_PATH, TEST_PATH, CLEANED_TRAIN_PATH, CLEANED_TEST_PATH, STOPWORDS_LANGUAGE

nltk.download('stopwords')
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

STOPWORDS = set(stopwords.words(STOPWORDS_LANGUAGE))


def clean_text(text: str) -> str:
    """
    Clean and normalize text data.

    Args:
        text (str): Raw text data.

    Returns:
        str: Cleaned and normalized text.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()                                    # Lowercasing
    text = re.sub(r'http\S+', '', text)                  # Remove URLs
    text = re.sub(r'<.*?>', '', text)                     # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)              # Remove special characters
    tokens = word_tokenize(text)                         # Tokenization
    cleaned_text = ' '.join([word for word in tokens if word not in STOPWORDS])
    return cleaned_text


def transform_data(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Apply text cleaning on a specific column of a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the column containing text data.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    logging.info(f"Transforming text data in column: {text_column}")
    df[text_column] = df[text_column].apply(clean_text)
    logging.info("Text data transformation complete")
    return df


if __name__ == "__main__":
    train_data = pd.read_csv(TRAIN_PATH)
    transformed_train = transform_data(train_data, 'text')
    transformed_train.to_csv(CLEANED_TRAIN_PATH, index=False)

    test_data = pd.read_csv(TEST_PATH)
    transformed_test = transform_data(test_data, 'text')
    transformed_test.to_csv(CLEANED_TEST_PATH, index=False)
