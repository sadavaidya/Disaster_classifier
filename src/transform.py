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


def validate_data(df: pd.DataFrame, expected_columns: list):
    """
    Validate data for schema, null values, duplicates, and target column integrity.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        expected_columns (list): List of expected columns.
    """
    # Check if expected columns exist
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        raise ValueError(f"Missing columns: {missing_columns}")

    # Check for null values
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count.any():
            logging.warning(f"Column '{col}' has {null_count} missing values.")
    # null_counts = df.isnull().sum()
    # if null_counts.any():
    #     logging.warning(f"Null values found:\n{null_counts}")

    # Remove duplicates
    before_dedup = df.shape[0]
    df.drop_duplicates(inplace=True)
    after_dedup = df.shape[0]
    logging.info(f"Removed {before_dedup - after_dedup} duplicate rows")

    # Validate target column if it exists
    if 'target' in df.columns:
        invalid_targets = df[~df['target'].isin([0, 1])]
        if not invalid_targets.empty:
            logging.error(f"Invalid target values found:\n{invalid_targets}")
            raise ValueError("Invalid target values found")


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
    Apply data validation and text cleaning on a specific column of a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the column containing text data.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    expected_columns = ['id', text_column, 'target'] if 'target' in df.columns else ['id', text_column]
    validate_data(df, expected_columns)

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
