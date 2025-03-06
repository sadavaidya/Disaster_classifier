import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from src.config import DB_PATH
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_cleaned_data(table_name: str) -> pd.DataFrame:
    """
    Load cleaned data from the SQLite database.

    Args:
        table_name (str): Table name in the database.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            data = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            logging.info(f"Loaded data from {table_name} with shape {data.shape}")
            return data
    except Exception as e:
        logging.error(f"Failed to load data from database: {e}")
        raise e


def apply_tfidf(df: pd.DataFrame, text_column: str = 'text'):
    """
    Apply TF-IDF vectorization on the text column.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Column containing the text data.

    Returns:
        X (sparse matrix): TF-IDF feature matrix.
        y (pd.Series): Target labels.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df[text_column])
        y = df['target']
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
        logging.info("TF-IDF vectorizer saved successfully.")
        logging.info(f"TF-IDF applied, resulting shape: {X.shape}")
        return X, y
    except Exception as e:
        logging.error(f"Failed to apply TF-IDF: {e}")
        raise e


if __name__ == "__main__":
    train_data = load_cleaned_data('cleaned_train')
    X, y = apply_tfidf(train_data)
    print(X, y)
