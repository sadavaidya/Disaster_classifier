import pandas as pd
import sqlite3
import logging
from src.config import CLEANED_TRAIN_PATH, CLEANED_TEST_PATH, OUTPUT_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_PATH = f"{OUTPUT_DIR}/disaster_tweets.db"

def save_to_database(df: pd.DataFrame, table_name: str, db_path: str = DB_PATH):
    """
    Save DataFrame to SQLite database.

    Args:
        df (pd.DataFrame): DataFrame to save.
        table_name (str): Table name in the database.
        db_path (str): Path to the SQLite database.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logging.info(f"Data saved to {table_name} table in {db_path}")
    except Exception as e:
        logging.error(f"Failed to save data to database: {e}")
        raise e


if __name__ == "__main__":
    train_data = pd.read_csv(CLEANED_TRAIN_PATH)
    test_data = pd.read_csv(CLEANED_TEST_PATH)

    save_to_database(train_data, 'cleaned_train')
    save_to_database(test_data, 'cleaned_test')
