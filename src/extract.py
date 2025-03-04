import pandas as pd
import logging
from src.config import TRAIN_PATH, TEST_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        logging.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data with shape {data.shape}")
        return data
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f"File is empty: {file_path}")
        raise e
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        raise e


if __name__ == "__main__":
    train_data = load_data(TRAIN_PATH)
    test_data = load_data(TEST_PATH)
