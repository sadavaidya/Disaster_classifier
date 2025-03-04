import os

# Data paths
# BASE_DIR = "disaster_classifier"
# DATA_DIR = os.path.join(BASE_DIR, "data")
# OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DATA_DIR = "data"
OUTPUT_DIR = "output"

# File names
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
CLEANED_TRAIN_FILE = "cleaned_train.csv"
CLEANED_TEST_FILE = "cleaned_test.csv"

# Full file paths
TRAIN_PATH = os.path.join(DATA_DIR, TRAIN_FILE)
TEST_PATH = os.path.join(DATA_DIR, TEST_FILE)
CLEANED_TRAIN_PATH = os.path.join(OUTPUT_DIR, CLEANED_TRAIN_FILE)
CLEANED_TEST_PATH = os.path.join(OUTPUT_DIR, CLEANED_TEST_FILE)

# NLP config
STOPWORDS_LANGUAGE = 'english'

# Logging config
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
