import logging
from sklearn.model_selection import train_test_split
from src.extract import load_data
from src.transform import transform_data
from src.load import save_to_database
from src.tfidf_model.feature_engineering import load_cleaned_data, apply_tfidf
from src.tfidf_model.model_training import train_and_evaluate
from src.tfidf_model.model_evaluation import compare_models
from src.config import TRAIN_PATH, TEST_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting ETL and ML pipeline")

    # Extract
    train_data = load_data(TRAIN_PATH)
    test_data = load_data(TEST_PATH)

    # Transform
    train_data = transform_data(train_data, 'text')
    test_data = transform_data(test_data, 'text')

    # Load
    save_to_database(train_data, 'cleaned_train')
    save_to_database(test_data, 'cleaned_test')

    # Feature Engineering
    train_data = load_cleaned_data('cleaned_train')
    X, y = apply_tfidf(train_data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    models, results = train_and_evaluate(X_train, y_train, X_test, y_test)

    # Compare models
    compare_models(results)

    logging.info("Compared models using plots")

    logging.info("Pipeline execution complete")

if __name__ == "__main__":
    main()
