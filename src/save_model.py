import joblib
import logging
from src.model_evaluation import evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_best_model(models, X_test, y_test, output_path='output/best_model.pkl'):
    """
    Find and save the best-performing model based on accuracy.

    Args:
        models (dict): Dictionary of model names and trained models.
        X_test: Test feature matrix.
        y_test: True labels.
        output_path (str): Path to save the best model.
    """
    best_model = None
    best_model_name = ""
    best_accuracy = 0

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        accuracy = metrics['accuracy']
        logging.info(f"{name} Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    if best_model:
        try:
            joblib.dump(best_model, output_path)
            logging.info(f"Best model ({best_model_name}) saved with accuracy: {best_accuracy}")
        except Exception as e:
            logging.error(f"Error saving best model: {e}")
            raise e
    else:
        logging.warning("No model was selected as the best model.")

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from src.feature_engineering import load_cleaned_data, apply_tfidf

    # Dummy data loading and vectorization
    train_data = load_cleaned_data('cleaned_train')
    X, y = apply_tfidf(train_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dummy models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC()
    }

    # Train models
    for name, model in models.items():
        model.fit(X_train, y_train)

    # Find and save best model
    save_best_model(models, X_test, y_test)
