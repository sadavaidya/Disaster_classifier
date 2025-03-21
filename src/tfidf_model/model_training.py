import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple models.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        X_test: Test feature matrix.
        y_test: Test target labels.

    Returns:
        dict: Trained models.
        dict: Model evaluation results.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC()
    }

    results = {}

    for name, model in models.items():
        logging.info(f"Training {name}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[name] = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions)
        }

        logging.info(f"{name} Performance: {results[name]}")

        # Saving the model 
        model_path = f"models/{name.replace(' ', '_').lower()}_model.pkl"
        try:
            joblib.dump(model, model_path)
            logging.info(f"{name} saved at {model_path}")
        except Exception as e:
            logging.error(f"Error saving {name}: {e}")


    return models, results