import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from src.feature_engineering import load_cleaned_data, apply_tfidf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_and_evaluate(X, y):
    """
    Train and evaluate multiple models.

    Args:
        X: Feature matrix.
        y: Target labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC()
    }

    for name, model in models.items():
        logging.info(f"Training {name}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions)
        logging.info(f"\n{name} Performance:\n{report}")


if __name__ == "__main__":
    train_data = load_cleaned_data('cleaned_train')
    X, y = apply_tfidf(train_data)
    train_and_evaluate(X, y)
