import pandas as pd
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split



# Load Test Dataset
df = pd.read_csv("output/cleaned_train.csv")

# Split again (80% train, 20% test)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),  # Change to "text" if available
    df["target"].tolist(),
    test_size=0.2, random_state=42
)

# ------------------ Load TF-IDF Model ------------------
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
logistic_model = joblib.load("models/logistic_regression_model.pkl")

# Convert text into TF-IDF features
X_test_tfidf = tfidf_vectorizer.transform(test_texts)

# Make predictions
tfidf_preds = logistic_model.predict(X_test_tfidf)

# ------------------ Load Fine-tuned BERT Model ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model_path = "models/bert_model"
tokenizer = BertTokenizer.from_pretrained(bert_model_path)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path).to(device)
bert_model.eval()

# Tokenize test data
inputs = tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

# Get BERT model predictions
with torch.no_grad():
    outputs = bert_model(**inputs)
    bert_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

# ------------------ Evaluation Metrics ------------------
def evaluate(true_labels, preds, model_name):
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds)
    rec = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    print(f"--- {model_name} ---")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
    return [acc, prec, rec, f1]

# Get evaluation results
tfidf_results = evaluate(test_labels, tfidf_preds, "TF-IDF + Logistic Regression")
bert_results = evaluate(test_labels, bert_preds, "Fine-tuned BERT")

# ------------------ Plot Results ------------------
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
tfidf_values = tfidf_results
bert_values = bert_results

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, tfidf_values, width, label="TF-IDF + Logistic Regression")
rects2 = ax.bar(x + width/2, bert_values, width, label="Fine-tuned BERT")

ax.set_ylabel("Score")
ax.set_title("Model Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.ylim(0, 1)
plt.savefig("output/model_comparison.png")
plt.show()
