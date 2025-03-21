import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split



# Load the dataset
df = pd.read_csv("output/cleaned_train.csv")

# Split again (80% train, 20% test)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),  # Change to "text" if available
    df["target"].tolist(),
    test_size=0.2, random_state=42
)

print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")


# Define model path
model_path = "models/bert_model"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the trained model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def evaluate_model(test_texts, test_labels):
    model.eval()  # Set model to evaluation mode
    predictions = []
    true_labels = test_labels

    for text in tqdm(test_texts, desc="Evaluating"):
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get prediction
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).cpu().item()
        predictions.append(predicted_label)

    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="binary")
    recall = recall_score(true_labels, predictions, average="binary")
    f1 = f1_score(true_labels, predictions, average="binary")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1


evaluate_model(test_texts, test_labels)
