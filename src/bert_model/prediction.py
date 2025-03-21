import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define model path
model_path = "models/bert_model"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the trained model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_disaster(text):
    model.eval()  # Set model to evaluation mode

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get prediction probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    # Get predicted label
    predicted_label = torch.argmax(probabilities, dim=1).item()

    return predicted_label, probabilities.tolist()

# Test with an example
text = "Hurricane is coming! Evacuate now!"
predicted_label, probabilities = predict_disaster(text)
print(f"Predicted Label: {predicted_label}, Probabilities: {probabilities}")
