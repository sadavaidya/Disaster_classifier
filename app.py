import streamlit as st
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load('models\logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')  # Load the saved TF-IDF vectorizer

def predict_category(text):
    """Transform text using TF-IDF and predict the category."""
    text_tfidf = tfidf_vectorizer.transform([text])  # Convert text to TF-IDF features
    prediction = model.predict(text_tfidf)
    return prediction[0]

# Streamlit UI
st.title('Disaster Message Classifier')
st.write('Enter a message to classify its category.')

user_input = st.text_area('Message', '')

if st.button('Classify'):
    if user_input:
        category = predict_category(user_input)
        if category == 1:
            category_text = "Real Disaster"
        else:
            category_text = "Fake Disaster"
        st.write(f'The message is classified as: **{category_text}**')
    else:
        st.write('Please enter a message to classify.')
