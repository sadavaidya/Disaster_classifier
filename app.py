import streamlit as st
import joblib

# Load the trained model
model = joblib.load('models\logistic_regression_model.pkl')

def predict_category(text):
    """Predict the category of the given text using the trained model."""
    prediction = model.predict([text])
    return prediction[0]

# Streamlit UI
st.title('Disaster Message Classifier')
st.write('Enter a message to classify its category.')

user_input = st.text_area('Message', '')

if st.button('Classify'):
    if user_input:
        category = predict_category(user_input)
        st.write(f'The message is classified as: **{category}**')
    else:
        st.write('Please enter a message to classify.')