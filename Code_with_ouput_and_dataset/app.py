
import streamlit as st
import pickle
import nltk
import requests
import os
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd

import streamlit as st
import pickle
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure the app looks in the local nltk_data folder
nltk.data.path.append('https://github.com/jk-vishwanath/CCSS/tree/main/nltk_data')

# Download necessary NLTK resources (only if not already downloaded)
nltk.download('punkt', download_dir='https://github.com/jk-vishwanath/CCSS/tree/main/nltk_data')
nltk.download('stopwords', download_dir='https://github.com/jk-vishwanath/CCSS/tree/main/nltk_data')
nltk.download('wordnet', download_dir='https://github.com/jk-vishwanath/CCSS/tree/main/nltk_data')
nltk.download('punkt_tab', download_dir='https://github.com/jk-vishwanath/CCSS/tree/main/nltk_data')


# Path to the pickle file
pickle_url = "https://github.com/jk-vishwanath/Cotiviti_POC/raw/main/Code_with_ouput_and_dataset/model.pkl"

# Function to load the pickle file
def load_model(pickle_url):
    response = requests.get(pickle_url)
    if response.status_code == 200:
        pickle_data = BytesIO(response.content)
        return pickle.load(pickle_data)
    else:
        st.error("Failed to load the model. Check the URL or your internet connection.")
        return None

# Load the model
export_data = load_model(pickle_url)
if export_data:
    vectorizer = export_data['vectorizer']
    knn = export_data['knn_model']
    data = export_data['dataset']

# Text preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Predict function
def get_closest_icd_codes(input_text, N):
    input_text_processed = preprocess_text(input_text)
    input_vector = vectorizer.transform([input_text_processed])
    distances, indices = knn.kneighbors(input_vector, n_neighbors=N)
    closest_ids = data.iloc[indices[0]]['id'].values
    return closest_ids

# Streamlit app
st.title("ICD Code Predictor")
st.write("Enter symptoms to predict the closest ICD codes.")

# Input fields
input_text = st.text_area("Enter symptoms here:")
num_codes = st.number_input("Number of ICD codes to retrieve:", min_value=1, max_value=10, step=1, value=5)

if st.button("Predict"):
    if input_text.strip():
        try:
            closest_codes = get_closest_icd_codes(input_text, num_codes)
            st.write("### Predicted ICD Codes:")
            for code in closest_codes:
                st.write(f"- {code}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter symptoms to predict.")
