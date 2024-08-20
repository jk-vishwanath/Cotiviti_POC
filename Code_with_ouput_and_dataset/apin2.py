import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the dataset
file_path = r'C:\Users\JKVish\api_3\icd.csv'  # Replace with your correct file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Add this line to download omw-1.4

# Text preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the 'description' column
data['processed_description'] = data['description'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['processed_description'])

# Train KNN model
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X)

# Function to get the closest N CCSS Ids
def get_closest_ccss_ids(input_text, N):
    input_text_processed = preprocess_text(input_text)
    input_vector = vectorizer.transform([input_text_processed])
    distances, indices = knn.kneighbors(input_vector, n_neighbors=N)
    closest_ids = data.iloc[indices[0]]['id'].values
    return closest_ids

@app.route('/get_closest_ids', methods=['POST'])
def get_closest_ids():
    try:
        data = request.get_json()
        input_text = data.get('text')
        N = data.get('N')
        if not input_text or not N:
            return jsonify({'error': 'Invalid input'}), 400
        
        try:
            N = int(N)
        except ValueError:
            return jsonify({'error': 'N must be an integer'}), 400

        closest_ccss_ids = get_closest_ccss_ids(input_text, N)
        return jsonify({'closest_ids': closest_ccss_ids.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
