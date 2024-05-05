from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model using pickle
with open('model_pickle', 'rb') as f:
    model = pickle.load(f)
with open('tfidf','rb') as f:
    tfidf_vectorizer=pickle.load(f)

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
#tfidf_vectorizer = TfidfVectorizer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'\W+', ' ', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize tokens
    preprocessed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # Join preprocessed tokens into a string
    preprocessed_text = ' '.join(preprocessed_tokens)
    return preprocessed_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        # Get the text input from the request
        text = request.json['text']
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)
        # Vectorize the preprocessed text
        vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
        # Predict sentiment using the loaded model
        sentiment = model.predict(vectorized_text)[0]
        # Return the sentiment analysis result as JSON
        return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
