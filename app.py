from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer from files
with open('fake_news_model.pkl', 'rb') as model_file:
    grid_search = pickle.load(model_file)
    model = grid_search.best_estimator_  # Get the best model from GridSearchCV

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        try:
            text = request.form['text']
            
            if not text or not isinstance(text, str):
                return render_template('index.html', error='Invalid input. Please provide a valid text string.')
            
            text_vectorized = vectorizer.transform([text])
            prediction = model.predict(text_vectorized)[0]
            probability = model.predict_proba(text_vectorized)[0]
            
            return render_template('results.html', 
                                 prediction=prediction,
                                 fake_prob=float(probability[0]),
                                 real_prob=float(probability[1]))
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text or not isinstance(text, str):
            return jsonify({'error': 'Invalid input. Please provide a valid text string.'}), 400

        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]

        return jsonify({
            'prediction': prediction,
            'probability': {
                'FAKE': float(probability[0]),
                'REAL': float(probability[1])
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)