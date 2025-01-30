# Fake-News-Detection
This project is a fake news detection system that uses a machine learning model to classify news articles as fake or real. The project structure indicates the following key components.
Model Training and Data:

model.ipynb: A Jupyter notebook likely used for training the fake news detection model.
fake_news_model.pkl: The trained machine learning model saved as a pickle file.
fake_or_real_news.csv, preprocessed_fake_news_dataset.csv, resampled_fake_news_dataset.csv: Datasets used for training and preprocessing the model.
tfidf_vectorizer.pkl: A TF-IDF vectorizer used for transforming text data into numerical features for the model.
Web Deployment:

app.py: A Flask application that serves as the web interface for the fake news detection system.
Templates: Directory containing HTML templates for the web application, including index.html and results.html.
Environment and Dependencies:

requirements.txt: Lists the Python dependencies required for the project.
venv: A virtual environment containing the necessary packages and executables for running the project.
The project workflow involves training the model using the datasets and Jupyter notebook, saving the trained model and vectorizer, and deploying the model using a Flask web application to classify news articles as fake or real.

