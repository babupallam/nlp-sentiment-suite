import joblib
import os

# Define paths

current_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_dir, "../../models/")
naive_bayes_model_path = os.path.join(model_save_path, "Naive_Bayes.pkl")
tfidf_vectorizer_path = os.path.join(model_save_path, "tfidf_vectorizer.pkl")

# Load Naive Bayes model and vectorizer
naive_bayes_model = joblib.load(naive_bayes_model_path)
vectorizer = joblib.load(tfidf_vectorizer_path)


# Function to predict sentiment using Naive Bayes model
def predict_naive_bayes(text):
    """
    Function to predict sentiment for given input text using the Naive Bayes model.
    """
    # Transform text using TF-IDF vectorizer
    text_tfidf = vectorizer.transform([text])

    # Make prediction
    prediction = naive_bayes_model.predict(text_tfidf)[0]

    return prediction
