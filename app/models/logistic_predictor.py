import joblib
import os

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_dir, "../../models/")
logistic_model_path = os.path.join(model_save_path, "Logistic_Regression.pkl")
tfidf_vectorizer_path = os.path.join(model_save_path, "tfidf_vectorizer.pkl")

# Load Logistic Regression model and vectorizer
logistic_model = joblib.load(logistic_model_path)
vectorizer = joblib.load(tfidf_vectorizer_path)


# Function to predict sentiment using Logistic Regression model
def predict_logistic(text):
    """
    Function to predict sentiment for given input text using the Logistic Regression model.
    """
    # Transform text using TF-IDF vectorizer
    text_tfidf = vectorizer.transform([text])

    # Make prediction
    prediction = logistic_model.predict(text_tfidf)[0]

    return prediction
