import joblib
import os

# Define paths

current_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_dir, "../../models/")
svm_model_path = os.path.join(model_save_path, "SVM.pkl")
tfidf_vectorizer_path = os.path.join(model_save_path, "tfidf_vectorizer.pkl")

# Load SVM model and vectorizer
svm_model = joblib.load(svm_model_path)
vectorizer = joblib.load(tfidf_vectorizer_path)


# Function to predict sentiment using SVM model
def predict_svm(text):
    """
    Function to predict sentiment for given input text using the SVM model.
    """
    # Transform text using TF-IDF vectorizer
    text_tfidf = vectorizer.transform([text])

    # Make prediction
    prediction = svm_model.predict(text_tfidf)[0]

    return prediction
