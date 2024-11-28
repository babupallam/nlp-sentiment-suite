import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off oneDNN optimizations

# Import TensorFlow after setting environment variables
import tensorflow as tf

# Other imports
from flask import Flask, render_template, request, jsonify
import numpy as np

# Import model prediction functions
from models.lstm_predictor import predict_lstm
from models.bert_predictor import predict_bert
from models.logistic_predictor import predict_logistic
from models.naive_bayes_predictor import predict_naive_bayes
from models.svm_predictor import predict_svm

# Initialize Flask app
app = Flask(__name__)

# Prediction logic
def predict_sentiment(model_name, text):
    try:
        if model_name == "Logistic Regression":
            return predict_logistic(text)

        elif model_name == "Naive Bayes":
            return predict_naive_bayes(text)

        elif model_name == "SVM":
            return predict_svm(text)

        elif model_name == "LSTM":
            return predict_lstm(text)

        elif model_name == "BERT":
            return predict_bert(text)

        else:
            return "Invalid Model Selected"

    except Exception as e:
        return f"Error: {str(e)}"

# Flask routes
@app.route('/')
def home():
    models = ["Logistic Regression", "Naive Bayes", "SVM", "LSTM", "BERT"]
    return render_template("index.html", models=models)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    text = data.get("text")
    model_name = data.get("model")
    if not text or not model_name:
        return jsonify({"error": "Please provide both text and model selection."})

    sentiment = predict_sentiment(model_name, text)
    if "Error" in sentiment:
        return jsonify({"error": sentiment})

    return jsonify({"sentiment": sentiment})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)