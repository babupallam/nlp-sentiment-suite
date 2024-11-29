from flask import Flask, render_template, request, jsonify
import joblib
from models.lstm_predictor import predict_lstm
from models.bert_predictor import predict_bert
from models.logistic_predictor import predict_logistic
from models.naive_bayes_predictor import predict_naive_bayes
from models.svm_predictor import predict_svm

app = Flask(__name__)

@app.route('/')
def index():
    models = ["Logistic Regression", "LSTM", "BERT", "Naive Bayes", "SVM"]
    return render_template('index.html', models=models)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']
        model = request.form['model']

        if model == 'logistic_regression':
            sentiment = predict_logistic(text)
        elif model == 'lstm':
            sentiment = predict_lstm(text)
        elif model == 'bert':
            sentiment = predict_bert(text)
        elif model == 'naive_bayes':
            sentiment = predict_naive_bayes(text)
        elif model == 'svm':
            sentiment = predict_svm(text)
        else:
            raise ValueError("Invalid Model Selected")

        return jsonify({'sentiment': str(sentiment)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
