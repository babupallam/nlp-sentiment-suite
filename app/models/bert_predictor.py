import os
import joblib
import numpy as np
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast

# Adjust paths to properly load the files from the existing directory structure
current_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.abspath(os.path.join(current_dir, "../../models/distilbert_model/"))
label_mapping_path = os.path.abspath(os.path.join(current_dir, "../../models/distilbert_label_mapping.pkl"))

# Check if required files exist
if not os.path.exists(model_save_path):
    raise FileNotFoundError(f"BERT model directory not found at {model_save_path}")
if not os.path.exists(label_mapping_path):
    raise FileNotFoundError(f"Label mapping file not found at {label_mapping_path}")

# Load the DistilBERT model using Hugging Face Transformers
try:
    bert_model = TFDistilBertForSequenceClassification.from_pretrained(model_save_path)
    print("[DEBUG] Model loaded successfully from Hugging Face API.")
except Exception as e:
    print(f"[ERROR] Failed to load model from Hugging Face API: {e}")
    raise

# Load tokenizer from the model directory
tokenizer = DistilBertTokenizerFast.from_pretrained(model_save_path)

# Load label mapping
label_mapping = joblib.load(label_mapping_path)

# Function to predict sentiment using BERT model
def predict_bert(text):
    """
    Function to predict sentiment for given input text using the DistilBERT model.
    """
    try:
        # Preprocess text using the tokenizer
        encodings = tokenizer(
            text,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="tf"
        )

        # Predict using the DistilBERT model
        y_probs = bert_model.predict(
            {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]}
        ).logits
        y_pred = np.argmax(y_probs, axis=1)[0]

        # Convert the prediction to the corresponding label
        predicted_label = list(label_mapping.keys())[list(label_mapping.values()).index(y_pred)]
        return predicted_label

    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Example usage
if __name__ == "__main__":
    sample_text = "I am very happy with the product quality!"
    prediction = predict_bert(sample_text)
    print(f"Predicted Sentiment: {prediction}")
