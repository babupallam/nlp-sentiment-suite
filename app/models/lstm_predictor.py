import os
import joblib
import torch
import torch.nn as nn
import numpy as np

# Define paths for the models and tokenizer based on the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.abspath(os.path.join(current_dir, "../../models"))
lstm_model_path = os.path.join(model_save_path, "lstm_model.pth")
tokenizer_path = os.path.join(model_save_path, "tokenizer.pkl")
label_encoder_path = os.path.join(model_save_path, "label_encoder.pkl")

# Load the tokenizer and label encoder
tokenizer = joblib.load(tokenizer_path)
label_encoder = joblib.load(label_encoder_path)

# Define the LSTM model architecture
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, lstm_units=64, output_dim=3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(lstm_units * 2, 32)  # Bidirectional LSTM output is doubled
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Take output of last time step
        fc1_out = torch.relu(self.fc1(lstm_out))
        out = self.fc2(fc1_out)
        return out

# Initialize LSTM model and load state dictionary
vocab_size = len(tokenizer.word_index) + 1  # Ensure the vocabulary size matches the one used during training
lstm_model = LSTMClassifier(input_dim=vocab_size)
lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=torch.device('cpu')))
lstm_model.eval()  # Set the model to evaluation mode

def predict_lstm(text):
    """
    Function to predict sentiment using LSTM model.
    """
    try:
        # Preprocess the text: tokenize and pad the input sequence
        seq = tokenizer.texts_to_sequences([text])

        # Pad the sequence to match the model's input requirements (max length 100)
        padded_seq = torch.zeros((1, 100), dtype=torch.long)
        if len(seq[0]) > 0:
            seq_tensor = torch.tensor(seq[0][:100], dtype=torch.long)
            padded_seq[0, :len(seq_tensor)] = seq_tensor

        # Run the model to get predictions
        with torch.no_grad():
            output = lstm_model(padded_seq)

        # Ensure output is a torch tensor and convert to numpy
        if isinstance(output, torch.Tensor):
            y_probs = output.numpy()  # Convert tensor to NumPy array

        # Apply argmax on the NumPy array to get the predicted class index
        y_pred = np.argmax(y_probs, axis=1)

        # Since `y_pred` is a NumPy array of shape (1,), extract the scalar value
        predicted_class = y_pred[0]

        # Decode the predicted label using the label encoder
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        return predicted_label

    except Exception as e:
        return f"Error: {str(e)}"


