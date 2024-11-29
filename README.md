# Sentiment Analysis Suite

A comprehensive project for sentiment analysis on social media posts. This repository covers various aspects of natural language processing (NLP), including data preprocessing, machine learning, deep learning, and the deployment of a sentiment analysis web application.

## Features
- Data preprocessing pipeline for cleaning and preparing text data.
- Sentiment classification using multiple models:
  - Logistic Regression, Naive Bayes, Support Vector Machine (SVM) for baseline ML approaches.
  - Deep Learning models like LSTM and Transformer-based BERT.
- Interactive web application for real-time sentiment prediction using different models.
- Comprehensive exploratory data analysis (EDA) in Jupyter notebooks.

## Project Structure

- **data/**: Contains raw and processed datasets.
  
- **notebooks/**: Includes Jupyter notebooks for exploratory data analysis and model experimentation:
  - `baseline_models.ipynb`: Classical machine learning model experiments.
  - `data_preparation.ipynb`: Data cleaning and preprocessing.
  - `deep_learning_models.ipynb`: Deep learning experiments using LSTM.
  - `model_evaluation_and_testing.ipynb`: Model testing and evaluation with metrics.
  - `transformer_models_distilbert.ipynb`: Experiments with transformer-based models.
  - `data_preprocessing.log`: Log file detailing data preprocessing steps.

- **app/**: Web app backend and components:
  - **models/**: Python scripts for each model used in the web app for real-time predictions:
    - `bert_predictor.py`: Script to load and predict sentiment using a BERT model.
    - `logistic_predictor.py`: Script for logistic regression predictions.
    - `lstm_predictor.py`: Script to predict sentiment using an LSTM model.
    - `naive_bayes_predictor.py`: Script for Naive Bayes sentiment predictions.
    - `svm_predictor.py`: Script for SVM sentiment analysis.
  - **static/**: Front-end assets like CSS for styling.
    - `style.css`: Main stylesheet for web application design and layout.
  - **templates/**: HTML files for rendering the web app interface.
    - `index.html`: Main web interface for the sentiment analysis application.
  - **app.py**: Main Flask backend server to run the web application.

- **models/**: Saved trained models for deployment, includes both traditional ML and deep learning models.

- **.gitignore**: Specifies files and directories to ignore in version control.

- **LICENSE**: Project license information.


## Web Application Interface
- **Sample Sentences**: Easy-to-click buttons provide sample inputs to test different models. This helps users quickly see how the app behaves with predefined inputs.
- **Model Selection**: Users can select the model for prediction from a list of available options using an intuitive and visually appealing button-based selection, replacing the traditional dropdown for easier accessibility.
- **Predict Sentiment**: Displays results with an engaging UI, indicating whether the sentiment is positive, negative, or neutral using emojis and color-coded panels. The prediction outcome is visually distinct with clear and easy-to-understand feedback.

### Screenshots:

1. **Initial App Interface with Sample Sentences:**
   ![App Screenshot - Sample Sentences](screenshots/Screenshot%202024-11-29%20045712.png)
   - This screenshot shows the easy-to-click sample sentences that users can quickly use to see how the app analyzes different types of text.

2. **User Input for Sentiment Prediction:**
   ![App Screenshot - User Input](screenshots/Screenshot%202024-11-29%20045743.png)
   - In this screenshot, a user has entered their own text for sentiment prediction and selected "SVM" as the prediction model.

3. **Predicted Sentiment Displayed:**
   ![App Screenshot - Sentiment Result](screenshots/Screenshot%202024-11-29%20045731.png)
   - This screenshot shows the sentiment prediction displayed with a colored label and an emoji to make the result more intuitive. The UI also maintains a soft, gradient background for a modern, professional feel.

4. **Final App Interface with All Elements:**
   ![App Screenshot - Final Interface](screenshots/Screenshot%202024-11-29%20045758.png)
   - This screenshot provides a complete overview of the application once all elements are in place, showcasing both the sample sentence input and the user's ability to manually input text for analysis.

## Getting Started
1. **Install Dependencies**:
   - Run `pip install -r requirements.txt` to install all necessary packages.
2. **Run the Web Application**:
   - Start the application using: `python app/app.py`
3. **Access the Web App**:
   - Open your browser and go to `http://localhost:5000` to interact with the sentiment analysis models.

## Models Implemented
1. **Logistic Regression, Naive Bayes, and SVM**:
   - Baseline classical machine learning models for quick training and reasonable accuracy.
2. **LSTM**:
   - A deep learning model that captures temporal dependencies in text using a bidirectional LSTM.
3. **BERT (DistilBERT)**:
   - A Transformer-based model that captures rich contextual information for robust sentiment analysis.

## Data Preprocessing and Feature Engineering
The data preprocessing pipeline includes the following steps:
1. **Dataset Analysis**:
   - Load the dataset, inspect its structure, and perform initial analysis to identify potential issues like missing values.
2. **Data Cleaning**:
   - Expand contractions, normalize text, remove special characters, URLs, mentions, hashtags, and stopwords.
3. **Feature Engineering**:
   - Add additional features such as text length, word count, and sentiment scores using tools like TextBlob.
4. **Handling Class Imbalance**:
   - Apply SMOTE to generate synthetic examples for minority classes, ensuring balanced data.
5. **Data Splitting**:
   - Use stratified sampling to split the data into training, validation, and test sets, preserving class distribution.
6. **Text Representation**:
   - Implement Bag of Words (BoW) and TF-IDF vectorization, as well as GloVe embeddings for representing text data.
7. **Embedding Integration**:
   - Use pre-trained GloVe embeddings to represent text data, enhancing the deep learning models' performance.
8. **Save Processed Data**:
   - Save the cleaned and feature-engineered datasets for future use.

## Running the Models and Predictions
- After preprocessing, all models are trained and saved in the `models/` directory.
- Each model has its dedicated predictor file (e.g., `logistic_predictor.py`, `lstm_predictor.py`) for loading the model and making predictions.

## How to Use the Web App
1. **Input Text**: Enter your text or click on a sample sentence.
2. **Select a Model**: Choose from Logistic Regression, Naive Bayes, SVM, LSTM, or BERT.
3. **Predict Sentiment**: Click on "Predict Sentiment" and view the result below, color-coded based on the sentiment.

## Technologies Used
- **Python** for data processing, model training, and backend.
- **Flask** for the web framework.
- **Bootstrap** for front-end design, ensuring a responsive and professional interface.
- **Machine Learning Libraries**: Scikit-learn, TensorFlow, Keras, PyTorch.
- **NLP Libraries**: NLTK, Gensim, Transformers by Hugging Face.

## License
This project is licensed under the MIT License.

## Next Steps
- Enhance the UI for better user experience.
- Expand to include more sophisticated Transformer-based models (e.g., GPT).
- Integrate more advanced NLP features like named entity recognition (NER) or topic modeling for deeper insights.

## Contribution
Feel free to contribute by submitting a pull request or reporting issues.

