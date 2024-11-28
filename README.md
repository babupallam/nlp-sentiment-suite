
# nlp-sentiment-suite

A comprehensive project for sentiment analysis on social media posts. This repository demonstrates skills in data preprocessing, machine learning, deep learning, and web application deployment.

## Features
- Data preprocessing and exploratory analysis.
- Sentiment classification using ML and DL models.
- Real-time predictions through an interactive web app.

## Project Structure
- **data/**: Contains raw and processed datasets.
- **notebooks/**: Includes exploratory data analysis and model experiments.
- **app/**: Web app backend, templates, and static assets.
- **models/**: Saved trained models for deployment.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the web app: `python app/app.py`
3. View the app in your browser at `http://localhost:5000`.

## License
This project is licensed under the MIT License.



1. Add comprehensive data preprocessing pipeline for text dataset

This commit includes the implementation of a full data preprocessing pipeline for a text dataset, covering multiple steps to ensure data quality and preparation for model training. The following steps have been added:

1. Dataset Analysis:
   - Load and inspect the dataset, check for missing values, and understand its structure.
   - Print initial statistics, such as shape, columns, data types, and missing values.

2. Data Cleaning:
   - Drop rows with missing values in critical columns like 'category'.
   - Expand contractions to standardize the text (e.g., "don't" -> "do not").
   - Normalize the text by converting to lowercase, removing URLs, mentions, hashtags, and special characters.
   - Tokenize the text into individual words for further processing.
   - Remove common stopwords and apply lemmatization to convert words to their base forms.

3. Feature Engineering:
   - Add new features to enrich the dataset, including:
     - Text length (number of characters).
     - Word count (number of words in the text).
     - Sentiment score using TextBlob to derive additional signals.
  
4. Handling Class Imbalance:
   - Use SMOTE (Synthetic Minority Oversampling Technique) to balance the training dataset by generating synthetic examples of the minority class.

5. Data Splitting:
   - Split the cleaned dataset into training, validation, and test sets.
   - Apply stratified sampling to ensure balanced class distributions in each subset.

6. Text Representation:
   - Implement Bag of Words (BoW) and TF-IDF vectorization to convert the text into numerical features for machine learning models.
   - Include demonstration examples that show the vocabulary size, feature vectors, and top terms with their respective scores.

7. Embedding Integration:
   - Integrate pre-trained GloVe embeddings using `gensim` to enrich text representation.
   - Calculate average word embeddings for each document.

8. Error Handling and Demonstrations:
   - Address common errors related to missing or incompatible data (e.g., NaNs in 'category').
   - Added print statements to demonstrate data transformations and provide clarity on the preprocessing results.

9. Save Processed Data:
   - Save cleaned datasets and numerical representations (BoW, TF-IDF, and embeddings) to CSV and serialized formats for future reuse.




