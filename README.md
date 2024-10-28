# Sentiment Analysers

I created this project as a learning exercise to explore basic machine learning concepts, including data pre-processing, feature extraction, model training and evaluation. The goal of the project was to build a sentiment analysis classifier that can identify if a movie review is positive or negative.

I plan on further experimenting with sentiment analysis and will update future files to this repo

## Project Overview

In this project, I used Python and several libraries (`pandas`, `nltk`, `scikit-learn`) to build a sentiment analyser. The classifier is trained on the NLTK movie reviews dataset and uses a Naive Bayes model to predict the sentiment of unseen movie reviews.

## Project Structure

- **Data Pre-processing**: The raw text data is processed by converting to lowercase and removing common English stop words.
- **Feature Extraction**: The text is transformed into numerical feature vectors using `CountVectorizer`, with a maximum vocabulary size of 2000 words.
- **Model Training**: A Naive Bayes classifier (`MultinomialNB`) is trained on 80% of the data.
- **Evaluation**: The model is evaluated on the remaining 20% of the data, and key performance metrics such as accuracy and a classification report are displayed.
- **Prediction**: A function allows for the prediction of sentiment on new text input.

## Results

The classifier achieved an accuracy of approximately **79.25%** on the test data, demonstrating that a simple Naive Bayes model can effectively distinguish between positive and negative movie reviews.

## Future Improvements

- Explore other models such as **Logistic Regression** or **Support Vector Machines**.
- Experiment with different feature extraction techniques, including **TF-IDF** and **n-grams**, to capture more nuanced sentiment.
- Use **cross-validation** to ensure robust performance.
- Expand the dataset to see if the model generalises to other types of reviews or sentiment-related texts.

## Dependencies

- `pandas`
- `nltk`
- `scikit-learn`
