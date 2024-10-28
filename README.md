# Sentiment Analysis Project

This project contains two separate scripts to perform sentiment analysis on text data, providing a hands-on exploration of basic machine learning and sentiment classification using Python.

I completed this as a learning exercise to explore basic machine learning concepts, including data pre-processing, feature extraction, model training and evaluation.

## Project Overview

### Files

1. **`_1_movie_sentiment.py`**: This script performs sentiment analysis on movie reviews using a Naive Bayes classifier. The script uses the NLTK movie reviews dataset and classifies reviews as either positive or negative.
   
2. **`_2_sentiment_emoji_reactor.py`**: This script provides a quick sentiment-based reaction to user input text, displaying an emoji based on the sentiment (positive, neutral, or negative). It uses TextBlob to analyse sentiment and displays a corresponding emoji.

### Key Components

- **Data Pre-processing**: Converts text to lowercase and removes stop words to standardise the text data.
- **Feature Extraction**: Uses CountVectorizer for movie reviews and TextBlob for emoji reactions.
- **Model Training and Evaluation**: Implements Naive Bayes for movie review classification, achieving around 79.25% accuracy.
- **Real-time User Interaction**: The emoji reactor allows for continuous user input, instantly displaying mood emojis based on text sentiment.

## Future Improvements

- Explore other models such as **Logistic Regression** or **Support Vector Machines** for `_1_movie_sentiment.py`.
- Experiment with feature extraction techniques (e.g., **TF-IDF**, **n-grams**) to capture more nuanced sentiment.
- Implement **cross-validation** to ensure robust performance.
- Expand the dataset for more diverse text sentiment analysis in `_2_sentiment_emoji_reactor.py`.

## Dependencies

- `pandas`
- `nltk`
- `scikit-learn`
- `textblob`
