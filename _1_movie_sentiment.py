import pandas as pd
import nltk
from nltk.corpus import movie_reviews, stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


nltk.download("movie_reviews")
nltk.download("stopwords")


# Create list of tuples 'documents', where each tuple has:
#           full text of review, created by joining words in file
#           sentiment category, which is either 'pos' or 'neg

documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Construct DataFrame
df = pd.DataFrame(documents, columns=["review", "sentiment"])

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = text.lower().split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

df["review"] = df["review"].apply(preprocess_text)

# Train Model
vectorizer = CountVectorizer(max_features=2000, stop_words="english") # Limits vocab to top 2000 words in dataset to improve efficiency
X = vectorizer.fit_transform(df["review"])      # Matrix. Each row in X corresponds to 1 review. Each column to a word
y = df["sentiment"]                             # Extracts "sentiment" column as y. Contains "pos"/"neg" in each review
                                                # y is target variable for training sentinment classifier 

# Randomly split data into test and training subsets
#   20% data for testing
#   80% for training 
#   random_state=42 keeps same random result each time code runs. Any number would work
X_train, X_test, y_train, y_test = train_test_split(     
    X, y, test_size=0.2, random_state=42
)

# Train Naive Bayes classifier. NB is suited for text data represented as word counts
model = MultinomialNB()
model.fit(X_train, y_train)     # Trains it on training data. 
                                # X_train contains word count vectors for each review
                                # y_train is target array with 'pos'/'neg' labels for each review

# Evaluate model
y_pred = model.predict(X_test)  # Use trained model to predict sentiment for each review in test set    
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Predict
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

print(predict_sentiment("I absolutely loved this movie! It was fantastic"))
print(predict_sentiment("Terrible film! I hated it"))
print(predict_sentiment("The move was okay, nothing special"))