from matplotlib import category
import pandas as pd
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download("movie_reviews")


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

# Train Model
vectorizer = CountVectorizer(max_features=2000) # Limits vocab to top 2000 words in dataset to improve efficiency
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
