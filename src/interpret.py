'''TODO: add high-level description of this Python script'''

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, accuracy_score
from helpers import create_vocabulary, read_corpus, create_arg_parser, parse_values
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from pprint import pprint

import mlflow
import mlflow.sklearn
import random

# Ensure reproducability
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NGRAM_RANGE = (1, 1)
USE_LEMMATIZATION = True
USE_STEMMING = False if USE_LEMMATIZATION else False
USE_TFIDF = True

if __name__ == "__main__":

    # Read in the data
    X, y = read_corpus("data/reviews.txt", False)

    # Setup the stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Create the vocabulary
    if USE_STEMMING:
        vocabulary = create_vocabulary(X, stemmer=stemmer)
    elif USE_LEMMATIZATION:
        vocabulary = create_vocabulary(X, lemmatizer=lemmatizer)
    else:
        vocabulary = create_vocabulary(X)

    # Convert the texts to vectors
    if USE_TFIDF:
        vec = TfidfVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x)

    # Transform the input data to the new vocabulary
    X = vec.fit_transform(X)

    # Create the classifier with the given parameters
    classifier = MultinomialNB()
    classifier.fit(X, y)

    classes = classifier.classes_
    for c, name in enumerate(classes):
        print(name)
        probs = np.array([np.exp(prob) for prob in classifier.feature_log_prob_[c]])
        names = np.array(vec.get_feature_names())
        idxs = np.argsort(-probs)
        probs = probs[idxs]
        names = names[idxs]
        print(list(zip(names, probs))[:30])