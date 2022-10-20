"""
This script is used to interpret our best performing LinearSVC model.
"""
import random
import numpy as np
from sklearn.svm import LinearSVC
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from helpers import create_vocabulary, read_corpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def feature_importances(coef, names, top_k=10):
    """
    Prints out the feature importances from a given set of feature coefficients and names
    
    Parameters
    ----------
        coef : list of floats
            The list of LinearSVC coefficients
        names : PorterStemmer
            The names of the features
        top_k : int
            The top k features to print out

    Returns
    -------
    None
    """
    imp, names = zip(*sorted(zip(coef, names), reverse=True))
    imp = imp[:top_k]
    names = names[:top_k]
    for weight, word in zip(imp, names):
        print(word, "&", round(weight, 3), "\\\\")


# Ensure reproducability
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Hardcode the properties
NGRAM_RANGE = (1, 1)
USE_LEMMATIZATION = False
USE_STEMMING = False if USE_LEMMATIZATION else False
USE_TFIDF = True
REDUCE_WORDS = False

if __name__ == "__main__":

    # Read in the data
    X, y = read_corpus("data/cross.txt", False)

    # Setup the stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Create the vocabulary
    if USE_STEMMING:
        vocabulary = create_vocabulary(X, stemmer=stemmer, reduce_words=REDUCE_WORDS)
    elif USE_LEMMATIZATION:
        vocabulary = create_vocabulary(X, lemmatizer=lemmatizer, reduce_words=REDUCE_WORDS)
    else:
        vocabulary = create_vocabulary(X, reduce_words=REDUCE_WORDS)

    # Convert the texts to vectors
    if USE_TFIDF:
        vec = TfidfVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x)

    # Transform the input data to the new vocabulary
    X_= vec.fit_transform(X)

    # Create and fit the classifier with the given parameters
    classifier = LinearSVC(C=0.05)
    classifier.fit(X_, y)

    for i in range(len(np.unique(y))):
        # Print out the topic name
        print(classifier.classes_[i])
        # Print out the feature importances
        feature_importances(classifier.coef_[i], vec.get_feature_names())

    