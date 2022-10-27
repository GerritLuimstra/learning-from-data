"""
This script is used to interpret our best performing LinearSVC model.
"""
import random
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from helpers import create_vocabulary, read_corpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score

from main import my_preprocessor

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
NGRAM_RANGE = (1, 2)
USE_LEMMATIZATION = False
USE_STEMMING = False if USE_LEMMATIZATION else False
USE_TFIDF = False
REDUCE_WORDS = False

if __name__ == "__main__":

    # Read in the data from the specified file
    X_train, y_train = read_corpus("../data/train.tsv")
    X_dev, y_dev = read_corpus("../data/dev.tsv")

    # Setup the stemmer and lemmatizer
    stemmer = None
    lemmatizer = None
    if USE_STEMMING:
        stemmer = PorterStemmer()
    elif USE_LEMMATIZATION:
        lemmatizer = WordNetLemmatizer()

    # Convert the texts to vectors
    if USE_TFIDF:
        vec = TfidfVectorizer(preprocessor=lambda doc: my_preprocessor(doc, stemmer=stemmer, lemmatizer=lemmatizer, reduce_words=REDUCE_WORDS),
                                ngram_range = NGRAM_RANGE, 
                                min_df = 5,
                                token_pattern = '[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+')
    else:
        vec = CountVectorizer(preprocessor=lambda doc: my_preprocessor(doc, stemmer=stemmer, lemmatizer=lemmatizer, reduce_words=REDUCE_WORDS),
                                ngram_range = NGRAM_RANGE, 
                                min_df = 5,
                                token_pattern = '[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+')
                                

    # Transform the input data to the new vocabulary
    X_train = vec.fit_transform(X_train)
    X_dev = vec.transform(X_dev)

    # Create and fit the classifier with the given parameters
    classifier = LinearSVC()
    classifier.fit(X_train, y_train)
    
    # Obtain the scores on the train set
    y_pred = classifier.predict(X_dev)
    print("f1_weighted_train devset:", f1_score(y_dev, y_pred, average='weighted'))

    for i in range(1):
        # Print out the topic name
        print(classifier.classes_[i])
        # Print out the feature importances
        feature_importances(classifier.coef_[i], vec.get_feature_names())

    