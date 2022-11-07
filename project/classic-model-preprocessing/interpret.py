"""
This script is used to interpret our best performing LinearSVC model.
"""
import random
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from helpers import read_corpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt

from utilities import preprocessor

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
SEED = 4
random.seed(SEED)
np.random.seed(SEED)

# Hardcode the properties
NGRAM_RANGE = 1
USE_LEMMATIZATION = False
USE_STEMMING = False if USE_LEMMATIZATION else True
USE_TFIDF = False
REMOVE_EMOJIS = True
POS_TAGGING = False
GLOVE = True

if __name__ == "__main__":

    # Read in the data from the specified file
    X_train, y_train = read_corpus("../data/train.tsv")
    X_dev, y_dev = read_corpus("../data/dev.tsv")
    X_inf, y_inf = read_corpus("../data/test.tsv")

    # Setup the stemmer and lemmatizer
    stemmer = None
    lemmatizer = None
    if USE_STEMMING:
        stemmer = PorterStemmer()
    elif USE_LEMMATIZATION:
        lemmatizer = WordNetLemmatizer()

    if USE_TFIDF:
        vec = TfidfVectorizer(preprocessor=lambda doc: preprocessor(doc, glove=GLOVE, remove_emojis=REMOVE_EMOJIS,
                                pos_tags = POS_TAGGING, 
                                stemmer=stemmer, lemmatizer=lemmatizer),
                                ngram_range = (1,NGRAM_RANGE), 
                                min_df = 5,
                                token_pattern = '[a-zA-Z0-9@#_]+')
    else:
        vec = CountVectorizer(preprocessor=lambda doc: preprocessor(doc, glove=GLOVE, remove_emojis=REMOVE_EMOJIS,
                                pos_tags = POS_TAGGING, 
                                stemmer=stemmer, lemmatizer=lemmatizer),
                                ngram_range = (1,NGRAM_RANGE), 
                                min_df = 5,
                                token_pattern = '[a-zA-Z0-9@#_]+')

    # Transform the input data to the new vocabulary
    X_train = vec.fit_transform(X_train)
    X_dev = vec.transform(X_dev)
    X_inf = vec.transform(X_inf)

    # # Create and fit the classifier with the given parameters
    classifier = ExtraTreesClassifier(ccp_alpha = 0.0001, class_weight = 'balanced', min_samples_split = 5)
    classifier.fit(X_train, y_train)
    
    # Obtain the scores on the test set

    # Print classification report
    y_pred = classifier.predict(X_train)
    print(classification_report(y_train, y_pred))
    y_pred = classifier.predict(X_dev)
    print(classification_report(y_dev, y_pred))
    y_pred = classifier.predict(X_inf)
    print(classification_report(y_inf, y_pred))

    # Plot confusion matrix
    plot_confusion_matrix(classifier, X_inf, y_inf)
    plt.show()

    # Obtain list of feature importances
    feature_importances(classifier.feature_importances_, vec.get_feature_names_out())

    