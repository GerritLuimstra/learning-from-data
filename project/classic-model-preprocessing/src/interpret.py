"""
This script is used to interpret our best performing LinearSVC model.
"""
import random
import numpy as np
from sklearn.svm import SVC
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from helpers import read_corpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score

from helpers import my_preprocessor

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
NGRAM_RANGE = 1
USE_LEMMATIZATION = False
USE_STEMMING = False if USE_LEMMATIZATION else True
USE_TFIDF = True
REMOVE_EMOJIS = False
POS_TAGGING = False
PREPROCESS = True

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
        vec = TfidfVectorizer(preprocessor=lambda doc: my_preprocessor(doc, preprocess=PREPROCESS, remove_emojis=REMOVE_EMOJIS,
                                pos_tags = POS_TAGGING, 
                                stemmer=stemmer, lemmatizer=lemmatizer),
                                ngram_range = (1,NGRAM_RANGE), 
                                min_df = 5,
                                token_pattern = '[a-zA-Z0-9@#_]+')
    else:
        vec = CountVectorizer(preprocessor=lambda doc: my_preprocessor(doc, preprocess=PREPROCESS, remove_emojis=REMOVE_EMOJIS,
                                pos_tags = POS_TAGGING, 
                                stemmer=stemmer, lemmatizer=lemmatizer),
                                ngram_range = (1,NGRAM_RANGE), 
                                min_df = 5,
                                token_pattern = '[a-zA-Z0-9@#_]+')

    # Transform the input data to the new vocabulary
    X_train = vec.fit_transform(X_train)
    X_dev = vec.transform(X_dev)

    # Create and fit the classifier with the given parameters
    classifier = SVC(C = 1.4, kernel = 'linear')
    classifier.fit(X_train, y_train)
    
    # Obtain the scores on the train set
    y_pred = classifier.predict(X_dev)
    print("f1_macro_train devset:", f1_score(y_dev, y_pred, average='macro'))

    # for i in range(1):
    # Print out the topic name
    # print(classifier.classes_[i])
    # Print out the feature importances
    # print(vec.get_feature_names())
    # print(classifier.coef_)
    feature_importances(classifier.coef_.toarray()[0], vec.get_feature_names())

    