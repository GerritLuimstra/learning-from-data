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
from helpers import create_initial_vocabulary, read_corpus, create_arg_parser, parse_values
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np

import mlflow
import mlflow.sklearn
import random

# Ensure reproducability
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

STOP_WORDS = 'english'
NGRAM_RANGE = (1, 1)
USE_LEMMATIZATION = False
USE_STEMMING = False if USE_LEMMATIZATION else True

if __name__ == "__main__":

    # Setup the argument parser
    # and parse the arguments
    args = create_arg_parser()

    if not len(args.args) % 2 == 0:
        print("Invalid arguments specified. Should be in the form: param1 value1 param2 value2")
        exit(0)

    # Parse the arguments
    params = args.args[0::2]
    values = parse_values(args.args[1::2])
    param_dict = dict(zip(params, values))

    # Setup the connection to ML flow (for tracking)
    mlflow.set_tracking_uri("http://localhost:5000")
    _ = mlflow.set_experiment("Learning From Data Assignment 1")

    # Read in the data from the train and dev file
    X_train, y_train = read_corpus(args.train_file, False)
    X_test, y_test = read_corpus(args.dev_file, False)

    # Combine the train and test file into one big dataset
    # as we will be using cross validation instead of a single train/test split
    X = X_train + X_test
    y = y_train + y_test

    # Setup the classifier mapping
    classifiers = {
        'dt': DecisionTreeClassifier, 'knn': KNeighborsClassifier, 
        'rf': RandomForestClassifier, 'nb': MultinomialNB, 
        'svm': SVC
    }

    # Setup the metrics to track
    metrics = {
        'test_f1_macro': make_scorer(f1_score, average='macro'),
        'test_accuracy': make_scorer(accuracy_score)
    }
    for c in np.unique(y):
        metrics |= {
            'f1_' + str(c): make_scorer(f1_score, average=None, labels=[c]),
            'recall_' + str(c): make_scorer(recall_score, average=None, labels=[c]),
            'precision_' + str(c): make_scorer(precision_score, average=None, labels=[c])
        }

    # Start the experiment
    with mlflow.start_run():

        # Setup the stemmer and lemmatizer
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        # Setup a general analyzer function
        # so that we can optionally include
        # stemming or lemmatization
        analyzer = CountVectorizer(
            lowercase=True,
            stop_words='english', 
            ngram_range=NGRAM_RANGE,
            preprocessor=lambda x: x, 
            tokenizer=lambda x: x
        ).build_analyzer()

        # Setup the proper analyzer_
        if USE_LEMMATIZATION:
            analyzer_ = lambda doc: (lemmatizer.lemmatize(w) for w in analyzer(doc))
        elif USE_STEMMING:
            analyzer_ = lambda doc: (stemmer.stem(w) for w in analyzer(doc))
        else:
            analyzer_ = lambda x: x

        # Create the initial vocabulary (before stemming, stopword removal etc)
        initial_vocabulary = create_initial_vocabulary(X)

        # Remove words that are not in the dictionary
        X = [list(filter(lambda word : word in initial_vocabulary, words)) for words in X]

        # Convert the texts to vectors
        if args.tfidf:
            vec = TfidfVectorizer(analyzer=analyzer_)
        else:
            # Bag of Words vectorizer
            vec = CountVectorizer(analyzer=analyzer_)

        # Fit the vectorizer on the dataset
        X = vec.fit_transform(X)

        # Create the classifier with the given parameters
        classifier = classifiers[args.model_name](**param_dict)

        # Log the experiment in ML flow
        mlflow.log_param("SEED", SEED)
        mlflow.log_param("TFIDF", args.tfidf)
        mlflow.log_param("MODEL NAME", classifier.__class__.__name__)
        mlflow.log_param("FOLDS", args.folds)
        mlflow.log_param("LEMMATIZATION", USE_LEMMATIZATION)
        mlflow.log_param("STEMMING", USE_STEMMING)
        mlflow.log_param("NGRAM_RANGE", NGRAM_RANGE)
        mlflow.log_params(classifier.get_params())

        # Setup stratified cross validation
        # Stratification ensures that each fold has the 
        # same class proportion as the main dataset
        # https://en.wikipedia.org/wiki/Stratified_sampling
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True)

        # Obtain the scores
        scores = cross_validate(classifier, X, y, cv=skf, scoring=metrics)
        for metric in scores:
            if not "test_" in metric:
                continue
            mlflow.log_metric(metric.replace("test_", "") + " std", np.std(scores[metric]))
            mlflow.log_metric(metric.replace("test_", "") + " mean", np.mean(scores[metric]))