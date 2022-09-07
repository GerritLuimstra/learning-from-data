'''TODO: add high-level description of this Python script'''

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from helpers import read_corpus, create_arg_parser, parse_values
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
NGRAM_RANGE = (1, 2)
MAX_WORDS = 50000
STRIP_ACCENTS = False
USE_LEMMATIZATION = False
USE_STEMMING = True

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
    X_train, y_train = read_corpus(args.train_file, args.sentiment)
    X_test, y_test = read_corpus(args.dev_file, args.sentiment)

    # Combine the train and test file into one big dataset
    # as we will be using cross validation instead of a single train/test split
    X = X_train + X_test
    y = y_train + y_test

    metrics = ["accuracy", "f1_macro"]
    classifiers = {
        'dt': DecisionTreeClassifier, 'knn': KNeighborsClassifier, 
        'rf': RandomForestClassifier, 'nb': MultinomialNB, 
        'svm': SVC
    }
    with mlflow.start_run():

        # Setup the stemmer and lemmatizer
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        # Setup a general analyzer function
        # so that we can optionally include
        # stemming or lemmatization
        analyzer = CountVectorizer(
            lowercase=True, stop_words=STOP_WORDS, 
            ngram_range=NGRAM_RANGE, max_features=MAX_WORDS,
            strip_accents=STRIP_ACCENTS, 
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

        # Convert the texts to vectors
        if args.tfidf:
            vec = TfidfVectorizer(analyzer=analyzer_)
        else:
            # Bag of Words vectorizer
            vec = CountVectorizer(analyzer=analyzer_)

        # Create the classifier with the given parameters
        classifier = classifiers[args.model_name](**param_dict)

        # Log the experiment in ML flow
        mlflow.log_param("SEED", SEED)
        mlflow.log_param("SENTIMENT", args.sentiment)
        mlflow.log_param("TFIDF", args.tfidf)
        mlflow.log_param("MODEL NAME", classifier.__class__.__name__)
        mlflow.log_param("FOLDS", args.folds)
        mlflow.log_param("STOP_WORDS", STOP_WORDS)
        mlflow.log_param("LEMMATIZATION", USE_LEMMATIZATION)
        mlflow.log_param("STEMMING", USE_STEMMING)
        mlflow.log_param("NGRAM_RANGE", NGRAM_RANGE)
        mlflow.log_param("MAX_WORDS", MAX_WORDS)
        mlflow.log_param("STRIP_ACCENTS", STRIP_ACCENTS)
        mlflow.log_params(classifier.get_params())

        # Setup the pipeline
        classifier = Pipeline([('vec', vec), ('cls', classifier)])

        # Setup stratified cross validation
        # Stratification ensures that each fold has the 
        # same class proportion as the main dataset
        # https://en.wikipedia.org/wiki/Stratified_sampling
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True)

        # Log the stats in MLFlow
        for metric in metrics:
            scores = cross_val_score(classifier, X, y, cv=skf, scoring=metric)
            mlflow.log_metric(metric + " std", np.std(scores))
            mlflow.log_metric(metric + " mean", np.mean(scores))