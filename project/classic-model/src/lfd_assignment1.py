"""
With this script various models can be trained.
For more information on the parameters that can be used,
consult the argument parser documentation in the helpers.py file.
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, accuracy_score
from helpers import create_vocabulary, read_corpus, create_arg_parser, parse_values
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import random

# Ensure reproducability
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

if __name__ == "__main__":

    # Setup the argument parser
    # and parse the arguments
    args = create_arg_parser()

    # Check if the classifier arguments are properly given
    if not len(args.args) % 2 == 0:
        print("Invalid arguments specified. Should be in the form: param1 value1 param2 value2")
        exit(0)

    # Parse the arguments
    params = args.args[0::2]
    values = parse_values(args.args[1::2])
    param_dict = dict(zip(params, values))

    # Read in the data from the specified file
    X, y = read_corpus(args.cross_file, False)

    # Setup the classifier mapping
    classifiers = {
        'dt': DecisionTreeClassifier, 'knn': KNeighborsClassifier, 
        'rf': RandomForestClassifier, 'nb': MultinomialNB, 
        'svm': SVC, 'linearsvc': LinearSVC
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

    # Setup the stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Create the vocabulary
    if args.stemming:
        vocabulary = create_vocabulary(X, stemmer=stemmer, reduce_words=not args.no_reduce_words)
    elif args.lemmatization:
        vocabulary = create_vocabulary(X, lemmatizer=lemmatizer, reduce_words=not args.no_reduce_words)
    else:
        vocabulary = create_vocabulary(X, reduce_words=not args.no_reduce_words)

    # Convert the texts to vectors
    if args.tfidf:
        vec = TfidfVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=(1, args.ngram_range))
    else:
        vec = CountVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=(1, args.ngram_range))

    # Transform the input data to the new vocabulary
    X = vec.fit_transform(X)

     # Create the classifier with the given parameters
    classifier = classifiers[args.model_name](**param_dict)

    # Setup the connection to ML flow (for tracking)
    mlflow.set_tracking_uri("http://localhost:5050")
    _ = mlflow.set_experiment("Learning From Data Assignment 1")

    # Start the experiment
    with mlflow.start_run():

        # Log the experiment in ML flow
        mlflow.log_param("SEED", SEED)
        mlflow.log_param("TFIDF", args.tfidf)
        mlflow.log_param("MODEL NAME", classifier.__class__.__name__)
        mlflow.log_param("FOLDS", args.folds)
        mlflow.log_param("LEMMATIZATION", args.lemmatization)
        mlflow.log_param("STEMMING", args.stemming)
        mlflow.log_param("NGRAM_RANGE", args.ngram_range)
        mlflow.log_param("REDUCE WORDS", not args.no_reduce_words)
        mlflow.log_param("VOCAB SIZE", len(vec.vocabulary_))
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

        # Fit the model on the whole dataset
        # so that it can be used for inference
        classifier.fit(X, y)

    if args.inference_file is not None:

        # Load in the inference data
        X_inf, y_inf = read_corpus(args.inference_file, False)

        # Transform the input data to the new vocabulary
        X_inf = vec.fit_transform(X_inf)

        # Obtain the model predictions
        predictions = classifier.predict(X_inf)

        # Obtain the metrics
        for metric in metrics:
            print(metric, metrics[metric](classifier, X_inf, y_inf))

        # Display the confusion matrix
        plot_confusion_matrix(classifier, X_inf, y_inf)
        plt.show()