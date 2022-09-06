#!/usr/bin/env python

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
import numpy as np

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

        # Convert the texts to vectors
        if args.tfidf:
            vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
        else:
            # Bag of Words vectorizer
            vec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

        # Create the classifier with the given parameters
        classifier = classifiers[args.model_name](**param_dict)

        # Log the experiment in ML flow
        mlflow.log_param("SEED", SEED)
        mlflow.log_param("SENTIMENT", args.sentiment)
        mlflow.log_param("TFIDF", args.tfidf)
        mlflow.log_param("MODEL NAME", classifier.__class__.__name__)
        mlflow.log_param("FOLDS", args.folds)
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