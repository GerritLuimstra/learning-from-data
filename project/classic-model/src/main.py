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
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, accuracy_score
from helpers import read_corpus, create_arg_parser, parse_values
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
    X_train, y_train = read_corpus(args.train_file)
    X_dev, y_dev = read_corpus(args.dev_file)
    
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
    for c in np.unique(y_train):
        metrics |= {
            'f1_' + str(c): make_scorer(f1_score, average=None, labels=[c]),
            'recall_' + str(c): make_scorer(recall_score, average=None, labels=[c]),
            'precision_' + str(c): make_scorer(precision_score, average=None, labels=[c])
        }

    vec = CountVectorizer()

    # Transform the input data to the new vocabulary
    X_train = vec.fit_transform(X_train)
    X_dev = vec.transform(X_dev)

    # Create the classifier with the given parameters
    classifier = classifiers[args.model_name](**param_dict)
    
    # Fit the classifier to the train data
    classifier.fit(X_train, y_train)

    # Setup the connection to ML flow (for tracking)
    mlflow.set_tracking_uri("http://localhost:5050")
    _ = mlflow.set_experiment("Learning From Data Project - Classic Model (BoW)")

    # Start the experiment
    with mlflow.start_run():

        # Log the experiment in ML flow
        mlflow.log_param("SEED", SEED)
        mlflow.log_param("MODEL NAME", classifier.__class__.__name__)
        mlflow.log_param("VOCAB SIZE", len(vec.vocabulary_))
        mlflow.log_params(classifier.get_params())

        # Obtain the scores on the train set
        y_pred = classifier.predict(X_train)
        mlflow.log_metric("f1_weighted_train", f1_score(y_train, y_pred, average='weighted'))

        # Obtain the scores on the dev set
        y_pred = classifier.predict(X_dev)
        mlflow.log_metric("f1_weighted_dev", f1_score(y_dev, y_pred, average='weighted'))
        mlflow.log_text(classification_report(y_dev, y_pred), "classification_report.txt")


    if args.inference_file is not None:

        # Load in the inference data
        X_inf, y_inf = read_corpus(args.inference_file)

        # Transform the input data to the new vocabulary
        X_inf = vec.fit_transform(X_inf)

        # Obtain the scores on the inference set
        y_pred = classifier.predict(X_inf)

        print("f1_weighted", f1_score(y_inf, y_pred, average='weighted'))
        print(classification_report(y_inf, y_pred))

        # Display the confusion matrix
        plot_confusion_matrix(classifier, X_inf, y_inf)
        plt.show()