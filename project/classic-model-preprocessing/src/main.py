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
from nltk import pos_tag
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

    # Setup the stemmer and lemmatizer
    stemmer = None
    lemmatizer = None
    if args.stemming:
        stemmer = PorterStemmer()
    elif args.lemmatization:
        lemmatizer = WordNetLemmatizer()

    # Convert the texts to vectors
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=lambda doc: my_preprocessor(doc, stemmer=stemmer, lemmatizer=lemmatizer,
                                reduce_words=not args.no_reduce_words,
                                pos_tagging=args.pos_tagging),
                                ngram_range = (1,args.ngram_range), 
                                min_df = 2,
                                token_pattern = '[a-zA-Z0-9@#_]+')
    else:
        vec = CountVectorizer(preprocessor=lambda doc: my_preprocessor(doc, stemmer=stemmer, lemmatizer=lemmatizer,
                                reduce_words=not args.no_reduce_words,
                                pos_tagging=args.pos_tagging),
                                ngram_range = (1,args.ngram_range), 
                                min_df = 2,
                                token_pattern = '[a-zA-Z0-9@#_]+')

    # Transform the input data to the new vocabulary
    X_train = vec.fit_transform(X_train)
    X_dev = vec.transform(X_dev)
    
    # Create the classifier with the given parameters
    classifier = classifiers[args.model_name](**param_dict)

    # Fit the classifier to the train data
    classifier.fit(X_train, y_train)

    # Setup the connection to ML flow (for tracking)
    mlflow.set_tracking_uri("http://localhost:5050")
    _ = mlflow.set_experiment("Learning From Data Project - Classic Model (Preprocessing)")

    # Start the experiment
    with mlflow.start_run():

        # Log the experiment in ML flow
        mlflow.log_param("SEED", SEED)
        mlflow.log_param("TFIDF", args.tfidf)
        mlflow.log_param("MODEL NAME", classifier.__class__.__name__)
        mlflow.log_param("LEMMATIZATION", args.lemmatization)
        mlflow.log_param("STEMMING", args.stemming)
        mlflow.log_param("NGRAM_RANGE", args.ngram_range)
        mlflow.log_param("REDUCE WORDS", not args.no_reduce_words)
        mlflow.log_param("VOCAB SIZE", len(vec.vocabulary_))
        mlflow.log_params(classifier.get_params())

        # Obtain the scores on the train set
        y_pred = classifier.predict(X_train)
        mlflow.log_metric("f1_weighted_train", f1_score(y_train, y_pred, average='weighted'))

        # Obtain the scores on the dev set
        y_pred = classifier.predict(X_dev)
        mlflow.log_metric("f1_weighted_dev", f1_score(y_dev, y_pred, average='weighted'))
        mlflow.log_metric("f1_macro_dev", f1_score(y_dev, y_pred, average='macro'))
        mlflow.log_text(classification_report(y_dev, y_pred), "classification_report.txt")


    if args.inference_file is not None:

        # Load in the inference data
        X_inf, y_inf = read_corpus(args.inference_file)

        # Transform the input data to the new vocabulary
        X_inf = vec.fit_transform(X_inf)

        # Obtain the model predictions
        y_pred = classifier.predict(X_inf)

        # Print metrics
        print("f1_weighted_test", f1_score(y_inf, y_pred, average='weighted'))
        print("f1_macro_test", f1_score(y_inf, y_pred, average='macro'))

        # Display the confusion matrix
        plot_confusion_matrix(classifier, X_inf, y_inf)
        plt.show()