from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.metrics import f1_score
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

import argparse
import random as python_random
import numpy as np
from joblib import dump, load

from utilities import read_tweets, parse_values, set_seed

# Setup the classifier mapping
CLASSIFIERS = {
    'dt': DecisionTreeClassifier, 'knn': KNeighborsClassifier, 
    'rf': RandomForestClassifier, 'et': ExtraTreesClassifier,
    'nb': MultinomialNB, 
    'svc': SVC, 'linearsvc': LinearSVC
}

def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", type=str,
                        help="File that will be used to perform training.", default = "../data/train.tsv")
    parser.add_argument("-mf", "--model_file", type=str,
                        help="File where the trained model will be saved")
    parser.add_argument("-ff", "--feature_file", type=str,
                        help="File where the top 10 feature importances will be saved")
    parser.add_argument("-s", "--seed", type=int,
                        help="If added, random seed to use for reproducible \
                        results")
    parser.add_argument("-m", "--model_name", type=str, default='nb', help="The model to use. Can be one of ['nb', 'et', 'rf', 'knn', 'svc', 'linearsvc']")
    parser.add_argument("-a", "--args", default=[], nargs='+', help="The arguments passed to the ML model")
    args = parser.parse_args()
    return args

def read_args():
    """Read the parsed command line arguments"""
    args = create_arg_parser()

    # Check if the classifier arguments are properly given
    if not len(args.args) % 2 == 0:
        print("Invalid arguments specified. Should be in the form: param1 value1 param2 value2")
        exit(0)

    # Parse the arguments
    params = args.args[0::2]
    values = parse_values(args.args[1::2])
    param_dict = dict(zip(params, values))

    return args, param_dict

def save_vocabulary_size(size, output_file):
    """Save the number of features."""
    output = [["Vocabulary size:", str(size)]]
    np.savetxt(output_file, output, delimiter=" ", fmt="%s")

def main():
    """Main function to construct and train classical model for offensive tweet
    classification."""

    # Read the command line arguments.
    args, params = read_args()

    # Make results reproducible
    set_seed(args.seed)

    # Read in the train from the specified file
    X_train, y_train = read_tweets(args.train_file)
    
    # Setup bag-of-words vectorizer
    vec = CountVectorizer()

    # Fit the vectorizer and transform the training data
    X_train_vec = vec.fit_transform(X_train)

    # Save the number of features
    save_vocabulary_size(len(vec.vocabulary_), args.feature_file)

    # Create the classifier with the given parameters
    classifier = CLASSIFIERS[args.model_name](**params)

    # Train the model with the train data
    classifier.fit(X_train_vec, y_train)

    if args.model_file:
        dump((classifier, vec), args.model_file)
       
if __name__ == "__main__":
    main()