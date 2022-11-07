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
import dill

from utilities import *

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
    parser.add_argument("-bf", "--best_features", action = "store_true", default = False,
                        help="If true, the top 10 feature importances will be printed and saved")
    parser.add_argument("-ff", "--feature_file", type=str,
                        help="File where the top 10 feature importances will be saved")
    parser.add_argument("-s", "--seed", type=int,
                        help="If added, random seed to use for reproducible \
                        results")
    parser.add_argument("-td", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-lm", "--lemmatization", action="store_true",
                        help="Whether to use Lemmatization (default False)")
    parser.add_argument("-st", "--stemming", action="store_true",
                        help="Whether to use Stemming (default False).")
    parser.add_argument("-ng", "--ngram_range", type=int, default=1, 
                        help="The upper n-gram range. This includes n-grams in the range (1, n). (default 1)")
    parser.add_argument("-g", "--glove", action="store_true", default=False, 
                        help="Preprocess the data using glove embedding.")
    parser.add_argument("-pt", "--pos_tags", action="store_true", default=False, 
                        help="Use pos tagging to filter words.")
    parser.add_argument("-rm", "--remove_emojis", action="store_true", default=False, 
                        help="Remove emojis.")
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

def save_feature_importances(coef, names, output_file, top_k=10):
    """Prints out the feature importances from a given set of feature coefficients and names"""
    imp, names = zip(*sorted(zip(coef, names), reverse=True))
    result = [["Vocabulary size:", len(names)], ["",""]]
    imp = imp[:top_k]
    names = names[:top_k]
    
    result.append(["Top 10 most important features",""])
    for weight, word in zip(imp, names):
        result.append([word, round(weight, 3)])

    np.savetxt(output_file, result, delimiter=" ", fmt="%s")

def setup_vectorizer(args):
    # Setup the stemmer and lemmatizer
    stemmer = None
    lemmatizer = None
    if args.stemming:
        stemmer = PorterStemmer()
    elif args.lemmatization:
        lemmatizer = WordNetLemmatizer()
    
    # Convert the texts to vectors
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=lambda doc: preprocessor(doc, glove=args.glove, remove_emojis=args.remove_emojis,
                                pos_tags = args.pos_tags, 
                                stemmer=stemmer, lemmatizer=lemmatizer),
                                ngram_range = (1,args.ngram_range), 
                                min_df = 5,
                                token_pattern = '[a-zA-Z0-9@#_]+')
    else:
        vec = CountVectorizer(preprocessor=lambda doc: preprocessor(doc, glove=args.glove, remove_emojis=args.remove_emojis,
                                pos_tags = args.pos_tags, 
                                stemmer=stemmer, lemmatizer=lemmatizer), 
                                ngram_range = (1,args.ngram_range),
                                min_df = 5,
                                token_pattern = '[a-zA-Z0-9@#_]+')
    return vec

def main():
    """Main function to construct and train classical model for offensive tweet
    classification."""

    # Read the command line arguments.
    args, params = read_args()

    # Make results reproducible
    set_seed(args.seed)

    # Read in the train from the specified file
    X_train, y_train = read_tweets(args.train_file)

    # Setup vectorizer
    vec = setup_vectorizer(args)
    
    # Fit the vectorizer and transform the training data
    X_train_vec = vec.fit_transform(X_train)

    # Save the number of features
    save_vocabulary_size(len(vec.vocabulary_), args.feature_file)

    # Create the classifier with the given parameters
    classifier = CLASSIFIERS[args.model_name](**params)

    # Train the model with the train data
    classifier.fit(X_train_vec, y_train)

    if args.model_file:
        dill.dump((classifier, vec), open(args.model_file, 'wb'))
    
    # Save top 10 important features for certain models
    if args.best_features:
        if args.model_name in ['rf', 'et']:
            save_feature_importances(classifier.feature_importances_, vec.get_feature_names_out(), args.feature_file)


if __name__ == "__main__":
    main()