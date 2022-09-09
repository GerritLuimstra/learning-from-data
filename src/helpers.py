import argparse
import numpy as np

def read_corpus(corpus_file, use_sentiment):
    '''TODO: add function description'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='train.txt', type=str,
                        help="Train file to learn from (default train.txt)")
    parser.add_argument("-df", "--dev_file", default='dev.txt', type=str,
                        help="Dev file to evaluate on (default dev.txt)")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-m", "--model_name", type=str, default='nb', help="The model to use. Can be one of ['nb', 'dt', 'rf', 'knn', 'svm']")
    parser.add_argument("-f", "--folds", type=int, default=5, help="The amount of folds to use for the cross validation")
    parser.add_argument("-a", "--args", default=[], nargs='+', help="The arguments passed to the ML model")
    args = parser.parse_args()
    return args

def parse_values(values):
    values_ = []
    for value in values:
        if "'" in value:
            values_.append(value.replace("'", ""))
        elif "." in value:
            values_.append(float(value))
        elif value == "None":
            values_.append(None)
        else:
            values_.append(int(value))
    return values_

def create_initial_vocabulary(X):

    # Obtain all words in the dataset
    flattened = [word for sample in X for word in sample]

    # Obtain the unique words and their frequencies
    words, frequency = np.unique(flattened, return_counts=True)

    # Remove all words that have a frequency of less than 5
    words = words[frequency >= 5]

    # Load in all the words from the english dictionary
    # from https://github.com/dwyl/english-words
    with open("src/english_wordlist.txt") as f:
        english_words = set(f.read().split("\n"))

    # Remove words that are not in the english language dictionary
    words = words[[word in english_words for word in words]]

    return set(words)