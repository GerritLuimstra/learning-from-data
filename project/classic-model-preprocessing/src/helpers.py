"""
This script provides helper functions for the main program
"""

import argparse
import numpy as np
import emoji
import re
import string

def read_corpus(corpus_file):
    """
    Reads in the dataset from a txt file and parses it into documents and corresponding labels.

    Each line is of the form [topic sentiment id content]
    and will be turned in a [content], [topic/sentiment]
    
    Parameters
    ----------
        corpus_file : str
            A link to the file containing the reviews

    Returns
    -------
    The parsed documents and labels
    """
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1][:-1])
    return documents, labels

def create_arg_parser():
    """
    Sets up the argument parser
    and parses the results from the terminal

    Parameters
    ----------
    None

    Returns
    -------
    The parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", type=str,
                        help="Txt file that will be used to perform cross-validation.")
    parser.add_argument("-d", "--dev_file", type=str,
                        help="File that will be used to perform testing.")
    parser.add_argument("-if", "--inference_file", default=None, type=str,
                        help="Optional test set to run inferences on. (default None).")
    parser.add_argument("-tf", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-l", "--lemmatization", action="store_true",
                        help="Whether to use Lemmatization (default False)")
    parser.add_argument("-s", "--stemming", action="store_true",
                        help="Whether to use Stemming (default False).")
    parser.add_argument("-nr", "--no_reduce_words", action="store_true", default=False,
                        help="Do not reduce words by removing stopwords and only considering words in the english dictionary.")
    parser.add_argument("-n", "--ngram_range", type=int, default=1, 
                        help="The upper n-gram range. This includes n-grams in the range (1, n). (default 1)")
    parser.add_argument("-p", "--pos_tagging", action="store_true", default=False, 
                        help="Consider part-of-speech tagging .")
    parser.add_argument("-m", "--model_name", type=str, default='nb', help="The model to use. Can be one of ['nb', 'dt', 'rf', 'knn', 'svm']")
    parser.add_argument("-a", "--args", default=[], nargs='+', help="The arguments passed to the ML model")
    args = parser.parse_args()
    return args

def parse_values(values):
    """
    Parses the values of the classifier

    A value with a ' in it should be turned into a string
    A value with a . in it should be turned into a float
    A value of None should be turned into a None
    Everything else is treated as an integer
    
    Parameters
    ----------
        values : list
            The list of values to be parsed

    Returns
    -------
    A parsed set of values
    """
    values_ = []
    for value in values:
        if "x" in value:
            values_.append(value.replace("x", ""))
        elif "." in value:
            values_.append(float(value))
        elif value == "None":
            values_.append(None)
        else:
            values_.append(int(value))
    return values_

def my_preprocessor(doc, stemmer=None, lemmatizer=None, reduce_words=True, pos_tagging=False):
    """
    Used by the vectorizers to preprocesses the documents (tweets) from our data set.

    The following steps are performed:
    - Removal of words that contain numbers
    - Removal of special characters
    - Removal of words that are not part of the English dictionary
    - Removal of stopwords
    - Stemming or Lemmatization on each term
    
    Parameters
    ----------
        doc : string 
            Referring to a tweet, to be preprocessed
        stemmer : PorterStemmer
            The stemmer to be used (optional)
        lemmatizer : WordNetLemmatizer
            The lemmatizer to be used (optional)
        reduce_words : boolean
            Whether or not to reduce the vocabulary by excluding stopwords and words not in the english dictionary

    Returns
    -------
    A string to be used by the vectorizers
    """
    doc = doc.split()

    words = []
    for word in doc:
        demoji = emoji.demojize(word, delimiters=(" ", " "))
        # We split again because before demojize there could have been emoji's connected to words
        # without whitespace
        demoji = demoji.split()
        words += [word.lower() for word in demoji]
    
    with open("src/special_characters.txt") as f:
            special_characters = set(f.read().split("\n"))
    
    # Remove certain special characters such as quotes
    for c in special_characters:
        words = list(map(lambda word: word.replace(c, ""), words))

    # Remove strings that contain numbers, except words with hashtags
    words = list(filter(lambda word: (not any(char.isdigit() for char in word) or '#' in word), words))
    
    # Only consider nouns as possible features
    if pos_tagging:
        pos_tags = pos_tag(words)
        words = [tag[0] for tag in pos_tags if tag[-1] == "NN"]

    # TODO: Single words such as 'i' still appears in vocabulary_ for some reason  
    # Remove words that are shorter than 2 characters
    words = list(filter(lambda word: len(word) >= 2, words))

    if reduce_words:
        # Load in all the words from the english dictionary
        # from https://github.com/dwyl/english-words
        with open("src/english_wordlist.txt") as f:
            english_words = set(f.read().split("\n"))
        
        # Remove words that are not in the english language dictionary
        words = list(filter(lambda word: word in english_words, words))

        # Load in the stop words
        # from https://gist.github.com/rg089/35e00abf8941d72d419224cfd5b5925d
        with open("src/stopwords.txt") as f:
            stopwords = set(f.read().split("\n"))

        # Remove stopwords
        words = list(filter(lambda word: word not in stopwords, words))
   
    # Perform stemming or lemmatization
    if lemmatizer is not None:
        words = list(map(lambda word: lemmatizer.lemmatize(word), words))
    if stemmer is not None:
        words = list(map(lambda word: stemmer.stem(word), words))

    processed_tweet = " ".join(words)
    return processed_tweet

# TODO: remove later
def pos_tags_summary():
    import collections
    from nltk import pos_tag

    X_train, y_train = read_corpus("../data/train.tsv")
    pos_list = pos_tag(X_train[0].split())
    pos_counts = collections.Counter((subl[1] for subl in pos_list))
    for tweet in X_train[1:]:
        pos_list = pos_tag(tweet.split())
        pos_counts += collections.Counter((subl[1] for subl in pos_list))
    print(pos_counts)