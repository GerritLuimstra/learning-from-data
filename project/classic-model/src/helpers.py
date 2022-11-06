"""
This script provides helper functions for the main program
"""

import argparse
import numpy as np

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
                        help="File that will be used to perform training.")
    parser.add_argument("-d", "--dev_file", type=str,
                        help="File that will be used to perform testing.")
    parser.add_argument("-if", "--inference_file", default=None, type=str,
                        help="Optional test set to run inferences on. (default None).")
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