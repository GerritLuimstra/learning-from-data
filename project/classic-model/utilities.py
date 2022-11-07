import argparse
import numpy as np
import random

def read_tweets(corpus_file):
    """Read the tweets dataset and return tweets and toxicity labels."""
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1][:-1])
    return documents, labels

def parse_values(values):
    """Parses the values of the classifier"""
    values_ = []
    for value in values:
        if ":" in value:
            values_.append(value.replace(":", ""))
        elif "." in value:
            values_.append(float(value))
        elif value == "None":
            values_.append(None)
        else:
            values_.append(int(value))
    return values_

def set_seed(seed):
    """Set the seed to obtain reproducible results."""

    np.random.seed(seed)
    random.seed(seed)
