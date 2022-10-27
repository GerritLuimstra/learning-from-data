#!/usr/bin/env python

"""
This script allows for the evaluations of tweet offensiveness predictions. All
relevant settings can be specified via the command line. Use the -h or --help
 flag to show a help message.
"""

import argparse

import numpy as np
from sklearn.metrics import classification_report
from utilities import read_tweets


def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-lf", "--labels_file", type=str,
                        help="File containing true labels")
    parser.add_argument("-pf", "--predictions_file", type=str,
                        help="File containing predicted labels")
    args = parser.parse_args()
    return args


def main():
    """Main function to read true labels and preductions and evaluate."""

    # Read the command line arguments.
    args = create_arg_parser()

    # Read the true labels and predictions.
    _, labels = read_tweets(args.labels_file)
    predictions = np.rint(np.loadtxt(args.predictions_file))

    # Print results.
    print(classification_report(labels, predictions))


if __name__ == "__main__":
    main()
