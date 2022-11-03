#!/usr/bin/env python

"""
This script allows for the evaluations of tweet offensiveness predictions. All
relevant settings can be specified via the command line. Use the -h or --help
 flag to show a help message.
"""

import argparse
import logging
import os
import sys

import numpy as np
from sklearn.metrics import classification_report, f1_score
from utilities import read_tweets


def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--true_file", type=str,
                        help="File containing true labels")
    parser.add_argument("-of", "--output_file", type=str,
                        help="Output file where macro F1 scores will be saved")
    parser.add_argument("-pd", "--predictions_directory", type=str,
                        help="Directory of files with predicted labels")
    parser.add_argument("-lf", "--log_file", type=str,
                        help="File where the logs will be saved")
    args = parser.parse_args()
    return args


def main():
    """Main function to read true labels and predictions and evaluate."""

    scores = []

    # Read the command line arguments.
    args = create_arg_parser()

    # Set up logging if a log file was provided.
    if args.log_file:
        logging.basicConfig(filename=args.log_file,
                            format="%(asctime)s %(levelname)s %(message)s",
                            level=logging.DEBUG)
        logging.info("Running %s", ' '.join(sys.argv))

    # Read the true labels.
    logging.info("Reading true labels from %s", args.true_file)
    _, labels = read_tweets(args.true_file)

    # Find the files containing predictions.
    predictions_files = os.listdir(args.predictions_directory)
    predictions_files.sort()

    # Read the predictions.
    for predictions_file in predictions_files:
        predictions_file = f"{args.predictions_directory}/{predictions_file}"
        logging.info("Reading predictions from %s", predictions_file)
        predictions = np.rint(np.loadtxt(predictions_file))

        scores.append(f1_score(labels, predictions, average="macro"))

        # Print individual results.
        print(classification_report(labels, predictions))

    # Save scores to a file.
    scores = np.asarray(scores)
    np.savetxt(f"{args.output_file}", scores)

    # Print averaged results.
    average = np.average(scores)
    std = np.std(scores)
    results = f"Macro F1 = {average:.3f} ± {std:.3f}"
    logging.info(results)
    print(results)


if __name__ == "__main__":
    main()
