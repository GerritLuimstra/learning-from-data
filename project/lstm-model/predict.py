#!/usr/bin/env python

"""
This script allows for the predicting of tweet offensiveness from a trained
model file. All relevant settings can be specified via the command line. Use
the -h or --help flag to show a help message.
"""

import argparse
import logging
import sys

import numpy as np
from tensorflow import keras
from utilities import read_tweets


def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-if", "--input_file", type=str,
                        help="Input file to predict labels for")
    parser.add_argument("-of", "--output_file", type=str,
                        help="Output file where predictions will be saved")
    parser.add_argument("-mf", "--model_file", type=str,
                        help="File from which the model will be loaded")
    parser.add_argument("-lf", "--log_file", type=str,
                        help="File where the logs will be saved")
    args = parser.parse_args()
    return args


def load_model(model_file):
    """Load a model from a file."""

    logging.info("Loading model from %s", model_file)
    return keras.models.load_model(model_file)


def predict(model, samples):
    """Use the given model to predict the offensiveness of the given tweets."""

    logging.info("Making predictions for %s samples", len(samples))
    return model.predict(samples).flatten()


def save_predictions(predictions, output_file):
    """Save predictions to a plain text file."""

    logging.info("Saving predictions to %s", output_file)
    np.savetxt(f"{output_file}", predictions)


def main():
    """Main function to load an LSTM model and use it to predict the
    offensiveness of tweets."""

    # Read the command line arguments.
    args = create_arg_parser()

    # Set up logging if a log file was provided.
    if args.log_file:
        logging.basicConfig(filename=args.log_file,
                            format="%(asctime)s %(levelname)s %(message)s",
                            level=logging.DEBUG)
        logging.info("Running %s", ' '.join(sys.argv))

    model = load_model(args.model_file)
    samples, _ = read_tweets(args.input_file)
    predictions = predict(model, samples)
    save_predictions(predictions, args.output_file)


if __name__ == "__main__":
    main()
