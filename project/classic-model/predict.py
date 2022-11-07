import argparse
import numpy as np

from utilities import read_tweets
from joblib import dump, load

def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-if", "--input_file", type=str,
                        help="Input file to predict labels for")
    parser.add_argument("-of", "--output_file", type=str,
                        help="Output file where predictions will be saved")
    parser.add_argument("-mf", "--model_file", type=str,
                        help="File from which the model will be loaded")
    args = parser.parse_args()
    return args

def load_model(model_file):
    """Load the classifier and the vectorizer"""
    model_vec = load(model_file)
    return model_vec[0], model_vec[1]

def save_predictions(predictions, output_file):
    """Save predictions to a plain text file."""

    np.savetxt(f"{output_file}", predictions, delimiter=" ", fmt="%s")

def main():
    """Main function to load a classifier and use it to predict the
    offensiveness of tweets."""

    # Read the command line arguments.
    args = create_arg_parser()

    # Load the classifier and vectorizer
    classifier, vec = load_model(args.model_file)
    
    # Load in the samples for prediction
    samples, labels = read_tweets(args.input_file)

    # Transform the data 
    samples = vec.transform(samples)

    # Predict the labels
    predictions = classifier.predict(samples)

    # Save the predictions to the output file
    save_predictions(predictions, args.output_file)

if __name__ == "__main__":
    main()
