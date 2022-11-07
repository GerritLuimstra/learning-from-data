import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
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
    args = parser.parse_args()
    return args

def main():
    """Main function to read true labels and predictions and evaluate."""

    scores = []

    # Read the command line arguments.
    args = create_arg_parser()

    # Read the true labels.
    _, labels = read_tweets(args.true_file)

    # Find the files containing predictions.
    predictions_files = os.listdir(args.predictions_directory)
    predictions_files.sort()

    # Read the predictions.
    store = True
    for predictions_file in predictions_files:
        predictions_file = f"{args.predictions_directory}/{predictions_file}"
        predictions = np.loadtxt(predictions_file, dtype=str)

        scores.append(f1_score(labels, predictions, average="macro"))

        # Print individual results.
        print(classification_report(labels, predictions))

        # Store the confusion matrix of the first test model
        if store and "test" in args.predictions_directory:
            cm = confusion_matrix(labels, predictions, labels=['NOT', 'OFF'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NOT', 'OFF'])
            disp.plot()
            plt.savefig(f"{os.path.dirname(args.predictions_directory)}/test_confusion_matrix.png")
            store = False

    # Save scores to a file.
    scores = np.asarray(scores)
    np.savetxt(f"{args.output_file}", scores)

    # Print averaged results.
    average = np.average(scores)
    std = np.std(scores)
    results = f"Macro F1 = {average:.3f} ± {std:.3f}"
    print(results)



if __name__ == "__main__":
    main()