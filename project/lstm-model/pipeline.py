#!/usr/bin/env python

"""
This script runs train.py, predict.py, and evaluate.py. All relevant settings
can be specified via the command line. Use the -h or --help flag to show a help
message.
"""

import argparse
import os
import time

train_file = "../data/train_glove.tsv"
val_file = "../data/dev_glove.tsv"
test_file = "../data/test_glove.tsv"


def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str,
                        default=time.strftime("%Y%m%d-%H%M%S"),
                        help="The name for this experiment")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="The number of runs for this experiment")
    parser.add_argument("-t", "--test", default=False, action="store_true",
                        help="Whether to predict and evaluate the test set")
    parser.add_argument("-ow", "--overwrite", default=False,
                        action="store_true", help="Whether to overwrite \
                        previous results")
    parser.add_argument("-sl", "--sequence_length", type=int, default=50,
                        help="Number of tokens before the input is cut off")
    parser.add_argument("-ef", "--embeddings_file", type=str,
                        help="If added, pre-trained embedding file to use")
    parser.add_argument("-ed", "--embedding_dimension", type=int, default=300,
                        help="Dimension of the embeddings, ignored if a \
                        embedding file is specified")
    parser.add_argument("-tr", "--trainable", default=False,
                        action="store_true", help="Whether the embedding \
                        layer is updated during training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3,
                        help="Learning rate used to train the model")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "adam"],
                        default="adam", help="Optimizer used to train the \
                        model")
    parser.add_argument("-ly", "--layers", type=int, default=1,
                        help="Number of LSTM layers in the model")
    parser.add_argument("-do", "--dropout", type=float, default=0,
                        help="Dropout fraction to use for linear \
                        transformation of inputs in LSTM")
    parser.add_argument("-rdo", "--recurrent_dropout", type=float, default=0,
                        help="Dropout fraction to use for linear \
                        transformation of recurrent states in LSTM")
    parser.add_argument("-bi", "--bidirectional", default=False,
                        action="store_true", help="If added, use \
                        bidirectional LSTM layers in the model")
    parser.add_argument("-e", "--epochs", type=int, default=5,
                        help="Maximum number of epochs to train the model for")
    parser.add_argument("-p", "--patience", type=int, default=3,
                        help="Number of epochs with no improvement before \
                        training is stopped early")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="Batch size used to train the model")
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2],
                        default=1, help="Verbosity level of the model \
                        training process")
    args = parser.parse_args()
    return args


def train(args, directory, run):
    """Run train.py with the specified arguments and in the given experiment
    directory."""

    # Skip if we already have a trained model and are not overwriting.
    previous_exists = os.path.exists(f"{directory}/models/{run}")
    if not args.overwrite and previous_exists:
        return

    # Train a new model.
    args = get_train_args(args, directory, run)
    os.system(f"python3 train.py {args}")


def get_train_args(args, directory, run):
    """Get command line arguments for train.py."""

    train_args = ""

    train_args += f"-tf {train_file} "
    train_args += f"-vf {val_file} "
    train_args += f"-mf {directory}/models/{run} "
    train_args += f"-lf {directory}/log/train.log "
    train_args += f"-s {run} "
    train_args += f"-sl {args.sequence_length} "
    train_args += f"-ed {args.embedding_dimension} "
    train_args += f"-lr {args.learning_rate} "
    train_args += f"-o {args.optimizer} "
    train_args += f"-ly {args.layers} "
    train_args += f"-do {args.dropout} "
    train_args += f"-rdo {args.recurrent_dropout} "
    train_args += f"-e {args.epochs} "
    train_args += f"-p {args.patience} "
    train_args += f"-b {args.batch_size} "
    train_args += f"-v {args.verbose} "

    if args.embeddings_file:
        train_args += f"-ef {args.embeddings_file} "
    if args.trainable:
        train_args += "-tr "
    if args.bidirectional:
        train_args += "-bi "

    return train_args


def predict(args, directory, run):
    """Run predict.py with the specified arguments and in the given experiment
    directory."""

    # Only predict if there are no predictions yet or we are overwriting.
    previous_exists = os.path.exists(f"{directory}/out/{run}.out")
    if args.overwrite or not previous_exists:
        # Get predictions on dev.
        predict_args = get_predict_args(directory, run, False)
        os.system(f"python3 predict.py {predict_args}")

    # Skip predicting on test set if not requested.
    if not args.test:
        return

    # Only predict if there are no predictions yet or we are overwriting.
    previous_exists = os.path.exists(f"{directory}/test-out/{run}.out")
    if args.overwrite or not previous_exists:
        # Get predictions on test.
        predict_args = get_predict_args(directory, run, True)
        os.system(f"python3 predict.py {predict_args}")


def get_predict_args(directory, run, test):
    """Get command line arguments for predict.py."""

    predict_args = ""

    if test:
        predict_args += f"-if {test_file} "
        predict_args += f"-of {directory}/test-out/{run}.out "
    else:
        predict_args += f"-if {val_file} "
        predict_args += f"-of {directory}/out/{run}.out "

    predict_args += f"-mf {directory}/models/{run} "
    predict_args += f"-lf {directory}/log/predict.log "

    return predict_args

<<<<<<< HEAD
def get_evaluate_args(directory):
=======

def evaluate(args, directory):
    """Run evaluate.py with the specified arguments and in the given experiment
    directory."""

    # Get scores on dev.
    evaluate_args = get_evaluate_args(directory, False)
    os.system(f"python3 evaluate.py {evaluate_args}")

    # Skip evaluating on test set if not requested.
    if not args.test:
        return

    # Get scores on test.
    evaluate_args = get_evaluate_args(directory, True)
    os.system(f"python3 evaluate.py {evaluate_args}")


def get_evaluate_args(directory, test):
>>>>>>> ae293e70ade18d618f1b5c36e41fb4791a34472d
    """Get command line arguments for evaluate.py."""

    evaluate_args = ""

    if test:
        evaluate_args += f"-tf {test_file} "
        evaluate_args += f"-of {directory}/test_scores.txt "
        evaluate_args += f"-pd {directory}/test-out "
    else:
        evaluate_args += f"-tf {val_file} "
        evaluate_args += f"-of {directory}/scores.txt "
        evaluate_args += f"-pd {directory}/out "

    evaluate_args += f"-lf {directory}/log/evaluate.log "

    return evaluate_args


def main():
    """Main function to run the pipeline."""

    # Read the command line arguments.
    args = create_arg_parser()

    # Create required directories.
    experiments_directory = f"results/{args.name}"
    os.makedirs(experiments_directory, exist_ok=True)
    os.makedirs(f"{experiments_directory}/log", exist_ok=True)
    os.makedirs(f"{experiments_directory}/out", exist_ok=True)
    os.makedirs(f"{experiments_directory}/test-out", exist_ok=True)

    # Run the pipeline a given number of times.
    for run in range(args.runs):
        train(args, experiments_directory, run)
        predict(args, experiments_directory, run)

    # Evaluate all runs.
    evaluate(args, experiments_directory)


if __name__ == "__main__":
    main()
