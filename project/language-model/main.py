#!/usr/bin/env python

"""
This script allows for the training and testing of pretained language models. 
All relevant settings can be specified via the command line.
Use the -h or --help flag to show a help message.
"""

import argparse
import json
import random as python_random
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                             confusion_matrix)
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import roc_curve, auc

# Make results reproducible.
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


class ModelSettings(NamedTuple):
    """Settings used to construct the model."""

    trainable: bool
    learning_rate: float
    use_lr_decay: bool
    optimizer: str


class TrainSettings(NamedTuple):
    """Settings used to train the model."""

    epochs: int
    batch_size: int
    patience: int
    verbose: int
    loss_plot: str
    confusion_matrix: bool


def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default="../data/train.tsv",
                        type=str, help="Input file to learn from")
    parser.add_argument("-d", "--dev_file", type=str, default="../data/dev.tsv",
                        help="Separate dev set to read in")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test \
                        set")
    parser.add_argument("-lm", "--language_model", type=str,
                        help="If added, the pre-trained language model to use")
    parser.add_argument("-s", "--sequence_length", type=int, default=50,
                        help="Number of tokens before the input is cut off")
    parser.add_argument("-tr", "--trainable", default=False,
                        action="store_true", help="Whether the embedding \
                        layer is updated during training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-3,
                        help="Learning rate used to train the model")
    parser.add_argument("-ld", "--use_lr_decay", default=False,
                        action="store_true", help="Whether to use Exponential \
                        LR decay")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "adam"],
                        default="sgd", help="Optimizer used to train the \
                        model")
    parser.add_argument("-ep", "--epochs", type=int, default=10,
                        help="Maximum number of epochs to train the model for")
    parser.add_argument("-p", "--patience", type=int, default=3,
                        help="Number of epochs with no improvement before \
                        training is stopped early")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="Batch size used to train the model")
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2],
                        default=1, help="Verbosity level of the model \
                        training process")
    parser.add_argument("-lp", "--loss_plot", type=str,
                        help="If added, file name of loss curve plot")
    parser.add_argument("-cm", "--confusion_matrix", default=False,
                        action="store_true", help="If added, plot and save a \
                        confusion matrix")
    args = parser.parse_args()
    return args


def read_args():
    """Read the parsed command line arguments into the ModelSettings and
    TrainSettings classes."""

    args = create_arg_parser()
    model_settings = ModelSettings(args.trainable, args.learning_rate,
                                   args.use_lr_decay, args.optimizer)
    train_settings = TrainSettings(args.epochs, args.batch_size, args.patience,
                                   args.verbose, args.loss_plot,
                                   args.confusion_matrix)
    return args, model_settings, train_settings


def read_tweets(corpus_file):
    """Read in tweets dataset and return tweets and toxicity labels."""
    tweets = []
    labels = []
    with open(corpus_file, encoding="utf-8") as corpus:
        for line in corpus:
            tweet = line[:-4].strip()
            label = line.strip()[-3:]
            tweets.append(tweet)
            labels.append(label)
    return tweets, labels


def create_optimizer(name, learning_rate):
    """Create an optimizer object based on its name."""
    optimizer_dict = {"sgd": SGD, "adam": Adam}
    return optimizer_dict[name](learning_rate=learning_rate)

def create_language_model(model_name, settings):
    """Create the pre-trained language model to use."""

    # Load the pre-trained language model.
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Decide whether we want to fine-tune the language model.
    model.layers[0].trainable = settings.trainable

    # Compile the model using our settings, check for accuracy.
    loss_function = BinaryCrossentropy(from_logits=True)

    if settings.use_lr_decay:
        lr_schedule = ExponentialDecay(
            initial_learning_rate=settings.learning_rate,
            decay_steps=100,
            decay_rate=0.9
        )
        optimizer = create_optimizer(settings.optimizer, lr_schedule)
    else:
        optimizer = create_optimizer(settings.optimizer,
                                     settings.learning_rate)

    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=["accuracy"])
    print(model.summary())

    return model


def plot_loss(history, file_name):
    """Plot a loss curve from the given model history."""

    train_loss = history["loss"]
    val_loss = history["val_loss"]
    plt.title("Train and Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="train")
    plt.plot(range(1, len(train_loss) + 1), val_loss, label="dev")
    plt.legend()
    plt.savefig(file_name, bbox_inches="tight", format="pdf")


def train_model(model, X_train, Y_train, X_dev, Y_dev, settings, labels):
    """Trains the model using the specified training and validation sets."""

    # Stop training when there are some number (3 by default) consecutive
    # epochs without improving.
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                patience=settings.patience, 
                                                restore_best_weights=True)

    # Fit the model to our data.
    history = model.fit(X_train, Y_train, verbose=settings.verbose,
                        epochs=settings.epochs, callbacks=[callback],
                        batch_size=settings.batch_size,
                        validation_data=(X_dev, Y_dev))

    # Plot a loss curve if specified.
    if settings.loss_plot:
        plot_loss(history.history, settings.loss_plot)

    # Print the final accuracy for the model.
    test_set_predict(model, X_dev, Y_dev, "dev", labels,
                     settings.confusion_matrix)
    return model


def test_set_predict(model, X_test, Y_test, ident, labels, plot_cm):
    """Do predictions and measure accuracy on the given test set."""

    # Get predictions using the trained model.
    Y_pred = tf.round(tf.nn.sigmoid(model.predict(X_test)["logits"]))

    # Print a classification report.
    print(f"Classification results on {ident} set:")
    print(classification_report(Y_test, Y_pred, target_names=labels))
    fpr, tpr, _ = roc_curve(Y_test, Y_pred, pos_label=1)
    print("AUC", auc(fpr, tpr))

    # Save a confusion matrix if specified.
    if plot_cm:
        cm = confusion_matrix(Y_pred, Y_test)
        ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=labels).plot()
        plt.savefig(f"confusion_matrix_{ident}.pdf", bbox_inches="tight",
                    format="pdf")


def run_language_model(args, model_settings, train_settings):
    """Constructs, trains, and runs the specified pre-trained language
    model."""

    # Read in the data.
    X_train, Y_train = read_tweets(args.train_file)
    X_dev, Y_dev = read_tweets(args.dev_file)

    # Transform labels into binary
    Y_train_bin = np.array([1 if l == "OFF" else 0 for l in Y_train])
    Y_dev_bin = np.array([1 if l == "OFF" else 0 for l in Y_dev])

    # Create the model.
    model = create_language_model(args.language_model, model_settings)

    # Tokenize the input.
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokens_train = tokenizer(X_train, padding=True,
                             max_length=args.sequence_length, truncation=True,
                             return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True,
                           max_length=args.sequence_length,
                           truncation=True, return_tensors="np").data

    # Train the model.
    model = train_model(model, tokens_train, Y_train_bin, tokens_dev,
                        Y_dev_bin, train_settings, ["NOT", "OFF"])

    # If specified, do predictions on the test set.
    if args.test_file:
        
        # Read in test set and tokenize.
        X_test, Y_test = read_tweets(args.test_file)
        Y_test_bin = np.array([1 if l == "OFF" else 0 for l in Y_test])
        tokens_test = tokenizer(X_test, padding=True,
                                max_length=args.sequence_length,
                                truncation=True, return_tensors="np").data

        # Do the predictions.
        test_set_predict(model, tokens_test, Y_test_bin, "test",
                         ["NOT", "OFF"], train_settings.confusion_matrix)


def main():
    """Main function to train and test neural network given command line
    arguments."""

    # Read the command line arguments.
    args, model_settings, train_settings = read_args()

    # Run the model
    run_language_model(args, model_settings, train_settings)


if __name__ == "__main__":
    main()
