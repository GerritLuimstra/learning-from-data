#!/usr/bin/env python

"""
This script allows for the training of LSTM models for offensive tweet
classification. All relevant settings can be specified via the command line.
Use the -h or --help flag to show a help message.
"""

import argparse
import logging
import pickle
import random as python_random
import sys
from typing import NamedTuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import (Bidirectional, Dense, Embedding, Input,
                                     LSTM, TextVectorization)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from utilities import read_tweets


class ModelSettings(NamedTuple):
    """Settings used to construct the model."""

    embeddings_file: str
    embedding_dimension: int
    trainable: bool
    learning_rate: float
    optimizer: str
    layers: int
    dropout: float
    recurrent_dropout: float
    bidirectional: bool


class TrainSettings(NamedTuple):
    """Settings used to train the model."""

    epochs: int
    batch_size: int
    patience: int
    verbose: int


def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", type=str,
                        default="../data/train_glove.tsv",
                        help="Input file to learn from")
    parser.add_argument("-vf", "--val_file", type=str,
                        default="../data/dev_glove.tsv",
                        help="Separate validation to use during training")
    parser.add_argument("-mf", "--model_file", type=str,
                        help="File where the trained model will be saved")
    parser.add_argument("-lf", "--log_file", type=str,
                        help="File where the logs will be saved")
    parser.add_argument("-s", "--seed", type=int,
                        help="If added, random seed to use for reproducible \
                        results")
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


def read_args():
    """Read the parsed command line arguments into the ModelSettings and
    TrainSettings classes."""

    args = create_arg_parser()
    model_settings = ModelSettings(args.embeddings_file,
                                   args.embedding_dimension, args.trainable,
                                   args.learning_rate, args.optimizer,
                                   args.layers, args.dropout,
                                   args.recurrent_dropout, args.bidirectional)
    train_settings = TrainSettings(args.epochs, args.batch_size, args.patience,
                                   args.verbose)
    return args, model_settings, train_settings


def set_seed(seed):
    """Set the seed to obtain reproducible results."""

    logging.info("Setting seed to %s", seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    python_random.seed(seed)


def get_embedding_matrix(vocabulary, embeddings):
    """Get embedding matrix given the vocabulary and embeddings."""

    # Construct an empty embedding matrix of the correct size.
    num_tokens = len(vocabulary)
    embedding_dim = len(embeddings["the"])  # Bit of a hack.
    embedding_matrix = np.zeros((num_tokens, embedding_dim))

    # Get an embedding vector for each word in the vocabulary.
    for index, word in enumerate(vocabulary):
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[index] = embedding_vector

    return embedding_matrix


def create_optimizer(name, learning_rate):
    """Create an optimizer object based on its name."""

    optimizer_dict = {"sgd": SGD, "adam": Adam}
    return optimizer_dict[name](learning_rate=learning_rate)


def create_lstm_layers(num_layers, units, dropout, recurrent_dropout):
    """Create a list of LSTM layers with the specified settings."""

    layers = []

    # Add n-1 LSTM layers that return sequences and one that does not.
    for _ in range(num_layers - 1):
        layers.append(LSTM(units, dropout=dropout,
                           recurrent_dropout=recurrent_dropout,
                           return_sequences=True))
    layers.append(LSTM(units, dropout=dropout,
                       recurrent_dropout=recurrent_dropout))

    return layers


def create_model(vectorizer, embedding_matrix, settings):
    """Create the LSTM model for offensive tweet classification."""

    # Get the embedding dimension and size from the embedding matrix.
    embedding_dim = len(embedding_matrix[0])
    num_tokens = len(embedding_matrix)

    # Start building the model.
    model = Sequential()

    # Add the input layer.
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorizer)

    # Add the embedding layer. Initialize it with uniform random values if no
    # embeddings file was provided.
    if settings.embeddings_file:
        initializer = Constant(embedding_matrix)
    else:
        initializer = "uniform"
    model.add(Embedding(num_tokens, embedding_dim,
                        embeddings_initializer=initializer,
                        trainable=settings.trainable))

    # Create the specified number of LSTM layers.
    lstm_layers = create_lstm_layers(settings.layers, embedding_dim,
                                     settings.dropout,
                                     settings.recurrent_dropout)

    # Make the LSTM layers bidirectional if specified.
    for layer in lstm_layers:
        if settings.bidirectional:
            model.add(Bidirectional(layer))
        else:
            model.add(layer)

    # End with a dense sigmoid layer for binary classification.
    model.add(Dense(1, input_dim=embedding_dim, activation="sigmoid"))

    # Compile the model using our settings, check for accuracy.
    optimizer = create_optimizer(settings.optimizer, settings.learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])

    return model


def train_model(model, x_train, y_train, x_val, y_val, settings):
    """Trains the model using the specified training and validation sets."""

    logging.info("Training model on %s samples", len(y_train))

    # Stop training when there are some number (3 by default) consecutive
    # epochs without improving.
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                patience=settings.patience)

    # Fit the model to our data.
    model.fit(x_train, y_train, verbose=settings.verbose,
              epochs=settings.epochs, callbacks=[callback],
              batch_size=settings.batch_size, validation_data=(x_val, y_val))

    return model


def main():
    """Main function to construct and train an LSTM model for offensive tweet
    classification."""

    # Read the command line arguments.
    args, model_settings, train_settings = read_args()

    # Set up logging if a log file was provided.
    if args.log_file:
        logging.basicConfig(filename=args.log_file,
                            format="%(asctime)s %(levelname)s %(message)s",
                            level=logging.DEBUG)
        logging.info("Running %s", ' '.join(sys.argv))

    # Make results reproducible if a seed was provided.
    if args.seed is not None:
        set_seed(args.seed)

    # Read in the data.
    x_train, y_train = read_tweets(args.train_file)
    x_val, y_val = read_tweets(args.val_file)

    # Read in the embeddings if specified, otherwise create dummy embeddings.
    if args.embeddings_file:
        logging.info("Reading embeddings from %s", args.embeddings_file)
        with open(args.embeddings_file, "rb") as ef:
            embeddings = pickle.load(ef)
    else:
        logging.info("Using randomized embeddings")
        embeddings = {"the": np.zeros(args.embedding_dimension)}

    # Transform words to indices using a vectorizer.
    vectorizer = TextVectorization(standardize=None,
                                   output_sequence_length=args.sequence_length)

    # Use train and dev to create the vocabulary.
    all_text = np.concatenate((x_train, x_val))
    text_dataset = tf.data.Dataset.from_tensor_slices(all_text)
    vectorizer.adapt(text_dataset)
    vocabulary = vectorizer.get_vocabulary()

    # Create the model.
    embedding_matrix = get_embedding_matrix(vocabulary, embeddings)
    model = create_model(vectorizer, embedding_matrix, model_settings)

    # Train the model and save it to a file.
    model = train_model(model, x_train, y_train, x_val, y_val, train_settings)
    if args.model_file:
        logging.info("Saving model to %s", args.model_file)
        model.save(f"{args.model_file}", save_format="tf")


if __name__ == "__main__":
    main()
